package naphon.capstone.ai.service;

import naphon.capstone.ai.object.Chapter;

import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * Business logic coordinating embedding and vector storage.
 *
 * All methods that touch the AI model are async (return CompletableFuture).
 * Methods that only touch the VectorStore are synchronous (thread-safe).
 */
public class NovelService {

    private final EmbeddingService embedder;
    private final VectorStore store;

    public NovelService(EmbeddingService embedder, VectorStore store) {
        this.embedder = embedder;
        this.store = store;
    }

    // ---- Indexing ----------------------------------------------------------

    /**
     * Embed a chapter's content and index it in the vector store.
     * Returns the embedded Chapter object (without the vector — metadata only).
     */
    public CompletableFuture<Chapter> indexChapter(long novelId, long chapter,
                                                    String novelName, String chapterName,
                                                    String content, List<String> tags) {
        return embedder.embed(content).thenApply(vector -> {
            store.put(novelId, chapter, novelName, chapterName, tags, vector);
            return new Chapter(novelId, chapter, chapterName, vector, tags.toArray(new String[0]), "");
        });
    }

    // ---- Search ------------------------------------------------------------

    /**
     * Search for novels similar to a given novel by querying each chapter
     * individually and aggregating the results (best score per novel).
     *
     * <p>This per-chapter approach preserves the semantic richness of each
     * chapter rather than collapsing everything into a single average vector.
     */
    public CompletableFuture<List<VectorStore.ScoredEntry>> searchByNovel(long novelId, int topK) {
        List<VectorStore.Entry> chapters = store.getByNovel(novelId);
        if (chapters.isEmpty()) {
            return CompletableFuture.completedFuture(List.of());
        }

        // Exclude all chapters from the query novel itself
        Set<VectorStore.Key> exclude = new HashSet<>();
        for (var ch : chapters) {
            exclude.add(new VectorStore.Key(ch.novelId(), ch.chapter()));
        }

        // Search with each chapter's vector in parallel
        List<CompletableFuture<List<VectorStore.ScoredEntry>>> futures = new ArrayList<>(chapters.size());
        for (var ch : chapters) {
            futures.add(CompletableFuture.supplyAsync(() ->
                    store.search(ch.vector(), topK, exclude)));
        }

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> {
                    // Aggregate: keep the highest-scoring result per novel
                    Map<Long, VectorStore.ScoredEntry> bestByNovel = new LinkedHashMap<>();
                    for (var future : futures) {
                        for (var se : future.join()) {
                            long nid = se.entry().novelId();
                            VectorStore.ScoredEntry existing = bestByNovel.get(nid);
                            if (existing == null || se.score() > existing.score()) {
                                bestByNovel.put(nid, se);
                            }
                        }
                    }
                    return bestByNovel.values().stream()
                            .sorted(Comparator.comparingDouble(VectorStore.ScoredEntry::score).reversed())
                            .limit(topK)
                            .toList();
                });
    }

    /**
     * Search for chapters similar to a specific chapter.
     */
    public CompletableFuture<List<VectorStore.ScoredEntry>> searchByChapter(long novelId, long chapter, int topK) {
        VectorStore.Entry entry = store.get(novelId, chapter);
        if (entry == null) {
            return CompletableFuture.completedFuture(List.of());
        }
        return CompletableFuture.supplyAsync(() -> {
            Set<VectorStore.Key> exclude = Set.of(new VectorStore.Key(novelId, chapter));
            return store.search(entry.vector(), topK, exclude);
        });
    }

    /**
     * Search for chapters similar to a free-text keyword query.
     * The keyword is embedded first, then the vector is used for KNN search.
     */
    public CompletableFuture<List<VectorStore.ScoredEntry>> searchByKeyword(String keyword, int topK) {
        return embedder.embed(keyword).thenApply(vector ->
                store.search(vector, topK));
    }

    // ---- Delete ------------------------------------------------------------

    public boolean deleteChapter(long novelId, long chapter) {
        return store.remove(novelId, chapter);
    }

    public int deleteNovel(long novelId) {
        return store.removeAll(novelId);
    }

    // ---- Health ------------------------------------------------------------

    public int getIndexedCount() {
        return store.size();
    }
}
