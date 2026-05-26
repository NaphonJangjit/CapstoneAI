package naphon.capstone.ai.service;

import naphon.capstone.ai.object.Chapter;

import java.util.*;

/**
 * Storage and similarity search for chapter embeddings.
 *
 * Implementations: {@link InMemoryVectorStore} (dev/test),
 * {@link SolrVectorStore} (production — Apache Solr with dense vector KNN).
 */
public interface VectorStore extends AutoCloseable {

    /** Composite key for a chapter. */
    record Key(long novelId, long chapter) {
        public static Key of(long novelId, long chapter) { return new Key(novelId, chapter); }
        @Override public String toString() { return novelId + "/" + chapter; }
    }

    /** Metadata + embedding for one chapter. */
    class Entry {
        private final long novelId;
        private final long chapter;
        private final String novelName;
        private final String chapterName;
        private final List<String> tags;
        private final float[] vector;

        public Entry(long novelId, long chapter, String novelName, String chapterName,
                     List<String> tags, float[] vector) {
            this.novelId = novelId;
            this.chapter = chapter;
            this.novelName = novelName;
            this.chapterName = chapterName;
            this.tags = List.copyOf(tags);
            this.vector = vector;
        }

        public long novelId() { return novelId; }
        public long chapter() { return chapter; }
        public String novelName() { return novelName; }
        public String chapterName() { return chapterName; }
        public List<String> tags() { return tags; }
        public float[] vector() { return vector; }

        public Chapter toChapter() {
            return new Chapter(novelId, chapter, chapterName, vector,
                    tags.toArray(new String[0]), "");
        }
    }

    /** A search result with similarity score. */
    record ScoredEntry(Entry entry, double score) {}

    // ---- Write ops ----

    /** Index (or replace) a chapter with its embedding. */
    void put(long novelId, long chapter,
             String novelName, String chapterName,
             List<String> tags, float[] vector);

    /** Remove a specific chapter. Returns true if it existed. */
    boolean remove(long novelId, long chapter);

    /** Remove all chapters of a novel. Returns count removed. */
    int removeAll(long novelId);

    // ---- Read ops ----

    /** Get a specific chapter entry, or null. */
    Entry get(long novelId, long chapter);

    /** Get all chapters of a novel, ordered by chapter number. */
    List<Entry> getByNovel(long novelId);

    /**
     * Average embedding across all chapters of a novel (L2-normalized),
     * or null if the novel has no indexed chapters.
     */
    float[] getNovelAverage(long novelId);

    /**
     * K-nearest neighbors by cosine similarity.
     * @param excludeKeys entries matching these keys are excluded from results.
     */
    List<ScoredEntry> search(float[] queryVector, int k, Set<Key> excludeKeys);

    default List<ScoredEntry> search(float[] queryVector, int k) {
        return search(queryVector, k, Set.of());
    }

    /** Number of indexed chapters. */
    int size();

    @Override
    default void close() throws Exception {}
}
