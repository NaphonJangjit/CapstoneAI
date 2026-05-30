package naphon.capstone.ai.service;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.impl.Http2SolrClient;
import org.apache.solr.client.solrj.request.UpdateRequest;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.client.solrj.response.UpdateResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.apache.solr.common.params.SolrParams;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Apache Solr-backed vector store using dense-vector ANN search (HNSW by default).
 *
 * <h3>Required Solr schema</h3>
 * <pre>{@code
 * curl -X POST http://localhost:8983/solr/chapters/schema -H 'Content-Type: application/json' -d '{
 *   "add-field-type": {
 *     "name": "knn_vector_256",
 *     "class": "solr.DenseVectorField",
 *     "vectorDimension": 256,
 *     "similarityFunction": "cosine"
 *   },
 *   "add-field": [
 *     {"name": "id",          "type": "string",  "indexed": true, "stored": true, "required": true},
 *     {"name": "novel_id",    "type": "plong",   "indexed": true, "stored": true},
 *     {"name": "chapter",     "type": "plong",   "indexed": true, "stored": true},
 *     {"name": "novel_name",  "type": "string",  "indexed": true, "stored": true},
 *     {"name": "chapter_name","type": "string",  "indexed": true, "stored": true},
 *     {"name": "content",     "type": "text_general", "indexed": true, "stored": true},
 *     {"name": "tags",        "type": "strings", "indexed": true, "stored": true, "multiValued": true},
 *     {"name": "embedding",   "type": "knn_vector_256", "indexed": true, "stored": true}
 *   ],
 *   "add-unique-key": {"name": "id"}
 * }'
 * }</pre>
 *
 * <h3>Config</h3>
 * {@code SOLR_URL} env var (default: {@code http://localhost:8983/solr/chapters}).
 */
public class SolrVectorStore implements VectorStore {

    private final SolrClient solr;
    private final int dim;

    public SolrVectorStore(String baseUrl, int dim) {
        this.solr = new Http2SolrClient.Builder(baseUrl).build();
        this.dim = dim;
    }

    public SolrVectorStore(String baseUrl) {
        this(baseUrl, 256);
    }

    // ---- Write ops ---------------------------------------------------------

    @Override
    public void put(long novelId, long chapter,
                    String novelName, String chapterName,
                    List<String> tags, float[] vector) {
        String docId = makeId(novelId, chapter);

        SolrInputDocument doc = new SolrInputDocument();
        doc.setField("id", docId);
        doc.setField("novel_id", novelId);
        doc.setField("chapter", chapter);
        doc.setField("novel_name", novelName);
        doc.setField("chapter_name", chapterName);
        doc.setField("tags", tags);
        doc.setField("embedding", toList(vector));

        try {
            UpdateRequest req = new UpdateRequest();
            req.add(doc);
            req.setCommitWithin(500); // near-real-time
            req.process(solr);
        } catch (SolrServerException | IOException e) {
            throw new RuntimeException("Solr indexing failed", e);
        }
    }

    @Override
    public boolean remove(long novelId, long chapter) {
        try {
            UpdateResponse resp = solr.deleteById(makeId(novelId, chapter));
            solr.commit();
            return resp.getStatus() == 0;
        } catch (SolrServerException | IOException e) {
            throw new RuntimeException("Solr delete failed", e);
        }
    }

    @Override
    public int removeAll(long novelId) {
        try {
            UpdateResponse resp = solr.deleteByQuery("novel_id:" + novelId);
            solr.commit();
            // Solr doesn't return deleted count from deleteByQuery easily;
            // status 0 means success. Return a sentinel.
            return resp.getStatus() == 0 ? -1 : 0;
        } catch (SolrServerException | IOException e) {
            throw new RuntimeException("Solr deleteByQuery failed", e);
        }
    }

    // ---- Read ops ----------------------------------------------------------

    @Override
    public Entry get(long novelId, long chapter) {
        try {
            SolrDocument doc = solr.getById(makeId(novelId, chapter));
            return doc != null ? toEntry(doc) : null;
        } catch (SolrServerException | IOException e) {
            throw new RuntimeException("Solr get failed", e);
        }
    }

    @Override
    public List<Entry> getByNovel(long novelId) {
        try {
            SolrQuery q = new SolrQuery("novel_id:" + novelId);
            q.setRows(Integer.MAX_VALUE);
            q.setSort("chapter", SolrQuery.ORDER.asc);
            QueryResponse resp = solr.query(q);
            return resp.getResults().stream()
                    .map(this::toEntry)
                    .collect(Collectors.toList());
        } catch (SolrServerException | IOException e) {
            throw new RuntimeException("Solr getByNovel failed", e);
        }
    }

    @Override
    public List<ScoredEntry> search(float[] queryVector, int k, Set<Key> excludeKeys) {
        // Build KNN query: {!knn f=embedding topK=K}[v0,v1,...,vN]
        StringBuilder knn = new StringBuilder("{!knn f=embedding topK=").append(k).append("}");
        knn.append(Arrays.toString(queryVector));

        SolrQuery q = new SolrQuery(knn.toString());
        q.setFields("id", "novel_id", "chapter", "novel_name", "chapter_name",
                "tags", "embedding", "score");

        // Exclude specific doc IDs
        if (!excludeKeys.isEmpty()) {
            String excludeFilter = excludeKeys.stream()
                    .map(key -> makeId(key.novelId(), key.chapter()))
                    .collect(Collectors.joining(" OR ", "-id:(", ")"));
            q.addFilterQuery(excludeFilter);
        }

        try {
            QueryResponse resp = solr.query(q);
            return resp.getResults().stream()
                    .map(doc -> {
                        Entry entry = toEntry(doc);
                        double score = asDouble(doc.getFieldValue("score"));
                        return new ScoredEntry(entry, score);
                    })
                    .collect(Collectors.toList());
        } catch (SolrServerException | IOException e) {
            throw new RuntimeException("Solr KNN search failed", e);
        }
    }

    @Override
    public int size() {
        try {
            SolrQuery q = new SolrQuery("*:*");
            q.setRows(0);
            return (int) solr.query(q).getResults().getNumFound();
        } catch (SolrServerException | IOException e) {
            return 0;
        }
    }

    @Override
    public void close() throws IOException {
        solr.close();
    }

    // ---- Internal helpers --------------------------------------------------

    static String makeId(long novelId, long chapter) {
        return novelId + "_" + chapter;
    }

    private static long asLong(Object val) {
        if (val instanceof Number n) return n.longValue();
        if (val instanceof List<?> list && !list.isEmpty() && list.getFirst() instanceof Number n)
            return n.longValue();
        return 0L;
    }

    private static String asString(Object val) {
        if (val instanceof String s) return s;
        if (val instanceof List<?> list && !list.isEmpty() && list.getFirst() instanceof String s)
            return s;
        return val != null ? val.toString() : "";
    }

    private static double asDouble(Object val) {
        if (val instanceof Number n) return n.doubleValue();
        if (val instanceof List<?> list && !list.isEmpty() && list.getFirst() instanceof Number n)
            return n.doubleValue();
        return 0.0;
    }

    private Entry toEntry(SolrDocument doc) {
        long novelId = asLong(doc.getFieldValue("novel_id"));
        long chapter = asLong(doc.getFieldValue("chapter"));
        String novelName = asString(doc.getFieldValue("novel_name"));
        String chapterName = asString(doc.getFieldValue("chapter_name"));

        @SuppressWarnings("unchecked")
        List<String> tags = (List<String>) doc.getFieldValue("tags");
        if (tags == null) tags = List.of();

        float[] vector = extractVector(doc.getFieldValue("embedding"));
        return new Entry(novelId, chapter, novelName, chapterName, tags, vector);
    }

    @SuppressWarnings("unchecked")
    private float[] extractVector(Object fieldValue) {
        if (fieldValue == null) return new float[dim];
        if (fieldValue instanceof List<?> list) {
            float[] v = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                v[i] = ((Number) list.get(i)).floatValue();
            }
            return v;
        }
        if (fieldValue instanceof float[] fa) return fa;
        return new float[dim];
    }

    private static List<Float> toList(float[] v) {
        List<Float> list = new ArrayList<>(v.length);
        for (float f : v) list.add(f);
        return list;
    }
}
