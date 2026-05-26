package naphon.capstone.ai;

import naphon.capstone.ai.api.HttpChannelInboundHandler;
import naphon.capstone.ai.api.NettyHttp;
import naphon.capstone.ai.service.EmbeddingService;
import naphon.capstone.ai.service.NovelService;
import naphon.capstone.ai.service.ThreadPoolEmbeddingService;
import naphon.capstone.ai.service.SolrVectorStore;
import naphon.capstone.ai.service.VectorStore;

/**
 * Production entry point.
 *
 * Architecture:
 * <pre>
 *   Netty I/O threads (never block)
 *        │
 *        ▼  CompletableFuture
 *   AI worker pool (CPU-bound ONNX inference)
 *        │
 *        ▼
 *   SolrVectorStore (Apache Solr dense-vector KNN)
 * </pre>
 */
public class Main {
    // Solr is running on Docker
    public static void main(String[] args) throws Exception {
        // ---- Configuration -------------------------------------------------
        String host = System.getenv().getOrDefault("CAPSTONE_HOST", "0.0.0.0");
        int port = Integer.parseInt(System.getenv().getOrDefault("CAPSTONE_PORT", "9922"));
        int aiThreads = Integer.parseInt(System.getenv().getOrDefault("CAPSTONE_AI_THREADS",
                String.valueOf(Runtime.getRuntime().availableProcessors())));
        int maxSeqLen = Integer.parseInt(System.getenv().getOrDefault("CAPSTONE_MAX_SEQ_LEN", "10000"));

        System.out.println("=== CapstoneAI Sentence Embedding Server ===");
        System.out.printf("  host=%s  port=%d  ai_threads=%d  max_seq_len=%d%n",
                host, port, aiThreads, maxSeqLen);

        // ---- Tokenizer -----------------------------------------------------
        Tokenizer tokenizer = Tokenizer.fromResources("spm_multi.model");
        System.out.printf("[tok]   vocab_size=%d%n", tokenizer.getVocabSize());

        // ---- ONNX Model ----------------------------------------------------
        Network network = Network.fromResources("sentence_encoder_int8.onnx");
        System.out.println("[model] int8 ONNX loaded");

        // ---- Services ------------------------------------------------------
        EmbeddingService embedder = new ThreadPoolEmbeddingService(
                tokenizer, network, aiThreads, maxSeqLen);
        String solrUrl = System.getenv().getOrDefault("SOLR_URL", "http://localhost:8983/solr/chapters");
        VectorStore vectorStore = new SolrVectorStore(solrUrl);
        System.out.printf("[store] Solr: %s%n", solrUrl);
        NovelService novelService = new NovelService(embedder, vectorStore);

        // ---- HTTP Server ---------------------------------------------------
        HttpChannelInboundHandler handler = new HttpChannelInboundHandler(novelService);
        NettyHttp server = new NettyHttp(host, port, handler);

        // ---- Graceful shutdown --------------------------------------------
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n[main]  shutdown signal received");
            try {
                server.close();
                embedder.close();
                network.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
            System.out.println("[main]  goodbye");
        }, "shutdown-hook"));

        System.out.println("[main]  ready — press Ctrl+C to stop");
        server.awaitShutdown();
    }
}
