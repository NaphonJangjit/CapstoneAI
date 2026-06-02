package naphon.capstone.ai.service;

import naphon.capstone.ai.Network;
import naphon.capstone.ai.Tokenizer;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.*;

/**
 * Thread-pool backed embedding service. Keeps ONNX inference off
 * the Netty event loop by running all AI work on a dedicated executor.
 */
public class ThreadPoolEmbeddingService implements EmbeddingService {

    private final Tokenizer tokenizer;
    private final Network network;
    private final ExecutorService executor;
    private final int maxSeqLen;

    public ThreadPoolEmbeddingService(Tokenizer tokenizer,
                                      Network network,
                                      int numThreads,
                                      int maxSeqLen) {
        this.tokenizer = tokenizer;
        this.network = network;
        this.maxSeqLen = maxSeqLen;
        this.executor = Executors.newFixedThreadPool(numThreads, r -> {
            Thread t = new Thread(r, "ai-worker");
            t.setDaemon(true);
            return t;
        });
    }

    @Override
    public CompletableFuture<float[]> embed(String text) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return embedBlocking(text);
            } catch (Exception e) {
                throw new CompletionException(e);
            }
        }, executor);
    }

    @Override
    public float[] embedBlocking(String text) {
        int[] ids = tokenizer.tokenizeAsId(text);
        if (ids.length > maxSeqLen) {
            ids = Arrays.copyOf(ids, maxSeqLen);
        }
        long[] longIds = new long[ids.length];
        for (int i = 0; i < ids.length; i++) longIds[i] = ids[i];

        try {
            return network.embedSingle(longIds);
        } catch (Exception e) {
            throw new RuntimeException("Embedding failed", e);
        }
    }

    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    @Override
    public void close() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
