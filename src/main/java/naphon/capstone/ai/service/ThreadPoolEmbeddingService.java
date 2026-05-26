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
    public CompletableFuture<float[][]> embedBatch(String... texts) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return embedBatchBlocking(texts);
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

    @Override
    public float[][] embedBatchBlocking(String... texts) {
        // Concatenate all token sequences
        int totalTokens = 0;
        int[][] allIds = new int[texts.length][];
        for (int i = 0; i < texts.length; i++) {
            int[] ids = tokenizer.tokenizeAsId(texts[i]);
            if (ids.length > maxSeqLen) ids = Arrays.copyOf(ids, maxSeqLen);
            allIds[i] = ids;
            totalTokens += ids.length;
        }

        long[] flatIds = new long[totalTokens];
        long[] offsets = new long[texts.length];
        int pos = 0;
        for (int i = 0; i < texts.length; i++) {
            for (int id : allIds[i]) flatIds[pos++] = id;
            offsets[i] = pos;
        }

        try {
            return network.embed(flatIds, offsets);
        } catch (Exception e) {
            throw new RuntimeException("Batch embedding failed", e);
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
