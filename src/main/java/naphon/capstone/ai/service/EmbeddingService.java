package naphon.capstone.ai.service;

import ai.onnxruntime.OrtException;

import java.util.concurrent.CompletableFuture;

/**
 * Async wrapper around the ONNX sentence encoder model.
 *
 * Inference is CPU-bound and must never run on a Netty I/O thread.
 * This service owns a dedicated thread pool for AI work.
 */
public interface EmbeddingService extends AutoCloseable {

    /**
     * Encode a single text asynchronously, returning an L2-normalized embedding.
     */
    CompletableFuture<float[]> embed(String text);

    /**
     * Encode a batch of texts asynchronously.
     */
    CompletableFuture<float[][]> embedBatch(String... texts);

    /** Blocking encode — use only from worker threads. */
    float[] embedBlocking(String text) throws OrtException;

    /** Blocking batch encode — use only from worker threads. */
    float[][] embedBatchBlocking(String... texts) throws OrtException;
}
