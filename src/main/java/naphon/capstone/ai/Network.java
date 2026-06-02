package naphon.capstone.ai;

import ai.onnxruntime.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

public class Network implements AutoCloseable {

    private static final int WORKERS;

    static {
        String workersStr = System.getenv().getOrDefault("CAPSTONE_AI_WORKER", "4");
        int workers;
        try {
            workers = Integer.parseInt(workersStr);
        } catch (NumberFormatException ex) {
            workers = 4;
        }
        WORKERS = workers;
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    public Network(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);
        opts.setInterOpNumThreads(WORKERS);
        opts.setIntraOpNumThreads(2);
        opts.addCPU(true);
        session = env.createSession(modelPath, opts);
    }

    /**
     * Load the model from the classpath (JAR resources).
     */
    public static Network fromResources(String resourcePath) throws OrtException, IOException {
        // ONNX Runtime can't read directly from classpath resources, so copy to a temp file.
        Path tempFile = Files.createTempFile("sentence_encoder_", ".onnx");
        tempFile.toFile().deleteOnExit();
        try (InputStream in = Network.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
            Files.copy(in, tempFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        return new Network(tempFile.toAbsolutePath().toString());
    }

    /**
     * Encode a batch of token sequences into L2-normalized embeddings.
     *
     * @param ids     flat array of all token IDs concatenated
     * @param offsets cumulative token counts per sequence (length = batch size)
     * @return float[][] of shape [batch_size][dim]
     */
    public float[][] embed(long[] ids, long[] offsets) throws OrtException {
        int batchSize = offsets.length;

        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{ids.length});
             OnnxTensor offsetsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(offsets), new long[]{batchSize})) {

            Map<String, OnnxTensor> inputs = Map.of("ids", idsTensor, "offsets", offsetsTensor);

            OrtSession.Result result = session.run(inputs, Collections.singleton("embedding"));

            OnnxTensor outTensor = (OnnxTensor) result.get("embedding")
                    .orElseThrow(() -> new OrtException("Missing 'embedding' output"));

            FloatBuffer buffer = outTensor.getFloatBuffer();
            int dim = (int) outTensor.getInfo().getShape()[1];

            float[][] embeddings = new float[batchSize][dim];
            for (int b = 0; b < batchSize; b++) {
                float[] row = new float[dim];
                buffer.position(b * dim);
                buffer.get(row, 0, dim);
                embeddings[b] = row;
            }
            outTensor.close();
            return embeddings;
        }
    }

    /**
     * Encode a single text (convenience wrapper — caller must tokenize first).
     */
    public float[] embedSingle(long[] ids) throws OrtException {
        return embed(ids, new long[]{ids.length})[0];
    }

    @Override
    public void close() throws OrtException {
        session.close();
        env.close();
    }
}
