package naphon.capstone.ai;

import ai.djl.sentencepiece.SpTokenizer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class Tokenizer {

    private final byte[] modelBytes;
    private final SpTokenizer tokenizer;

    public Tokenizer(Path modelPath) throws IOException {
        this(Files.readAllBytes(modelPath));
    }

    public Tokenizer(byte[] modelBytes) {
        this.modelBytes = modelBytes;
        this.tokenizer = new SpTokenizer(modelBytes);
    }

    public static Tokenizer fromResources(String resourcePath) throws IOException {
        try (InputStream in = Tokenizer.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
            return new Tokenizer(in.readAllBytes());
        }
    }

    public List<String> tokenize(String sentence) {
        return tokenizer.tokenize(sentence);
    }


    public int[] tokenizeAsId(String sentence) {
        return tokenizer.getProcessor().encode(sentence);
    }

    public int getVocabSize() {
        // SentencePiece default. The native vocab_size is not exposed
        // by DJL's SpProcessor; keep in sync with the tokenizer model.
        return 100_000;
    }

}
