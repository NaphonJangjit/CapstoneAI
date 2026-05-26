package naphon.capstone.ai.object;

public record Chapter(long novelId, long chapter, String content, float[] vec, String[] tags, String genre) {
}
