package naphon.capstone.ai.object;

import java.util.List;

public record NovelSaveRequest(long novelId, long chapter, String novelName, String chapterName, String novelContent, List<String> tags) {
}
