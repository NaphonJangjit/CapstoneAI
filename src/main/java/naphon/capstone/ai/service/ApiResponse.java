package naphon.capstone.ai.service;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import java.util.List;

/**
 * Standard JSON response envelope for the REST API.
 *
 * Success: {"status":"ok", "data":{...}}
 * Error:   {"status":"error", "error":"message"}
 */
public final class ApiResponse {

    private static final Gson GSON = new Gson();

    public String status;
    public Object data;
    public String error;

    private ApiResponse(String status, Object data, String error) {
        this.status = status;
        this.data = data;
        this.error = error;
    }

    public static ApiResponse ok(Object data) {
        return new ApiResponse("ok", data, null);
    }

    public static ApiResponse error(String message) {
        return new ApiResponse("error", null, message);
    }

    public String toJson() {
        return GSON.toJson(this);
    }

    /** Lightweight search result entry for JSON serialization. */
    public static class SearchHit {
        @SerializedName("novel_id")   public long novelId;
        @SerializedName("chapter")     public long chapter;
        @SerializedName("novel_name") public String novelName;
        @SerializedName("chapter_name") public String chapterName;
        public List<String> tags;
        public double score;

        public SearchHit(VectorStore.ScoredEntry se) {
            var e = se.entry();
            this.novelId = e.novelId();
            this.chapter = e.chapter();
            this.novelName = e.novelName();
            this.chapterName = e.chapterName();
            this.tags = e.tags();
            this.score = se.score();
        }
    }
}
