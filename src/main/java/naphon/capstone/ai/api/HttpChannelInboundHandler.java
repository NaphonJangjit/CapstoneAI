package naphon.capstone.ai.api;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler.Sharable;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.*;
import naphon.capstone.ai.object.NovelSaveRequest;
import naphon.capstone.ai.service.ApiResponse;
import naphon.capstone.ai.service.NovelService;
import naphon.capstone.ai.service.VectorStore;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;

/**
 * Production HTTP API handler.
 *
 * NEVER blocks the Netty I/O thread. All AI inference is dispatched to
 * the {@link NovelService} which runs on its own thread pool. Responses
 * are written back on the event loop via {@code ctx.executor()}.
 *
 * <h3>Endpoints</h3>
 * <pre>
 *   POST   /api/v1/chapters              index a chapter
 *   GET    /api/v1/search/novel/{id}     search by novel (avg embedding)
 *   GET    /api/v1/search/chapter/{nid}/{ch}  search by specific chapter
 *   DELETE /api/v1/novels/{id}           delete all chapters of a novel
 *   DELETE /api/v1/chapters/{nid}/{ch}   delete one chapter
 *   GET    /api/v1/health                health + index size
 * </pre>
 */
@Sharable
public class HttpChannelInboundHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Gson GSON = new Gson();
    private static final String API_PREFIX = "/api/v1/";

    private final NovelService novelService;

    public HttpChannelInboundHandler(NovelService novelService) {
        this.novelService = novelService;
    }

    // =========================================================================
    // Netty callback
    // =========================================================================

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
        Executor eventLoop = ctx.executor();

        try {
            String path = cleanPath(request.uri());
            HttpMethod method = request.method();

            // -- POST /api/v1/chapters ------------------------------------------
            if (method == HttpMethod.POST && path.equals(API_PREFIX + "chapters")) {
                handleIndexChapter(ctx, request, eventLoop);
                return;
            }

            // -- GET /api/v1/search/novel/{id} -----------------------------------
            if (method == HttpMethod.GET && path.startsWith(API_PREFIX + "search/novel/")) {
                handleSearchByNovel(ctx, request, path, eventLoop);
                return;
            }

            // -- GET /api/v1/search/chapter/{nid}/{ch} ---------------------------
            if (method == HttpMethod.GET && path.startsWith(API_PREFIX + "search/chapter/")) {
                handleSearchByChapter(ctx, request, path, eventLoop);
                return;
            }

            // -- GET /api/v1/search/keyword?keyword=...&k=10 -----------------------
            if (method == HttpMethod.GET && path.equals(API_PREFIX + "search/keyword")) {
                handleSearchByKeyword(ctx, request, eventLoop);
                return;
            }

            // -- DELETE /api/v1/novels/{id} --------------------------------------
            if (method == HttpMethod.DELETE && path.startsWith(API_PREFIX + "novels/")) {
                handleDeleteNovel(ctx, path, eventLoop);
                return;
            }

            // -- DELETE /api/v1/chapters/{nid}/{ch} ------------------------------
            if (method == HttpMethod.DELETE && path.startsWith(API_PREFIX + "chapters/")) {
                handleDeleteChapter(ctx, path, eventLoop);
                return;
            }

            // -- GET /api/v1/health ----------------------------------------------
            if (method == HttpMethod.GET && path.equals(API_PREFIX + "health")) {
                handleHealth(ctx, eventLoop);
                return;
            }

            // -- 404 ------------------------------------------------------------
            write(ctx, HttpResponseStatus.NOT_FOUND,
                    ApiResponse.error("Not found: " + method + " " + path));

        } catch (Exception e) {
            write(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR,
                    ApiResponse.error("Internal error: " + e.getMessage()));
        }
    }

    // =========================================================================
    // Handlers
    // =========================================================================

    /** POST /api/v1/chapters — index a chapter. */
    private void handleIndexChapter(ChannelHandlerContext ctx, FullHttpRequest request,
                                     Executor eventLoop) {
        String body = request.content().toString(StandardCharsets.UTF_8);
        NovelSaveRequest req;
        try {
            req = GSON.fromJson(body, NovelSaveRequest.class);
        } catch (JsonSyntaxException e) {
            write(ctx, HttpResponseStatus.BAD_REQUEST,
                    ApiResponse.error("Invalid JSON: " + e.getMessage()));
            return;
        }
        if (req == null || req.novelContent() == null || req.novelContent().isBlank()) {
            write(ctx, HttpResponseStatus.BAD_REQUEST,
                    ApiResponse.error("novelContent is required"));
            return;
        }

        List<String> tags = req.tags() != null ? req.tags() : List.of();

        final String chapterName = req.chapterName() != null ? req.chapterName()
                : req.chapter() + "";
        novelService.indexChapter(
                req.novelId(), req.chapter(),
                req.novelName(), chapterName,
                req.novelContent(), tags
        ).thenAcceptAsync(chapter -> {
            Map<String, Object> data = Map.of(
                    "novel_id", chapter.novelId(),
                    "chapter", chapter.chapter(),
                    "chapter_name", chapterName,
                    "tags", tags,
                    "dim", chapter.vec().length
            );
            write(ctx, HttpResponseStatus.OK, ApiResponse.ok(data));
        }, eventLoop).exceptionally(ex -> {
            write(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR,
                    ApiResponse.error(causeMessage(ex)));
            return null;
        });
    }

    /** GET /api/v1/search/novel/{id}?k=10 */
    private void handleSearchByNovel(ChannelHandlerContext ctx, FullHttpRequest request,
                                      String path, Executor eventLoop) {
        String remainder = path.substring((API_PREFIX + "search/novel/").length());
        if (remainder.isEmpty()) {
            write(ctx, HttpResponseStatus.BAD_REQUEST, ApiResponse.error("Missing novel ID"));
            return;
        }
        long novelId;
        try {
            novelId = Long.parseLong(remainder);
        } catch (NumberFormatException e) {
            write(ctx, HttpResponseStatus.BAD_REQUEST, ApiResponse.error("Novel ID must be a number"));
            return;
        }
        int topK = parseIntParam(parseQueryParams(request.uri()), "k", 10);

        novelService.searchByNovel(novelId, topK)
                .thenAcceptAsync(results -> {
                    List<ApiResponse.SearchHit> hits = results.stream()
                            .map(ApiResponse.SearchHit::new)
                            .toList();
                    write(ctx, HttpResponseStatus.OK,
                            ApiResponse.ok(Map.of("results", hits, "count", hits.size())));
                }, eventLoop)
                .exceptionally(ex -> {
                    write(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR,
                            ApiResponse.error(causeMessage(ex)));
                    return null;
                });
    }

    /** GET /api/v1/search/chapter/{nid}/{ch}?k=10 */
    private void handleSearchByChapter(ChannelHandlerContext ctx, FullHttpRequest request,
                                        String path, Executor eventLoop) {
        String remainder = path.substring((API_PREFIX + "search/chapter/").length());
        String[] parts = remainder.split("/");
        if (parts.length < 2) {
            write(ctx, HttpResponseStatus.BAD_REQUEST,
                    ApiResponse.error("Expected /search/chapter/{novelId}/{chapter}"));
            return;
        }
        long novelId, chapter;
        try {
            novelId = Long.parseLong(parts[0]);
            chapter = Long.parseLong(parts[1]);
        } catch (NumberFormatException e) {
            write(ctx, HttpResponseStatus.BAD_REQUEST, ApiResponse.error("IDs must be numbers"));
            return;
        }
        int topK = parseIntParam(parseQueryParams(request.uri()), "k", 10);

        novelService.searchByChapter(novelId, chapter, topK)
                .thenAcceptAsync(results -> {
                    List<ApiResponse.SearchHit> hits = results.stream()
                            .map(ApiResponse.SearchHit::new)
                            .toList();
                    write(ctx, HttpResponseStatus.OK,
                            ApiResponse.ok(Map.of("results", hits, "count", hits.size())));
                }, eventLoop)
                .exceptionally(ex -> {
                    write(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR,
                            ApiResponse.error(causeMessage(ex)));
                    return null;
                });
    }

    /** GET /api/v1/search/keyword?keyword=...&k=10 */
    private void handleSearchByKeyword(ChannelHandlerContext ctx, FullHttpRequest request,
                                       Executor eventLoop) {
        Map<String, String> params = parseQueryParams(request.uri());
        String keyword = params.get("keyword");
        if (keyword == null || keyword.isBlank()) {
            write(ctx, HttpResponseStatus.BAD_REQUEST, ApiResponse.error("keyword is required"));
            return;
        }
        int topK = parseIntParam(params, "k", 10);

        novelService.searchByKeyword(keyword, topK)
                .thenAcceptAsync(results -> {
                    List<ApiResponse.SearchHit> hits = results.stream()
                            .map(ApiResponse.SearchHit::new)
                            .toList();
                    write(ctx, HttpResponseStatus.OK,
                            ApiResponse.ok(Map.of("results", hits, "count", hits.size())));
                }, eventLoop)
                .exceptionally(ex -> {
                    write(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR,
                            ApiResponse.error(causeMessage(ex)));
                    return null;
                });
    }

    /** DELETE /api/v1/novels/{id} */
    private void handleDeleteNovel(ChannelHandlerContext ctx, String path,
                                    Executor eventLoop) {
        String remainder = path.substring((API_PREFIX + "novels/").length());
        long novelId;
        try {
            novelId = Long.parseLong(remainder);
        } catch (NumberFormatException e) {
            write(ctx, HttpResponseStatus.BAD_REQUEST, ApiResponse.error("Novel ID must be a number"));
            return;
        }

        CompletableFuture.supplyAsync(() -> novelService.deleteNovel(novelId))
                .thenAcceptAsync(removed -> {
                    write(ctx, HttpResponseStatus.OK,
                            ApiResponse.ok(Map.of("deleted", removed)));
                }, eventLoop);
    }

    /** DELETE /api/v1/chapters/{nid}/{ch} */
    private void handleDeleteChapter(ChannelHandlerContext ctx, String path,
                                      Executor eventLoop) {
        String remainder = path.substring((API_PREFIX + "chapters/").length());
        String[] parts = remainder.split("/");
        if (parts.length < 2) {
            write(ctx, HttpResponseStatus.BAD_REQUEST,
                    ApiResponse.error("Expected /chapters/{novelId}/{chapter}"));
            return;
        }
        long novelId, chapter;
        try {
            novelId = Long.parseLong(parts[0]);
            chapter = Long.parseLong(parts[1]);
        } catch (NumberFormatException e) {
            write(ctx, HttpResponseStatus.BAD_REQUEST, ApiResponse.error("IDs must be numbers"));
            return;
        }

        boolean found = novelService.deleteChapter(novelId, chapter);
        write(ctx, found ? HttpResponseStatus.OK : HttpResponseStatus.NOT_FOUND,
                ApiResponse.ok(Map.of("deleted", found)));
    }

    /** GET /api/v1/health */
    private void handleHealth(ChannelHandlerContext ctx, Executor eventLoop) {
        write(ctx, HttpResponseStatus.OK,
                ApiResponse.ok(Map.of(
                        "status", "healthy",
                        "indexed_chapters", novelService.getIndexedCount()
                )));
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /** Write a JSON response and close the connection (or keep-alive). */
    private void write(ChannelHandlerContext ctx, HttpResponseStatus status, ApiResponse body) {
        byte[] bytes = body.toJson().getBytes(StandardCharsets.UTF_8);
        FullHttpResponse response = new DefaultFullHttpResponse(
                HttpVersion.HTTP_1_1,
                status,
                Unpooled.wrappedBuffer(bytes)
        );
        response.headers()
                .set(HttpHeaderNames.CONTENT_TYPE, "application/json; charset=UTF-8")
                .setInt(HttpHeaderNames.CONTENT_LENGTH, bytes.length);

        // Keep-Alive by default unless the request asked to close
        ctx.writeAndFlush(response).addListener(ChannelFutureListener.CLOSE_ON_FAILURE);
    }

    /** Clean URI path — strip query/fragment, normalize. */
    private static String cleanPath(String uri) {
        if (uri == null || uri.isBlank()) return "";
        String path = uri.trim();

        int qi = path.indexOf('?');
        if (qi != -1) path = path.substring(0, qi);
        int fi = path.indexOf('#');
        if (fi != -1) path = path.substring(0, fi);

        try {
            java.net.URI parsed = new java.net.URI(path);
            if (parsed.getPath() != null && !parsed.getPath().isEmpty()) {
                path = parsed.getPath();
            }
        } catch (Exception ignored) {}

        while (path.length() > 1 && path.endsWith("/")) {
            path = path.substring(0, path.length() - 1);
        }
        return path;
    }

    /** Parse query string into a map. */
    private static Map<String, String> parseQueryParams(String uri) {
        Map<String, String> params = new LinkedHashMap<>();
        int qi = uri.indexOf('?');
        if (qi == -1 || qi == uri.length() - 1) return params;

        String qs = uri.substring(qi + 1);
        for (String pair : qs.split("&")) {
            int eq = pair.indexOf('=');
            if (eq == -1) {
                params.put(decode(pair), "");
            } else {
                params.put(decode(pair.substring(0, eq)), decode(pair.substring(eq + 1)));
            }
        }
        return params;
    }

    private static String decode(String s) {
        try {
            return URLDecoder.decode(s, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return s;
        }
    }

    private static int parseIntParam(Map<String, String> params, String key, int defaultVal) {
        String val = params.get(key);
        if (val == null) return defaultVal;
        try {
            return Math.max(1, Math.min(Integer.parseInt(val), 100));
        } catch (NumberFormatException e) {
            return defaultVal;
        }
    }

    private static String causeMessage(Throwable ex) {
        Throwable cause = ex;
        while (cause instanceof CompletionException && cause.getCause() != null) {
            cause = cause.getCause();
        }
        return cause.getMessage() != null ? cause.getMessage() : cause.getClass().getSimpleName();
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        System.err.println("[handler] " + cause.getMessage());
        ctx.close();
    }
}
