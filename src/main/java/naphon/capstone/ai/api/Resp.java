package naphon.capstone.ai.api;

import io.netty.handler.codec.http.HttpResponseStatus;

public class Resp {

    private String content;
    private HttpResponseStatus status;

    public Resp(String content, HttpResponseStatus status){
        this.content = content;
        this.status = status;
    }

    public String getContent() {
        return content;
    }

    public HttpResponseStatus getStatus() {
        return status;
    }
}
