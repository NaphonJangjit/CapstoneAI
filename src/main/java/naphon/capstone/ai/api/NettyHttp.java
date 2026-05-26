package naphon.capstone.ai.api;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.*;
import io.netty.channel.nio.NioIoHandler;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.http.*;

/**
 * Netty HTTP server configured for non-blocking I/O.
 *
 * The event loop threads never touch AI inference — the
 * {@link HttpChannelInboundHandler} dispatches all CPU work to the
 * {@code EmbeddingService} thread pool and writes responses back
 * on the event loop via {@code ctx.executor()}.
 */
public class NettyHttp implements AutoCloseable {

    private final EventLoopGroup bossGroup;
    private final EventLoopGroup workerGroup;
    private final Channel channel;
    private final String host;
    private final int port;

    public NettyHttp(String host, int port, HttpChannelInboundHandler handler) throws InterruptedException {
        this.host = host;
        this.port = port;

        bossGroup = new MultiThreadIoEventLoopGroup(1, NioIoHandler.newFactory());
        workerGroup = new MultiThreadIoEventLoopGroup(NioIoHandler.newFactory());

        ServerBootstrap bootstrap = new ServerBootstrap();
        bootstrap.group(bossGroup, workerGroup)
                .channel(NioServerSocketChannel.class)
                .childHandler(new ChannelInitializer<SocketChannel>() {
                    @Override
                    protected void initChannel(SocketChannel ch) {
                        ChannelPipeline p = ch.pipeline();
                        p.addLast(new HttpServerCodec());
                        p.addLast(new HttpObjectAggregator(65536));
                        p.addLast(handler);
                    }
                });

        channel = bootstrap.bind(this.host, this.port).sync().channel();
        System.out.printf("[http]  listening on %s:%d%n", this.host, this.port);
    }

    /** Block until the server channel is closed. */
    public void awaitShutdown() throws InterruptedException {
        channel.closeFuture().sync();
    }

    @Override
    public void close() throws InterruptedException {
        System.out.println("[http]  shutting down...");
        channel.close().sync();
        bossGroup.shutdownGracefully().sync();
        workerGroup.shutdownGracefully().sync();
        System.out.println("[http]  stopped");
    }
}
