# CapstoneAI

CapstoneAI is a high-performance Java-based AI server designed for novel embedding and semantic search. It provides a REST API for indexing novel chapters and performing vector-based searches using Apache Solr as a dense-vector store.

## Project Overview

- **Purpose:** Provide semantic search capabilities for novels by embedding chapters into high-dimensional vectors and performing KNN search.
- **Architecture:**
    - **Netty HTTP Layer:** Asynchronous request handling using `io.netty`.
    - **AI Worker Pool:** CPU-bound worker pool for ONNX model inference (sentence embeddings).
    - **Vector Store:** Integration with Apache Solr for KNN (K-Nearest Neighbors) dense-vector search.
    - **Service Layer:** `NovelService` orchestrates embeddings and storage logic.

## Core Technologies

- **Java 26:** Utilizes modern Java features (Eclipse Temurin 26).
- **Maven:** Project and dependency management.
- **ONNX Runtime:** Efficient model inference for sentence embeddings.
- **Netty:** High-performance asynchronous HTTP server.
- **SentencePiece:** Tokenization for AI models.
- **Apache Solr (SolrJ):** Dense-vector storage and similarity search.
- **GSON:** JSON serialization and deserialization.

## Building and Running

### Build Commands
```bash
mvn clean package
```

### Running the Application
**Local Execution:**
```bash
java -cp "target/CapstoneAI-1.0-SNAPSHOT.jar;target/dependency/*" naphon.capstone.ai.Main
```
*(Note: Use `:` instead of `;` as the path separator on Linux/macOS)*

**Docker Execution:**
```bash
docker build -t capstone-ai .
docker run -p 9922:9922 capstone-ai
```

### Testing
- No automated tests found in `src/test`.
- Manual verification can be done using the API endpoints listed below.

## Configuration (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `CAPSTONE_HOST` | Bind address for the HTTP server | `0.0.0.0` |
| `CAPSTONE_PORT` | Port for the HTTP server | `9922` |
| `CAPSTONE_AI_THREADS` | Number of threads for AI inference | `Runtime.availableProcessors()` |
| `CAPSTONE_MAX_SEQ_LEN` | Maximum sequence length for embeddings | `10000` |
| `SOLR_URL` | URL for the Solr chapters collection | `http://localhost:8983/solr/chapters` |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chapters` | Index a novel chapter (requires JSON body) |
| `GET` | `/api/v1/search/novel/{id}` | Search for novels similar to the specified novel |
| `GET` | `/api/v1/search/chapter/{nid}/{ch}` | Search for chapters similar to a specific chapter |
| `GET` | `/api/v1/search/keyword?keyword=...&k=10` | Search for chapters by free-text keyword |
| `DELETE` | `/api/v1/novels/{id}` | Delete all indexed chapters for a novel |
| `DELETE` | `/api/v1/chapters/{nid}/{ch}` | Delete a specific chapter |
| `GET` | `/api/v1/health` | Health check and current index size |

## Development Conventions

- **Asynchronous Execution:** All methods involving AI model inference or network I/O must be non-blocking. Use `CompletableFuture` for async operations.
- **Netty Threads:** Never perform blocking operations (e.g., synchronous I/O, heavy computation) on Netty's event loop threads. Dispatch AI tasks to the `EmbeddingService` thread pool.
- **Error Handling:** Use `ApiResponse` for consistent JSON error structures in the API.
- **Solr Integration:** Solr is expected to have a collection named `chapters` with a dense vector field configured.
