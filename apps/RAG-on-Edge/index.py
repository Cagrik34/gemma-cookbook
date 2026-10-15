from pathlib import Path

import edgeparse
from qdrant_edge import (
    Distance,EdgeConfig,EdgeShard,EdgeVectorParams,
    Point,Query,QueryRequest,UpdateOperation,
)
from litert_lm import Backend, EmbeddingEngine

from quiet_native import silence_stderr
silence_stderr()

MODEL_DIM = 768  
EMBED_MODEL_PATH = "models/embeddinggemma2/embeddinggemma-2-740m.litertlm"
DOCUMENT_PATH = "../sample-data/gemma4_technical_report.pdf"
SHARD_PATH = "db_gemma"
CACHE_DIR = ".cache"

class DocumentIngestion:
    def __init__(
        self,
        document_path: str,
        model_path: str,
        chunk_size: int = 1024,
        overlap: int = 0,
    ) -> None:
        self.document_path = Path(document_path)
        self.shard_path = Path(SHARD_PATH)
        self.chunk_size = chunk_size
        self.overlap = overlap

        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        self.engine = EmbeddingEngine(
            model_path, backend=Backend.GPU(), cache_dir=CACHE_DIR
        )

        self.shard = self._load_or_create_shard()

    def close(self) -> None:
        self.engine.close()

    def _load_or_create_shard(self) -> EdgeShard:
        config = EdgeConfig(
            vectors=EdgeVectorParams(size=MODEL_DIM, distance=Distance.Cosine)
        )

        self.shard_path.mkdir(parents=True, exist_ok=True)

        if any(self.shard_path.iterdir()):
            print(f"Already exists {self.shard_path}")
            return EdgeShard.load(str(self.shard_path))
        return EdgeShard.create(str(self.shard_path), config)

    def chunk_text(self, text: str) -> list[str]:
        return [
            chunk
            for start in range(0, len(text), self.chunk_size - self.overlap)
            if (chunk := text[start : start + self.chunk_size].strip())
        ]

    def index(self) -> int:
        markdown = edgeparse.convert(str(self.document_path), format="markdown") # load the data
        chunks = self.chunk_text(markdown)

        responses = self.engine.compute_embedding_batch(chunks)
        embeddings = [response.embedding for response in responses]

        self.shard.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        index,
                        embeddings[index - 1],
                        {
                            "text": chunk,
                            "source": self.document_path.name,
                            "chunk_index": index - 1,
                        },
                    )
                    for index, chunk in enumerate(chunks, start=1)
                ]
            )
        )
        self.shard.optimize()
        return len(chunks)

    def inference(self, user_query: str, limit: int = 2) -> str:
        query_vector = self.engine.compute_embedding(user_query).embedding
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit, with_payload=True,
            )
        )
        context = ""
        for point in results:
            context += point.payload['text']

        return context

def main() -> None:
    ingestion = DocumentIngestion(
        document_path=DOCUMENT_PATH, model_path=EMBED_MODEL_PATH
    )
    indexed_count = ingestion.index()
    print(f"Indexed {indexed_count} chunks")

    user_query = "What is the parameter range of models in the Gemma 4 suite?"
    context = ingestion.inference(user_query)
    print(context)

    ingestion.close()

if __name__ == "__main__":
    main()