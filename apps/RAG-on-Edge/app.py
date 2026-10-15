import os
import litert_lm
from litert_lm import Backend, EmbeddingEngine
from qdrant_edge import EdgeShard, Query, QueryRequest
from quiet_native import silence_stderr
os.environ["HF_HUB_OFFLINE"] = "1"
silence_stderr()

EMBED_MODEL_NAME = "models/embeddinggemma2/embeddinggemma-2-740m.litertlm"
SHARD_PATH = "db_gemma"
GEMMA4_LITELLM = "models/gemma4/gemma-4-E2B-it.litertlm"
CACHE_DIR = ".cache"

class GemmaInference:
    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        shard_path: str = SHARD_PATH,
        gemma_model_path: str = GEMMA4_LITELLM,
    ) -> None:

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.model = EmbeddingEngine(
            embed_model_name, backend=Backend.GPU(), cache_dir=CACHE_DIR
        )

        self.shard = EdgeShard.load(shard_path)

        self.gemma_model_path = gemma_model_path
        self.engine = litert_lm.Engine(
            self.gemma_model_path,
            backend=litert_lm.Backend.GPU(),
        )

    def close(self) -> None:
        self.engine.close()
        self.model.close()

    def retrieve_context(self, user_query: str, limit: int = 2) -> str:
        query_vector = self.model.compute_embedding(user_query).embedding
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)

    def _system_messages(self) -> list[litert_lm.Message]:
        return [
            litert_lm.Message.system(
                "You are a Gemma 4 technical report assistant. "
                "Answer the user question using only the provided context. "
                "If the answer is not in the context, say: I don't know"
                "Do not guess or add extra explanation."
                "Answer in 1-2 short sentences. Do not elaborate unless told by user"
            )
        ]

    def answer_stream(self, user_query: str, limit: int = 3, context: str | None = None):
        if context is None:
            context = self.retrieve_context(user_query, limit=limit)
        with self.engine.create_conversation(
            messages=self._system_messages(),
            sampler_config=litert_lm.SamplerConfig(top_k=1),
        ) as conversation:
            for chunk in conversation.send_message_async(
                f"Context:\n{context}\n\nQuestion:\n{user_query}"
            ):
                for item in chunk.get("content", []):
                    if item.get("type") == "text" and item.get("text"):
                        yield item["text"]

    def answer(self, user_query: str, limit: int = 3) -> str:
        return "".join(self.answer_stream(user_query, limit=limit))

QUERIES = [
    "What is the parameter range of models in the Gemma 4 suite?",
    "What architecture is Gemma 4 based on?",
    "What context length does Gemma 4 support?",
    "How does Gemma 4 perform on benchmarks compared to prior Gemma models?"
]

def main() -> None:
    inference = GemmaInference()

    for i, user_query in enumerate(QUERIES, start=1):
        print(f"\n[{i}] {user_query}")
        context = inference.retrieve_context(user_query)
        answer = "".join(inference.answer_stream(user_query, context=context))
        print(answer)

    inference.close()

if __name__ == "__main__":
    main()