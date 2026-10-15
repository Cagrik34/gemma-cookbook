from huggingface_hub import hf_hub_download

def download_embedding_model() -> None:
    print("Downloading EmbeddingGemma2 LiteRT model...")
    EMBEDDING_REPO_ID = "litert-community/embeddinggemma-2-740m-litert-lm"
    EMBEDDING_FILENAME = "embeddinggemma-2-740m.litertlm"
    EMBEDDING_LOCAL_DIR = "models/embeddinggemma2"

    model_path = hf_hub_download(
        repo_id=EMBEDDING_REPO_ID,
        filename=EMBEDDING_FILENAME,
        local_dir=EMBEDDING_LOCAL_DIR,
    )

    print(f"EmbeddingGemma2 model saved at: {model_path}")

def download_gemma_litert_model() -> None:
    print("Downloading Gemma LiteRT model...")
    GEMMA_REPO_ID = "litert-community/gemma-4-E2B-it-litert-lm"
    GEMMA_FILENAME = "gemma-4-E2B-it.litertlm"
    GEMMA_LOCAL_DIR = "models/gemma4"

    model_path = hf_hub_download(
        repo_id=GEMMA_REPO_ID,
        filename=GEMMA_FILENAME,
        local_dir=GEMMA_LOCAL_DIR,
    )
    print(f"Gemma LiteRT model saved at: {model_path}")

def main() -> None:
    download_embedding_model()
    download_gemma_litert_model()

if __name__ == "__main__":
    main()