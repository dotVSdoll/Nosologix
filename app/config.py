from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Med RAG Agent"
    environment: str = "development"
    database_url: str = "sqlite:///./data/app.db"
    vector_store_provider: str = "memory"
    vector_store_path: str = "./data/vectorstore"
    vector_store_collection: str = "med_rag_chunks"
    embedding_provider: str = "hash"
    embedding_model: str = "hash-local"
    embedding_dimension: int = 128
    embedding_device: str = "cpu"
    embedding_use_fp16: bool = False
    llm_provider: str = "template"
    llm_model: str = "template-local"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key: str = ""
    llm_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
