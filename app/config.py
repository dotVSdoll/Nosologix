from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Med RAG Agent"
    environment: str = "development"
    database_url: str = "sqlite:///./data/app.db"
    vector_store_path: str = "./data/vectorstore"
    embedding_provider: str = "hash"
    embedding_model: str = "hash-local"
    embedding_dimension: int = 128
    embedding_device: str = "cpu"
    embedding_use_fp16: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
