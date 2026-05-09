from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Med RAG Agent"
    environment: str = "development"
    database_url: str = "sqlite:///./data/app.db"
    vector_store_path: str = "./data/vectorstore"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
