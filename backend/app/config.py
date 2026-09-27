"""
Application configuration, loaded from environment variables (see
.env.example at the project root). Never hard-code secrets here.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/nepali_news"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    ml_models_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml", "models"
    )
    min_text_length: int = 3
    max_text_length: int = 20000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
