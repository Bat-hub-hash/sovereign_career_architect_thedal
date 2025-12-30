"""Configuration management for Sovereign Career Architect."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for GPT-4o")
    groq_api_key: Optional[str] = Field(None, description="Groq API key for Llama-3-70B")
    mem0_api_key: Optional[str] = Field(None, description="Mem0 API key")
    vapi_api_key: Optional[str] = Field(None, description="Vapi.ai API key")
    sarvam_api_key: Optional[str] = Field(None, description="Sarvam-1 API key")
    huggingface_api_key: Optional[str] = Field(None, description="Hugging Face API key for Sarvam-1")
    
    # Model Configuration
    reasoning_model: str = Field("llama-3.1-70b-versatile", description="High-reasoning model")
    vision_model: str = Field("gpt-4o", description="Vision-language model")
    voice_model: str = Field("gpt-4o-mini", description="Voice interaction model")
    
    # Memory Configuration
    mem0_api_key: Optional[str] = Field(None, description="Mem0 API key")
    mem0_host: str = Field("https://api.mem0.ai", description="Mem0 API host")
    vector_store_type: str = Field("qdrant", description="Vector store type (qdrant/chroma)")
    qdrant_url: Optional[str] = Field(None, description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(None, description="Qdrant API key")
    qdrant_collection_name: str = Field("sovereign_career_architect", description="Qdrant collection name")
    chroma_persist_directory: str = Field("./data/chroma", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field("sovereign_career_architect", description="ChromaDB collection name")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model for vector store")
    memory_search_limit: int = Field(10, description="Default memory search limit")
    
    # Browser Configuration
    browser_headless: bool = Field(True, description="Run browser in headless mode")
    browser_stealth: bool = Field(True, description="Enable stealth mode")
    browser_timeout: int = Field(30, description="Browser operation timeout in seconds")
    
    # Voice Configuration
    vapi_webhook_url: Optional[str] = Field(None, description="Vapi.ai webhook URL")
    voice_latency_target: int = Field(500, description="Target voice latency in ms")
    
    # Application Configuration
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    max_retries: int = Field(3, description="Maximum retry attempts")
    
    # Server Configuration
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, description="Server port")
    reload: bool = Field(False, description="Enable auto-reload")
    api_host: str = Field("0.0.0.0", description="API server host")
    api_port: int = Field(8000, description="API server port")
    allowed_origins: list[str] = Field(["*"], description="CORS allowed origins")
    allowed_hosts: Optional[list[str]] = Field(None, description="Trusted hosts")
    
    # Security
    cors_origins: list[str] = Field(["*"], description="CORS allowed origins")
    api_key_header: str = Field("X-API-Key", description="API key header name")
    jwt_secret_key: str = Field("your-secret-key-change-in-production", description="JWT secret key")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(24, description="JWT expiration hours")
    
    # Browser Configuration (additional)
    browser_stealth_mode: bool = Field(True, description="Enable browser stealth mode")
    browser_user_data_dir: Optional[str] = Field(None, description="Browser user data directory")
    
    # Voice Configuration (additional)
    vapi_webhook_secret: Optional[str] = Field(None, description="Vapi.ai webhook secret")


# Global settings instance
settings = Settings()