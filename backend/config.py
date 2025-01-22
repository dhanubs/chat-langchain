from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Azure OpenAI settings
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    
    # Vanilla OpenAI settings
    openai_api_key: str
    
    # Azure Search settings
    azure_search_service_endpoint: str
    azure_search_admin_key: str
    azure_search_index_name: str = "documents"
    
    # Other provider settings
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    fireworks_api_key: str | None = None
    groq_api_key: str | None = None
    cohere_api_key: str | None = None
    
    class Config:
        env_file = ".env"

settings = Settings() 