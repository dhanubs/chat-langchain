from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # Application settings
    environment: str = "development"
    mongodb_uri: str = "mongodb://localhost:27017"
    llm_provider: str = "azure"
    llm_model: str = "gpt-35-turbo-16k"

    # Azure OpenAI settings
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-15-preview"
    
    # Vanilla OpenAI settings
    openai_api_key: str | None = None
    
    # Azure Search settings
    azure_search_service_endpoint: str | None = None
    azure_search_service_name: str | None = None
    azure_search_key: str | None = None
    azure_search_admin_key: str | None = None
    azure_search_index_name: str = "documents"
    
    # Other provider settings
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    fireworks_api_key: str | None = None
    groq_api_key: str | None = None
    cohere_api_key: str | None = None
    
    class Config:
        env_file = ".env",
        case_sensitive = False,
        extra = "allow"

settings = Settings() 