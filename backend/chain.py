from typing import Optional, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import ConfigurableField
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_fireworks import ChatFireworks
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_core.messages import BaseMessage

from backend.dummy_retriever import DummyRetriever

# Updated provider mapping to include both OpenAI variants
_PROVIDER_MAP = {
    "azure": AzureChatOpenAI,
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
    "google": ChatGoogleGenerativeAI,
    "fireworks": ChatFireworks,
    "groq": ChatGroq,
    "cohere": ChatCohere,
}

# Updated model mapping
_MODEL_MAP = {
    "azure": "gpt-35-turbo-16k",  # Your Azure deployment name
    "openai": "gpt-4o-mini",  # Vanilla OpenAI model
    "anthropic": "claude-3-haiku-20240307",
    "google": "gemini-pro",
    "fireworks": "mixtral-8x7b",
    "groq": "llama3-70b-8192",
    "cohere": "command",
}

def get_retriever() -> BaseRetriever:
    """Initialize and return retriever"""
    # TODO: Replace with Azure Search when ready
    return DummyRetriever()

def get_azure_retriever() -> BaseRetriever:
    """Initialize and return Azure AI Search retriever"""
    # TODO: Replace with your Azure AI Search configuration
    from langchain_community.retrievers import AzureAISearchRetriever
    
    return AzureAISearchRetriever(
        service_name="your-search-service-name",
        index_name="your-index-name",
        api_key="your-api-key",
        content_key="content",  # The field containing your document content
        top_k=4,  # Number of documents to retrieve
    )

def get_llm(provider: str = "azure", model: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Get LLM instance based on provider and model name."""
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"Unsupported provider: {provider}")
    
    model = model or _MODEL_MAP[provider]
    
    if provider == "azure":
        # Azure OpenAI configuration
        return _PROVIDER_MAP[provider](
            azure_deployment=model,
            openai_api_version="2024-08-01-preview",
            temperature=kwargs.get("temperature", 0),
            **kwargs
        )
    elif provider == "openai":
        # Vanilla OpenAI configuration
        return _PROVIDER_MAP[provider](
            model=model,
            temperature=kwargs.get("temperature", 0),
            **kwargs
        )
    else:
        # Other providers
        return _PROVIDER_MAP[provider](
            model=model,
            temperature=kwargs.get("temperature", 0),
            **kwargs
        )

def create_chain(
    retriever: BaseRetriever,
    provider: str = "azure",
    model: Optional[str] = None,
) -> RunnablePassthrough:
    """Create a chain for question answering with citations."""
    
    # Create base LLM that can be configured at runtime
    base_llm = get_llm(provider, model).configurable_alternatives(
        ConfigurableField(id="model_name"),
        default_key="azure-gpt4",
        azure_gpt4=get_llm("azure", "gpt-4"),
        azure_gpt35=get_llm("azure", "gpt-35-turbo-16k"),
        openai_gpt4=get_llm("openai", "gpt-4o-mini"),
        openai_gpt35=get_llm("openai", "gpt-3.5-turbo")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer questions based on the provided context. 
        If you don't know the answer, say so - do not make up information.
        Use the following pieces of retrieved context to answer the question. Include relevant quotes and citations.
        
        Context: {context}"""),
        ("human", "Question: {question}"),
    ])

    # Construct the chain
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | base_llm  # Now using the configurable LLM
        | StrOutputParser()
    )

    return chain

def _format_docs(docs):
    """Format documents into a string with citations."""
    return "\n\n".join(f"Document [{i}]: {doc.page_content}" 
                      for i, doc in enumerate(docs)) 