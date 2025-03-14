import asyncio
from typing import Optional, Dict, Any
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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from backend.dummy_retriever import DummyRetriever
from backend.thread_manager import ThreadManager
from backend.models import ThreadChatMessageHistory

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
    "azure": "gpt-4o-mini",  # Your Azure deployment name
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

def get_llm(provider: str = "azure", model: Optional[str] = None, streaming: bool = False, **kwargs) -> BaseChatModel:
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
    thread_manager: ThreadManager,
    provider: str = "azure",
    model: str = None,
    streaming: bool = False
) -> Any:
    """Create a RAG chain with message history that strictly uses retrieved context"""
    
    # Initialize the language model
    llm = get_llm(provider, model, streaming)
    
    # Create the prompt template with strict context adherence
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable AI assistant whose primary role is to answer questions based STRICTLY on the provided context from documents. Your responses must be:

1. STRICTLY based on the information present in the provided context
2. Clear and direct, citing specific parts of the context when relevant
3. Honest about limitations - if the answer isn't in the context, say so clearly
4. Considerate of the chat history for contextual understanding

DO NOT:
- Make assumptions beyond the provided context
- Add external knowledge not present in the documents
- Try to be helpful by making up information

If the context doesn't contain enough information to fully answer the question, say: "I cannot provide a complete answer based on the available context. The documents only mention [relevant partial information if any]."

Previous conversation for context:
{chat_history}

Relevant document excerpts:
{context}"""),
        ("human", "{question}")
    ])
    
    # Create a chain that combines retrieval and response generation
    chain = (
        {
            "context": lambda x: _format_docs(retriever.get_relevant_documents(x["question"])),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: format_chat_history(x.get("chat_history", []))
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Create sync function using run_until_complete
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return ThreadChatMessageHistory(thread_manager.get_sync(session_id))
    
    # Wrap with message history
    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

def format_chat_history(chat_history):
    """Format chat history into a string."""
    formatted_messages = []
    for message in chat_history:
        if isinstance(message, dict):
            # Handle dict format
            role = message.get("role", "")
            content = message.get("content", "")
            formatted_messages.append(f"{role.capitalize()}: {content}")
        else:
            # Handle Message objects
            formatted_messages.append(f"{message.type.capitalize()}: {message.content}")
    return "\n".join(formatted_messages) if formatted_messages else "No previous conversation."

def _format_docs(docs):
    """Format documents into a string with clear section breaks and citations."""
    if not docs:
        return "No relevant documents found."
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        # Add clear section breaks and document numbers
        formatted_docs.append(f"[Document {i}]\n{doc.page_content}\n")
    
    return "\n---\n".join(formatted_docs) 