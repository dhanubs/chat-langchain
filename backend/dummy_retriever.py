from typing import Optional, Dict, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class DummyRetriever(BaseRetriever):
    """A simple retriever that returns dummy content for testing."""

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "test"}
            ),
            Document(
                page_content="LangChain provides many modules that can be used to build language model applications.",
                metadata={"source": "test"}
            )
        ]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "test"}
            ),
            Document(
                page_content="LangChain provides many modules that can be used to build language model applications.",
                metadata={"source": "test"}
            )
        ]