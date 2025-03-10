"""Load html from files, clean up, split, ingest into Weaviate."""
from datetime import datetime
import logging
import os
import re
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.indexes import SQLRecordManager, index
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain_community.vectorstores.azuresearch import AzureSearch

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.embeddings import get_embeddings_model
from backend.parser import langchain_docs_extractor
from backend.config import settings
from backend.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def metadata_extractor(
    meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "")
        if description_element
        else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langgraph_docs():
    return SitemapLoader(
        "https://langchain-ai.github.io/langgraph/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={"parse_only": SoupStrainer(name=("article", "title"))},
        meta_function=lambda meta, soup: metadata_extractor(
            meta, soup, title_suffix=" | 🦜🕸️LangGraph"
        ),
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError(
            "Input should be either BeautifulSoup object or an HTML string"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

        docs_from_documentation = load_langchain_docs()
        logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
        docs_from_api = load_api_docs()
        logger.info(f"Loaded {len(docs_from_api)} docs from API")
        docs_from_langsmith = load_langsmith_docs()
        logger.info(f"Loaded {len(docs_from_langsmith)} docs from LangSmith")
        docs_from_langgraph = load_langgraph_docs()
        logger.info(f"Loaded {len(docs_from_langgraph)} docs from LangGraph")

        docs_transformed = text_splitter.split_documents(
            docs_from_documentation
            + docs_from_api
            + docs_from_langsmith
            + docs_from_langgraph
        )
        docs_transformed = [
            doc for doc in docs_transformed if len(doc.page_content) > 10
        ]

        # We try to return 'source' and 'title' metadata when querying vector store and
        # Weaviate will error at query time if one of the attributes is missing from a
        # retrieved document.
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                doc.metadata["source"] = ""
            if "title" not in doc.metadata:
                doc.metadata["title"] = ""

        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )

        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
        logger.info(
            f"LangChain now has this many vectors: {num_vecs}",
        )


async def ingest_documents(
    directory_path: str,
    recursive: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    file_extensions: Optional[List[str]] = None,
    parallel: bool = True,
    max_concurrency: int = 5
) -> Dict[str, Any]:
    """
    Process all supported documents in a directory and add them to the vector store.
    
    Args:
        directory_path: Path to the directory containing documents
        recursive: Whether to search subdirectories
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        file_extensions: List of file extensions to process (if None, process all supported types)
        parallel: Whether to process files in parallel
        max_concurrency: Maximum number of files to process concurrently
        
    Returns:
        Dict: Processing statistics
    """
    # Initialize document processor
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_concurrency=max_concurrency
    )
    
    # Process all documents in the directory
    return await processor.process_directory(
        directory_path=directory_path,
        recursive=recursive,
        file_extensions=file_extensions,
        parallel=parallel
    )


if __name__ == "__main__":
    # Use this to ingest documents
    ingest_docs()
    # To ingest multiple document types:
    # import asyncio
    # asyncio.run(ingest_documents("path/to/your/documents"))
