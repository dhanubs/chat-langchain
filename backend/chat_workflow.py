from langgraph.graph import Graph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import AsyncIterator, Dict, Any, List, Optional
import os
from backend.chain import create_chain, get_retriever
from backend.thread_manager import ThreadManager

class ChatWorkflow:
    def __init__(self, provider: str = "azure", model: str = "gpt-35-turbo-16k", thread_manager: ThreadManager = None):
        self.provider = provider
        self.default_model = model
        self.thread_manager = thread_manager
        self.retriever = get_retriever()
        self.chain = create_chain(
            retriever=self.retriever,
            thread_manager=thread_manager,
            provider=provider,
            model=model
        )
        
        # Create the workflow graph
        self.graph = self.create_graph()
    
    def create_graph(self) -> Graph:
        async def generate_streaming_response(state: Dict[str, Any]):
            messages = state.get("messages", [])
            config = state.get("config", {})
            
            # Extract the last message and chat history
            last_message = messages[-1]["content"] if messages else ""
            chat_history = messages[:-1] if len(messages) > 1 else []
            
            # Use model from config if provided
            model_name = config.get("configurable", {}).get("model_name")
            if model_name:
                self.chain = create_chain(
                    retriever=self.retriever,
                    provider=self.provider,
                    model=model_name,
                    streaming=True
                )
            
            # Use the existing chain with streaming
            response_stream = await self.chain.astream({
                "question": last_message,
                "chat_history": chat_history
            })
            
            full_response = ""
            async for chunk in response_stream:
                if chunk:
                    full_response += chunk
                    yield {
                        "event": "on_llm_new_token",
                        "data": {"token": chunk}
                    }
            
            # Send completion event
            yield {
                "event": "on_llm_end",
                "data": {"output": full_response}
            }
        
        # Create and compile the graph
        workflow = Graph()
        workflow.add_node("generate", generate_streaming_response)
        workflow.set_entry_point("generate")
        
        return workflow.compile()
    
    async def run(self, messages: list, config: Dict = None) -> AsyncIterator[Dict]:
        """Run the graph with messages and config"""
        async for event in self.graph.astream({
            "messages": messages,
            "config": config or {}
        }):
            yield event
    
    async def stream_response(
        self, 
        input: str,
        thread_id: str,
        chat_history: List[Dict] = None,
        config: Optional[Dict] = None
    ) -> AsyncIterator[str]:
        """Stream a response using the chain"""
        # Create the input dictionary
        input_dict = {
            "input": input,
            "chat_history": chat_history or []
        }
        
        # Add session_id to config
        chain_config = {
            "configurable": {
                "session_id": thread_id
            }
        }
        
        # Get the stream with proper config
        stream = self.chain.astream(
            input_dict,
            config=chain_config
        )
        
        # Iterate over the stream
        async for chunk in stream:
            if isinstance(chunk, dict):
                yield chunk.get("output", "")
            else:
                yield str(chunk)
    
    async def generate_response(
        self, 
        message: str,
        thread_id: str,
        chat_history: List[Dict] = None,
        config: Optional[Dict] = None
    ) -> str:
        """Generate a complete response (non-streaming)"""
        # Update chain if different model specified in config
        if config and config.get("model_name"):
            self.chain = create_chain(
                retriever=self.retriever,
                thread_manager=self.thread_manager,
                provider=self.provider,
                model=config["model_name"]
            )
        
        result = await self.chain.ainvoke(
            {
                "input": message,
                "chat_history": chat_history or []
            },
            config={
                "configurable": {
                    "session_id": thread_id
                }
            }
        )
        
        if isinstance(result, dict):
            return result.get("output", "")
        return str(result) 