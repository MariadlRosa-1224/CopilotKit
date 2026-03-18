"""
FastAPI server for LangGraph agent with MCP tools and header injection.

Entry point that:
1. Imports singleton graph from src.agent (pre-initialized with MCP tools and checkpointer)
2. Wraps it with LangGraphAGUIAgent for SSE streaming
3. Sets up FastAPI endpoint using add_langgraph_fastapi_endpoint
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

# Load environment variables from current directory (.env file)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Import singleton graph with MCP tools and checkpointer already initialized
from src.agent import graph, header_tracker

from copilotkit import LangGraphAGUIAgent
from ag_ui_langgraph import add_langgraph_fastapi_endpoint

print("[Main] ✅ Singleton graph and MCP tools initialized")

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph MCP Agent",
    description="FastAPI server with LangGraph agent, MCP tools, and header injection",
    version="1.0.0"
)

# Add the LangGraph endpoint
add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name="sample_agent",
        description="LangGraph agent with MCP tools and header injection",
        graph=graph,
    ),
    path="/",
)

print("[Main] ✅ FastAPI endpoint configured with LangGraphAGUIAgent")


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8123"))
    print(f"[Main] Starting server on http://0.0.0.0:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

if __name__ == "__main__":
    main()
