"""
FastAPI server with dynamic LangGraph agent and request-specific header injection.

Architecture:
1. Startup (initialization):
   - Load MCP tools once with initialization headers
   - Build and cache initialization graph

2. Per request (app invocation):
   - Set request-specific headers in contextvar
   - Rebuild graph with same cached tools
   - Tools use current request headers via interceptor context
"""

import os
import warnings
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables from current directory (.env file)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from src.agent_factory import initialize_at_startup, build_request_graph, get_current_graph, header_tracker
from mcp_header_interceptor import set_request_headers

from copilotkit import LangGraphAGUIAgent
from ag_ui_langgraph import add_langgraph_fastapi_endpoint

print("[Main] ✅ Module imports successful")


# App lifespan for startup initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    # Startup: Initialize graph and load tools
    print("[Main] 🚀 Starting up - initializing agent...")
    await initialize_at_startup()
    print("[Main] ✅ Agent initialization complete")
    yield
    # Shutdown
    print("[Main] 🛑 Shutting down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LangGraph MCP Agent",
    description="FastAPI server with dynamic LangGraph agent, MCP tools, and request-specific header injection",
    version="1.0.0",
    lifespan=lifespan
)


class RequestAwareDynamicAgent:
    """
    Wraps LangGraphAGUIAgent to rebuild graph per request with current headers.

    This allows:
    - Cached tools loaded once at startup
    - Per-request graph rebuild to apply request-specific headers
    - Tools execute with contextvar-aware headers from interceptor
    """

    async def run(self, input_data, request_headers: dict):
        """
        Run agent for this request:
        1. Set request headers in contextvar
        2. Rebuild graph with same cached tools
        3. Stream responses using rebuilt graph
        """
        # Set request headers in contextvar (interceptor will use these)
        set_request_headers(request_headers)
        print(f"[RequestAwareDynamicAgent] 📝 Request headers set in context")

        # Rebuild graph for this request (uses cached tools, new headers context)
        graph = build_request_graph()
        print(f"[RequestAwareDynamicAgent] 🔄 Request graph rebuilt")

        # Wrap in LangGraphAGUIAgent
        agent = LangGraphAGUIAgent(
            name="sample_agent",
            description="LangGraph agent with dynamic MCP tools and request-specific headers",
            graph=graph,
        )

        # Stream responses
        async for event in agent.run(input_data):
            yield event


# Create request-aware agent
request_agent = RequestAwareDynamicAgent()


@app.post("/")
async def agent_endpoint(input_data, request: Request):
    """
    Main agent endpoint:
    1. Extract headers from HTTP request
    2. Build graph with current request headers
    3. Stream responses
    """
    # Extract headers for MCP context
    request_headers = dict(request.headers)

    # Run agent with request headers
    async for chunk in request_agent.run(input_data, request_headers):
        yield chunk


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "sample_agent",
        "architecture": "dynamic-graph-per-request"
    }


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8123"))
    print(f"[Main] Starting server on http://0.0.0.0:{port}")
    print(f"[Main] Architecture: Tools cached at startup, graph rebuilt per request")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

if __name__ == "__main__":
    main()
