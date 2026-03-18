"""
FastAPI server with dynamic LangGraph agent and request-specific header injection.

Architecture:
1. Startup (initialization):
   - Load MCP tools once with initialization headers
   - Build and cache initialization graph

2. Per request (app invocation):
   - Set request-specific headers in contextvar via middleware
   - Rebuild graph with same cached tools
   - Tools use current request headers via interceptor context
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables from current directory (.env file)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from src.agent_factory import initialize_at_startup, build_request_graph
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


# Middleware to set request headers before endpoint processes
@app.middleware("http")
async def set_headers_middleware(request: Request, call_next):
    """Middleware to extract and set request headers in context."""
    # Extract headers for MCP context
    request_headers = dict(request.headers)
    set_request_headers(request_headers)
    print(f"[Middleware] 📝 Request headers set in context")

    response = await call_next(request)
    return response


class RequestAwareDynamicAgent:
    """
    Wraps LangGraphAGUIAgent to rebuild graph per request with current headers.

    This allows:
    - Cached tools loaded once at startup
    - Per-request graph rebuild to apply request-specific headers
    - Tools execute with contextvar-aware headers from interceptor
    """

    async def run(self, input_data):
        """
        Run agent for this request:
        1. Rebuild graph with same cached tools
        2. Stream responses using rebuilt graph

        Note: Request headers are already set in context by middleware
        """
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
        event_count = 0
        print(f"[RequestAwareDynamicAgent] 📡 Starting response stream...")
        async for event in agent.run(input_data):
            event_count += 1
            # Log event type for visibility
            if isinstance(event, dict):
                event_type = event.get('type', 'unknown')
                print(f"[ResponseStream] 📤 Event #{event_count}: {event_type}")
            yield event
        print(f"[RequestAwareDynamicAgent] ✅ Response stream complete ({event_count} events)")
        print("="*60 + "\n")

    async def __call__(self, input_data):
        """Alias for run() for compatibility."""
        async for event in self.run(input_data):
            yield event


# Create request-aware agent
request_agent = RequestAwareDynamicAgent()

# Use add_langgraph_fastapi_endpoint to properly handle the CopilotKit protocol
add_langgraph_fastapi_endpoint(
    app=app,
    agent=request_agent,
    path="/",
)


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
