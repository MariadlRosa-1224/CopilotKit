"""
FastAPI server with dynamic LangGraph agent and hash-based cache invalidation.

Architecture:
1. Startup (initialization):
   - Load MCP tools once with initialization headers
   - Cache tools and header hash

2. Per request (app invocation):
   - Extract and hash request headers
   - If hash matches: reuse cached tools
   - If hash differs: reload tools with new headers
   - Build graph with appropriate tools
   - Tools use current request headers via interceptor context

Benefits:
- Smart caching: reloads only when headers actually change
- Stateless: no cache state assumptions between requests
- Flexible: supports x-reload-tools header to force reload
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


# Middleware to set request headers and inject into agent
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Middleware to extract request headers, set in context, and inject into agent."""
    # Extract headers for MCP context
    request_headers = dict(request.headers)
    set_request_headers(request_headers)
    print(f"[Middleware] 📝 Request headers set in context")

    # Store headers in request state for agent access
    request.state.request_headers = request_headers

    # Inject current request into agent factory for per-request graph building
    request_injector.set_current_request(request)

    response = await call_next(request)
    return response


class DynamicAgentFactory:
    """
    Factory for creating stateless per-request agents.

    Design principles:
    - No cache state assumptions
    - Tools loaded fresh based on REBUILD_TOOLS_PER_REQUEST or x-reload-tools header
    - Each request is independent
    """

    def __init__(self):
        """Initialize the factory."""
        self.request = None

    def set_request_context(self, request: Request):
        """Set request context for next agent creation."""
        self.request = request

    async def run(self, input_data):
        """
        Run agent for this request:
        1. Load tools (fresh or cached based on env var or custom header)
        2. Build graph with tools
        3. Stream responses

        Control via:
        - REBUILD_TOOLS_PER_REQUEST env var (default: true)
        - x-reload-tools custom header (overrides env var)
        """
        if not self.request:
            raise RuntimeError("Request context not set. Use set_request_context() before calling run()")

        # Get request headers from middleware state
        request_headers = getattr(self.request.state, 'request_headers', {})

        # Build graph for this request (tools handled per config)
        graph = await build_request_graph(request_headers_dict=request_headers)
        print(f"[DynamicAgentFactory] 🔄 Request graph built")

        # Wrap in LangGraphAGUIAgent
        agent = LangGraphAGUIAgent(
            name="sample_agent",
            description="LangGraph agent with fresh MCP tools and request-specific headers",
            graph=graph,
        )

        # Stream responses
        event_count = 0
        print(f"[DynamicAgentFactory] 📡 Starting response stream...")
        async for event in agent.run(input_data):
            event_count += 1
            # Log event type for visibility
            if isinstance(event, dict):
                event_type = event.get('type', 'unknown')
                print(f"[ResponseStream] 📤 Event #{event_count}: {event_type}")
            yield event
        print(f"[DynamicAgentFactory] ✅ Response stream complete ({event_count} events)")
        print("="*60 + "\n")

    async def __call__(self, input_data):
        """Alias for run() for compatibility."""
        async for event in self.run(input_data):
            yield event


# Create agent factory
agent_factory = DynamicAgentFactory()


# Create wrapper to inject request context into factory
class RequestInjectorAgent:
    """Wraps factory to inject request before each invocation."""

    def __init__(self, factory: DynamicAgentFactory):
        self.factory = factory
        self._current_request = None

    def set_current_request(self, request: Request):
        """Store current request (called by endpoint wrapper)."""
        self._current_request = request

    async def run(self, input_data):
        """Run with current request context."""
        if self._current_request:
            self.factory.set_request_context(self._current_request)
        async for event in self.factory.run(input_data):
            yield event

    async def __call__(self, input_data):
        """Alias for run() for compatibility."""
        async for event in self.run(input_data):
            yield event


request_injector = RequestInjectorAgent(agent_factory)


# Use add_langgraph_fastapi_endpoint to properly handle the CopilotKit protocol
add_langgraph_fastapi_endpoint(
    app=app,
    agent=request_injector,
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
    print(f"[Main] Architecture: Hash-based cache invalidation with per-request graph building")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

if __name__ == "__main__":
    main()
