"""
LangGraph Agent Factory with MCP Tool Support and Hash-Based Cache Invalidation

Architecture - Hash-based smart caching:
1. Module initialization (app startup):
   - Initialize singletons: header_tracker, mcp_client, checkpointer
   - Load and cache tools with initialization headers

2. Per request (app invocation):
   - Hash request headers + custom headers
   - If hash matches previous request: reuse cached tools
   - If hash differs: reload fresh tools with new headers
   - Build graph with tools (cached or fresh based on hash)

Key: Cache invalidation on header change, not on every request
Control: Hash detects when headers actually change
Benefit: Reduces unnecessary tool reloads while staying stateless
"""

import asyncio
import os
import hashlib
import json
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import Literal

from copilotkit import CopilotKitState
from src.util import should_route_to_tool_node
from mcp_client import get_client
from mcp_header_interceptor import get_interceptor, set_request_headers

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Control flag: Load tools fresh on every request (no caching)
REBUILD_TOOLS_PER_REQUEST = os.getenv("REBUILD_TOOLS_PER_REQUEST", "true").lower() == "true"

if REBUILD_TOOLS_PER_REQUEST:
    print("[AgentFactory] ⚙️  REBUILD_TOOLS_PER_REQUEST=true (tools loaded fresh every request)")
else:
    print("[AgentFactory] ⚙️  REBUILD_TOOLS_PER_REQUEST=false (tools cached at startup only)")

print("[AgentFactory] Initializing module-level singletons...")

# Module-level singletons (created once at app startup)
header_tracker = get_interceptor()
mcp_client, mcp_interceptor = get_client()
checkpointer = MemorySaver()

print("[AgentFactory] ✅ Header tracker initialized")
print("[AgentFactory] ✅ MCP client initialized")
print("[AgentFactory] ✅ Checkpointer initialized")


class AgentState(CopilotKitState):
    """Agent state extending CopilotKitState with custom fields."""
    proverbs: list


# Cache management with hash-based invalidation
_cached_tools = None
_cached_headers_hash = None


def _hash_headers(headers: dict = None) -> str:
    """
    Generate hash of request headers for cache invalidation.

    Args:
        headers: Dictionary of request headers

    Returns:
        SHA256 hash of headers as hex string
    """
    if not headers:
        headers = {}

    # Create a stable JSON representation of headers
    headers_json = json.dumps(headers, sort_keys=True)
    return hashlib.sha256(headers_json.encode()).hexdigest()


def _should_reload_tools(current_headers: dict = None) -> tuple[bool, str]:
    """
    Determine if tools should be reloaded based on header hash.

    Args:
        current_headers: Current request headers

    Returns:
        Tuple of (should_reload: bool, reason: str)
    """
    global _cached_headers_hash

    current_hash = _hash_headers(current_headers)

    # First request - no cache yet
    if _cached_headers_hash is None:
        return True, "first-request"

    # Headers changed - reload needed
    if current_hash != _cached_headers_hash:
        return True, f"hash-changed ({_cached_headers_hash[:8]}...→{current_hash[:8]}...)"

    # Headers same - use cache
    return False, f"hash-match ({current_hash[:8]}...)"


async def _load_tools_with_headers(headers: dict = None, stage_name: str = "request"):
    """Load MCP tools with current headers context and update cache hash."""
    global _cached_tools, _cached_headers_hash

    try:
        tools = await mcp_client.get_tools()

        # Update cache and hash on successful load
        _cached_tools = tools
        _cached_headers_hash = _hash_headers(headers)

        print(f"[AgentFactory] ✅ Loaded {len(tools)} MCP tools ({stage_name})")
        print(f"[AgentFactory] 📍 Header hash: {_cached_headers_hash[:8]}... (cached for reuse)")
        return tools
    except Exception as e:
        print(f"[AgentFactory] ⚠️  Error loading MCP tools ({stage_name}): {e}")
        return []


async def _load_tools_at_startup():
    """Load MCP tools once at app startup with initialization headers."""
    try:
        # Initialize request headers for tool loading
        init_headers = {
            "x-initialization": "true",
            "x-startup": "true"
        }
        set_request_headers(init_headers)

        return await _load_tools_with_headers(headers=init_headers, stage_name="startup")
    except Exception as e:
        print(f"[AgentFactory] ⚠️  Error loading MCP tools at startup: {e}")
        return []


# Global graph (rebuilt per request)
_current_request_graph = None


def _build_graph(tools: list):
    """
    Build a LangGraph with the provided tools.

    This is called:
    1. At app startup with initialization tools
    2. Per request with same cached tools (but different request headers in context)
    """

    async def chat_node(
        state: AgentState, config: RunnableConfig
    ) -> Command[Literal["tool_node", "__end__"]]:
        """Standard chat node based on the ReAct design pattern."""

        # 1. Define the model
        model = ChatOpenAI(model="gpt-4o")

        # 2. Bind the tools to the model (frontend tools + MCP tools)
        fe_tools = state.get("copilotkit", {}).get("actions", [])
        model_with_tools = model.bind_tools(
            [
                *fe_tools,
                *tools,
            ]
        )

        # 3. Define the system message
        system_message = SystemMessage(
            content=f"You are a helpful assistant with access to various tools. Current proverbs: {state.get('proverbs', [])}. Use the available tools to help accomplish tasks."
        )

        # 4. Run the model to generate a response
        response = await model_with_tools.ainvoke(
            [
                system_message,
                *state["messages"],
            ],
            config,
        )

        tool_calls = response.tool_calls
        if tool_calls and should_route_to_tool_node(tool_calls, fe_tools):
            return Command(goto="tool_node", update={"messages": response})

        # 5. We've handled all tool calls, so we can end the graph.
        return Command(goto="__end__", update={"messages": response})

    # Define the workflow graph
    workflow = StateGraph(AgentState)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tool_node", ToolNode(tools=tools))
    workflow.add_edge("tool_node", "chat_node")
    workflow.set_entry_point("chat_node")

    graph = workflow.compile(checkpointer=checkpointer)
    return graph


async def initialize_at_startup():
    """
    Initialize app at startup:
    - Load tools with initialization headers (if REBUILD_TOOLS_PER_REQUEST=false)
    - Otherwise just initialize checkpointer
    """
    print("\n" + "="*60)
    print("[AgentFactory] 🚀 === STAGE 1: STARTUP INITIALIZATION ===")
    print("="*60)

    if not REBUILD_TOOLS_PER_REQUEST:
        # Pre-load tools once if not rebuilding per request
        from mcp_header_interceptor import log_tool_execution
        log_tool_execution("MCP-Clients-Loading", stage="initialization")
        await _load_tools_at_startup()
        print(f"[AgentFactory] ✅ Tools cached for all requests")
    else:
        print(f"[AgentFactory] ℹ️  Tools will be loaded fresh on each request")

    print("="*60 + "\n")


async def build_request_graph(request_headers_dict: dict = None):
    """
    Build graph for request invocation with hash-based cache invalidation.

    Algorithm:
    1. Hash current request headers
    2. If hash matches previous request: reuse cached tools
    3. If hash differs or no cache: reload fresh tools
    4. Build graph with appropriate tools

    Can be overridden by custom header: x-reload-tools (forces fresh load)

    Args:
        request_headers_dict: Headers from HTTP request for context

    Returns:
        Compiled LangGraph ready for execution
    """
    global _current_request_graph, _cached_tools, _cached_headers_hash

    print("\n[AgentFactory] 📍 === STAGE 2: PER-REQUEST GRAPH BUILD (Hash-based caching) ===")

    # Check for override: custom header "x-reload-tools" forces fresh load
    force_reload = request_headers_dict and request_headers_dict.get("x-reload-tools", "").lower() == "true"

    # Determine if tools should be reloaded based on header hash
    should_reload, reason = _should_reload_tools(request_headers_dict)

    if force_reload:
        print("[AgentFactory] 🔄 Force reload triggered by x-reload-tools header")
        set_request_headers(request_headers_dict or {})
        tools = await _load_tools_with_headers(headers=request_headers_dict, stage_name="request-forced")
    elif should_reload:
        print(f"[AgentFactory] 🔄 Reloading tools - Reason: {reason}")
        set_request_headers(request_headers_dict or {})
        tools = await _load_tools_with_headers(headers=request_headers_dict, stage_name="request")
    else:
        print(f"[AgentFactory] ♻️  Reusing cached tools - Reason: {reason}")
        tools = _cached_tools if _cached_tools else []

    # Build fresh graph with tools for this request
    _current_request_graph = _build_graph(tools)
    print("[AgentFactory] ✅ Graph ready for request execution\n")

    return _current_request_graph


def get_current_graph():
    """Get the current request graph."""
    return _current_request_graph
