"""
LangGraph Agent Factory with MCP Tool Support and Header Injection

Architecture:
1. Module initialization (app startup):
   - Initialize singletons: header_tracker, mcp_client, checkpointer
   - Load tools using initialization headers
   - Build and cache graph with those tools

2. Per request (app invocation):
   - Set new request headers in contextvar (via interceptor)
   - Rebuild graph with same cached tools
   - Tools use current request headers via interceptor context
"""

import asyncio
import os
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

# Feature flag: Always rebuild graph with tools (allows for token invalidation, etc)
ALWAYS_REBUILD_GRAPH_WITH_TOOLS = os.getenv("ALWAYS_REBUILD_GRAPH_WITH_TOOLS", "false").lower() == "true"

if ALWAYS_REBUILD_GRAPH_WITH_TOOLS:
    print("[AgentFactory] ⚙️  ALWAYS_REBUILD_GRAPH_WITH_TOOLS=true (graphs rebuilt with tools every request)")

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


# Global tools cache (loaded once at startup)
_cached_tools = None


async def _load_tools_at_startup():
    """Load MCP tools once at app startup with initialization headers."""
    global _cached_tools

    try:
        # Initialize request headers for tool loading
        set_request_headers({
            "x-initialization": "true",
            "x-startup": "true"
        })

        tools = await mcp_client.get_tools()
        _cached_tools = tools
        print(f"[AgentFactory] ✅ Loaded {len(tools)} MCP tools at startup")
        return tools
    except Exception as e:
        print(f"[AgentFactory] ⚠️  Error loading MCP tools: {e}")
        _cached_tools = []
        return []


# Global graph cache (built twice: at startup and per request)
_initialization_graph = None
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
    1. Load tools with initialization headers
    2. Build and cache initialization graph
    """
    global _initialization_graph

    print("\n" + "="*60)
    print("[AgentFactory] 🚀 === STAGE 1: STARTUP INITIALIZATION ===")
    print("="*60)

    from mcp_header_interceptor import log_tool_execution
    log_tool_execution("MCP-Clients-Loading", stage="initialization")

    tools = await _load_tools_at_startup()
    _initialization_graph = _build_graph(tools)
    print(f"[AgentFactory] ✅ Initialization graph built with {len(tools)} tools")
    print("="*60 + "\n")
    return _initialization_graph


def build_request_graph():
    """
    Build graph for request invocation:
    - Uses cached tools (loaded at startup)
    - Intercepts will use current request headers from contextvar
    - Can optionally reload tools if ALWAYS_REBUILD_GRAPH_WITH_TOOLS=true
    """
    global _current_request_graph, _cached_tools

    print("[AgentFactory] 📍 === STAGE 2: PER-REQUEST GRAPH BUILD ===")

    set_request_headers({
            "x-initialization": "false",
            "x-startup": "false"
        })

    # Determine which tools to use
    tools_to_use = _cached_tools if _cached_tools is not None else []

    if ALWAYS_REBUILD_GRAPH_WITH_TOOLS and _cached_tools:
        print(f"[AgentFactory] 🔄 ALWAYS_REBUILD_GRAPH_WITH_TOOLS enabled - will reload tools on next request")
        # TODO: Implement async tool reloading on next request
        # This could be triggered by a custom header like "x-reload-tools: true"

    if tools_to_use:
        print(f"[AgentFactory] ✅ Building fresh graph with {len(tools_to_use)} cached tools")
        _current_request_graph = _build_graph(tools_to_use)
    else:
        print("[AgentFactory] ⚠️  No cached tools available, building with empty tools")
        _current_request_graph = _build_graph([])

    print("[AgentFactory] ✅ Graph ready for request execution")

    return _current_request_graph


def get_current_graph():
    """Get the current request graph."""
    return _current_request_graph
