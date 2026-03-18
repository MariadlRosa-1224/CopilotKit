"""
LangGraph Agent with MCP Tool Support

This module integrates:
1. MCP tools from configured servers (Composio, Excalidraw, etc.)
2. Header injection for MCP tool calls
3. LangGraph state graph with ReAct pattern
4. MemorySaver checkpointer for conversation state persistence

Tools and graph are initialized as singletons at module load time.
"""

import asyncio
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from copilotkit import CopilotKitState
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import Literal

from src.util import should_route_to_tool_node
from mcp_client import get_client
from mcp_header_interceptor import get_interceptor

# Load environment variables from parent directory (.env file)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

print("[Agent] Initializing LangGraph agent with MCP support...")

# Initialize header tracker as singleton
header_tracker = get_interceptor()
print("[Agent] ✅ Header tracker initialized")


class AgentState(CopilotKitState):
    """
    Agent state extending CopilotKitState with custom fields.
    Inherits messages, copilotkit actions from CopilotKitState.
    """

    proverbs: List[str]


# Load MCP tools at module initialization
async def _load_mcp_tools():
    """Load MCP tools from configured servers."""
    try:
        client, interceptor = get_client()
        tools = await client.get_tools()
        print(f"[Agent] ✅ Loaded {len(tools)} MCP tools")
        return tools
    except Exception as e:
        print(f"[Agent] ⚠️  Error loading MCP tools: {e}")
        return []


# Initialize tools at module load time
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    mcp_tools = loop.run_until_complete(_load_mcp_tools())
    loop.close()
except Exception as e:
    print(f"[Agent] ⚠️  Failed to load tools: {e}")
    mcp_tools = []


async def chat_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern.
    Uses both frontend tools and MCP tools.
    """

    # 1. Define the model
    model = ChatOpenAI(model="gpt-4o")

    # 2. Bind the tools to the model (frontend tools + MCP tools)
    fe_tools = state.get("copilotkit", {}).get("actions", [])
    model_with_tools = model.bind_tools(
        [
            *fe_tools,
            *mcp_tools,
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
workflow.add_node("tool_node", ToolNode(tools=mcp_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

print("[Agent] ✅ Graph built with MCP tools and checkpointer")
print(f"[Agent] ✅ Ready with {len(mcp_tools)} MCP tools")
