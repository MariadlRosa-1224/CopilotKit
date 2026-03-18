"""Interceptor for injecting custom headers into MCP tool calls."""
from typing import Callable, Awaitable
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_header_interceptor import log_tool_execution
import os

class CustomHeaderInterceptor:
    """Injects custom headers into each tool call."""

    def __init__(self):
        self._request_id = None
        self._user_id = None
        self._tool_call_count = 0

    def set_context(self, request_id: str = None, user_id: str = None):
        """Update context for next tool calls."""
        self._request_id = request_id
        self._user_id = user_id

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        """Intercept tool call to inject headers."""
        self._tool_call_count += 1

        # Extract tool name from request
        tool_name = getattr(request, 'name', 'unknown-tool')

        # Log which headers are active during this tool execution
        log_tool_execution(tool_name, stage="request")

        # Build headers from current context
        headers = request.headers.copy() if request.headers else {}

        if self._request_id:
            headers["X-Request-ID"] = self._request_id
        if self._user_id:
            headers["X-User-ID"] = self._user_id

        # Add hardcoded demonstration headers
        headers["X-Tool-Call-Number"] = str(self._tool_call_count)
        headers["X-Custom-Demo"] = "header-injection-demo"

        print(f"[CustomHeaderInterceptor] 📤 Injecting {len(headers)} headers into tool call #{self._tool_call_count}")

        # Create modified request with updated headers
        modified_request = request.override(headers=headers)

        # Execute tool with modified request
        return await handler(modified_request)


def get_client():
    """Create singleton MCP client with custom header interceptor."""
    # Create interceptor
    interceptor = CustomHeaderInterceptor()

    # Server configuration as per langchain-mcp-adapters docs
    server_config = {
        "composio": {
            "transport": "http",
            "url": os.getenv("COMPOSIO_MCP_URL"),
            "headers": {
                "x-api-key": os.getenv("COMPOSIO_API_KEY")
            }
        },
        "excalidraw": {
            "transport": "http",
            "url": os.getenv("EXCALIDRAW_MCP_URL"),
        },
    }

    # Pass to MultiServerMCPClient with interceptor
    client = MultiServerMCPClient(
        server_config,
        tool_interceptors=[interceptor]
    )

    return client, interceptor
