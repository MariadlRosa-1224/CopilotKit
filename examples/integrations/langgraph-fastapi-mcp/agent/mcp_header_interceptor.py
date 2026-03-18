"""
Header context manager for tracking and injecting custom headers into MCP tool calls.

This demonstrates the two-stage header pattern:
1. Hardcoded initialization headers (at app startup)
2. Request-specific headers (per request via contextvar)
"""

import contextvars
from typing import Dict
import json

# Async-aware context for storing custom headers per request
_custom_headers: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "custom_headers", default={}
)

# Stage 1: Hardcoded initialization headers (used when loading tools)
INITIALIZATION_HEADERS = {
    "x-stage": "initialization",
    "x-context": "tool-loading",
    "x-request-id": "init-12345",
}

# Stage 2: Request-specific headers (updated per request)
REQUEST_HEADERS = {
    "x-stage": "request",
    "x-context": "tool-execution",
}


class SimpleHeaderTracker:
    """
    Tracks custom headers passed from the frontend and for MCP tool execution.

    This is a simple tracker that logs headers being passed
    without requiring interceptor support.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._headers = {}
        self._request_count = 0

    def set_headers(self, headers: Dict[str, str]):
        """
        Set custom headers from frontend request.

        Args:
            headers: Dictionary of custom headers (e.g., {"test-header": "value"})
        """
        self._request_count += 1
        self._headers = headers
        _custom_headers.set(headers)

        if headers:
            print(f"\n[HeaderTracker] 📨 REQUEST #{self._request_count} - Custom headers received from frontend:")
            for key, value in headers.items():
                # Log header name but mask the value for security
                masked_value = value[:10] + "..." if len(value) > 10 else value
                print(f"  - {key}: {masked_value}")
        else:
            print(f"\n[HeaderTracker] ℹ️ REQUEST #{self._request_count} - No custom headers in this request")

    def get_headers(self) -> Dict[str, str]:
        """Get custom headers set for current request."""
        return _custom_headers.get({})

    def get_initialization_headers(self) -> Dict[str, str]:
        """Get hardcoded initialization headers (Stage 1)."""
        return INITIALIZATION_HEADERS.copy()

    def get_request_headers_with_context(self) -> Dict[str, str]:
        """Get request headers merged with current request context (Stage 2)."""
        request_headers = REQUEST_HEADERS.copy()
        request_headers["x-request-number"] = str(self._request_count)
        current_headers = self.get_headers()
        request_headers.update(current_headers)
        return request_headers

    def clear_headers(self):
        """Clear custom headers."""
        _custom_headers.set({})
        self._headers = {}


# Global tracker instance
header_tracker = SimpleHeaderTracker()


def get_interceptor() -> SimpleHeaderTracker:
    """Get the global header tracker instance."""
    return header_tracker


def set_request_headers(headers: Dict[str, str]):
    """
    Set custom headers from frontend request.

    Call this in middleware to track headers being passed from frontend.

    Args:
        headers: Dictionary with headers (e.g., {"test-header": "my-value"})
    """
    header_tracker.set_headers(headers)
    print("[HeaderTracker] ✅ Headers tracked and ready for inspection")


def get_request_headers() -> Dict[str, str]:
    """Get custom headers set for current request."""
    return header_tracker.get_headers()


def log_tool_execution(tool_name: str, stage: str = "request"):
    """
    Log tool execution with current headers context.

    This demonstrates which headers are active when tools execute.

    Args:
        tool_name: Name of the tool being executed
        stage: "initialization" or "request"
    """
    if stage == "initialization":
        headers = header_tracker.get_initialization_headers()
        print(f"\n[ToolExecution] 🔧 TOOL: {tool_name}")
        print(f"[ToolExecution] 📍 STAGE: {stage} (Tool Loading)")
        print(f"[ToolExecution] 📦 HEADERS ACTIVE:")
        for key, value in headers.items():
            print(f"  - {key}: {value}")
    else:
        headers = header_tracker.get_request_headers_with_context()
        print(f"\n[ToolExecution] 🔧 TOOL: {tool_name}")
        print(f"[ToolExecution] 📍 STAGE: {stage} (Tool Execution)")
        print(f"[ToolExecution] 📦 HEADERS ACTIVE:")
        for key, value in headers.items():
            print(f"  - {key}: {value}")
        print(f"[ToolExecution] 📋 REQUEST: #{header_tracker._request_count}")
