"""
Simple header context manager for tracking custom headers from frontend.

This allows us to track and log headers passed from the frontend
to ensure they're being properly received and can be used.
"""

import contextvars
from typing import Dict

# Async-aware context for storing custom headers per request
_custom_headers: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "custom_headers", default={}
)


class SimpleHeaderTracker:
    """
    Tracks custom headers passed from the frontend.

    This is a simple tracker that logs headers being passed
    without requiring interceptor support.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._headers = {}

    def set_headers(self, headers: Dict[str, str]):
        """
        Set custom headers from frontend request.

        Args:
            headers: Dictionary of custom headers (e.g., {"test-header": "value"})
        """
        self._headers = headers
        _custom_headers.set(headers)

        if headers:
            print("[HeaderTracker] ✅ Custom headers received from frontend:")
            for key, value in headers.items():
                # Log header name but mask the value for security
                masked_value = value[:10] + "..." if len(value) > 10 else value
                print(f"  - {key}: {masked_value}")
        else:
            print("[HeaderTracker] ℹ️ No custom headers in this request")

    def get_headers(self) -> Dict[str, str]:
        """Get custom headers set for current request."""
        return _custom_headers.get({})

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
    print("[HeaderTracker] Headers tracked and ready for inspection")


def get_request_headers() -> Dict[str, str]:
    """Get custom headers set for current request."""
    return header_tracker.get_headers()
