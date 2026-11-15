"""Moved to ``dirconn.diagnostics.order_stability``; this path is removed in 0.4.0."""

from dirconn._compat import redirect

__getattr__, __all__ = redirect(__name__, "dirconn.diagnostics.order_stability")
