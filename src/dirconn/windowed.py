"""Moved to ``dirconn.methods.timevarying.windowed``; this path is removed in 0.4.0."""

from dirconn._compat import redirect

__getattr__, __all__, __dir__ = redirect(__name__, "dirconn.methods.timevarying.windowed")
