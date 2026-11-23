"""The offline rule of a participant report, shared by the assemble and the end-to-end tests."""

from __future__ import annotations

import re

FORBIDDEN = (
    r'src="http',
    r'href="http',
    r"<script",
    r'<link\b[^>]*rel="stylesheet"',
    r"@import",
    r"url\(\s*['\"]?https?:",
)


def assert_offline(text: str) -> None:
    """Assert that a report loads nothing from outside: no script, no remote resource.

    DOI links (``<a href="https://doi.org/...">``) are allowed; they load nothing.
    """
    stripped = text.replace('href="https://doi.org/', 'href="DOI/')
    for pattern in FORBIDDEN:
        assert not re.search(pattern, stripped), pattern
    hrefs = re.findall(r'\bhref="([^"]*)"', text)
    assert hrefs
    for href in hrefs:
        assert href.startswith(("#", "https://doi.org/", "data:image/")), href
