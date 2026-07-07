"""Sphinx extension: a ``.. aimmd-params::`` directive.

Generates the full AIMMD parameter reference directly from the
:class:`aimmd.params._fields.ParamsFields` dataclass, so the documented
reference is always in sync with the code. Each field's name, type, default,
and ``metadata['description']`` are emitted as a definition list. Descriptions
are rendered as literal blocks so that the backticks / asterisks / inline
punctuation in the raw field descriptions do not trip reStructuredText
inline-markup parsing.

Imports ``aimmd.params._fields`` at build time; on Read the Docs this succeeds
because ``conf.py`` inserts the repo root on ``sys.path`` and stubs the heavy
runtime dependencies before this extension runs.
"""

from __future__ import annotations

import dataclasses
import math

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList


def _fmt_type(t) -> str:
    """Best-effort human-readable type name for a dataclass field annotation."""
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    return getattr(t, "__name__", None) or str(t).replace("typing.", "")


def _fmt_value(v) -> str:
    """Render a default value without volatile ``repr`` noise (e.g. addresses)."""
    if callable(v) and hasattr(v, "__name__"):
        return v.__name__
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return repr(v)


def _fmt_default(f: "dataclasses.Field"):
    """Return the field default as a string, or ``None`` if the field is required."""
    if f.default is not dataclasses.MISSING:
        return _fmt_value(f.default)
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
        try:
            return _fmt_value(f.default_factory())
        except Exception:
            return f"{getattr(f.default_factory, '__name__', 'factory')}()"
    return None


class AimmdParamsDirective(Directive):
    """Emit a definition-list reference of every ``ParamsFields`` field."""

    has_content = False

    def run(self):
        from aimmd.params._fields import ParamsFields

        lines: list[str] = []
        for f in dataclasses.fields(ParamsFields):
            typ = _fmt_type(f.type)
            default = _fmt_default(f)
            if default is None:
                lines.append(f"``{f.name}`` (*{typ}*, **required**)")
            else:
                lines.append(f"``{f.name}`` (*{typ}*, default ``{default}``)")

            # Definition body: a literal block (verbatim, no inline-markup parsing).
            lines.append("    ::")
            lines.append("")
            desc = (f.metadata.get("description") or "").strip()
            if not desc:
                desc = "(no description)"
            for dline in desc.split("\n"):
                lines.append(("        " + dline) if dline.strip() else "")
            lines.append("")

        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(StringList(lines), self.content_offset, node)
        return node.children


def setup(app):
    app.add_directive("aimmd-params", AimmdParamsDirective)
    return {"parallel_read_safe": True, "version": "0.1.0"}
