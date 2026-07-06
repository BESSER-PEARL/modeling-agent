"""Tests for the auto-generate GUI completion message (#3).

The "Auto-generate" GUI path previously only emitted "Generating GUI…" and
never confirmed completion. ``confirmation._build_auto_gui_message`` now
resolves the class diagram and returns a success message naming the pages that
were created.

``confirmation.py`` hard-imports the ``baf`` framework at module load, which is
not installed in this test environment (it is the source of the documented
baseline collection errors). To exercise ``_build_auto_gui_message`` WITHOUT
leaking a global ``baf`` stub into ``sys.modules`` / ``sys.meta_path`` — which
would change the outcome of the other test modules collected in the same
pytest process — each assertion runs in an isolated subprocess that installs a
``baf`` import hook only for itself.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# Driver template: installs a baf import hook, imports confirmation, patches the
# class-diagram resolver, builds the message, and prints it on a marker line.
_DRIVER = textwrap.dedent(
    """
    import sys, types, importlib.abc, importlib.machinery
    from unittest.mock import MagicMock

    sys.path.insert(0, {src!r})

    class _BafStub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "baf" or fullname.startswith("baf."):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None
        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []
            mod.__getattr__ = lambda name: MagicMock()
            return mod
        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _BafStub())

    import confirmation
    import utilities.model_resolution as mr

    MODEL = {model!r}
    RAISE = {raise_!r}

    if RAISE:
        def _boom(request):
            raise RuntimeError("no context")
        mr.resolve_class_diagram = _boom
    else:
        mr.resolve_class_diagram = lambda request: MODEL

    msg = confirmation._build_auto_gui_message(request=MagicMock())
    print("===MSG===" + msg)
    """
)


def _run_message(model=None, raise_=False) -> str:
    """Build the auto-GUI message in an isolated subprocess; return the message."""
    code = _DRIVER.format(src=_SRC, model=model, raise_=raise_)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    for line in proc.stdout.splitlines():
        if line.startswith("===MSG==="):
            return line[len("===MSG==="):]
    raise AssertionError(f"no message produced; stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


# A ClassDiagram model with two real classes and one enumeration. Only the two
# Class elements should become pages (the Genre enum is excluded).
_CLASS_MODEL = {
    "elements": {
        "c1": {"id": "c1", "name": "Book", "type": "Class"},
        "c2": {"id": "c2", "name": "Member", "type": "Class"},
        "e1": {"id": "e1", "name": "Genre", "type": "Enumeration"},
    },
    "relationships": {},
}


def test_message_confirms_completion_and_names_pages():
    msg = _run_message(_CLASS_MODEL)
    # Confirms it is DONE (not just "Generating…").
    assert "screen" in msg
    assert "Generating" not in msg
    # Names the two real classes as pages; the enum is not a page.
    assert "Book" in msg
    assert "Member" in msg
    assert "Genre" not in msg
    # Page count reflects the two classes.
    assert "2" in msg


def test_message_falls_back_when_no_classes():
    msg = _run_message({"elements": {}, "relationships": {}})
    # Still confirms completion even without resolvable page names.
    assert "screen" in msg
    assert "Generating" not in msg


def test_message_falls_back_when_resolver_raises():
    msg = _run_message(raise_=True)
    # Never raises; degrades to a generic completion message.
    assert "screen" in msg


def test_message_truncates_long_page_lists():
    elements = {
        f"c{i}": {"id": f"c{i}", "name": f"Class{i}", "type": "Class"}
        for i in range(9)
    }
    msg = _run_message({"elements": elements, "relationships": {}})
    assert "9" in msg          # total count
    assert "more" in msg       # "(+N more)" suffix for >6 pages
