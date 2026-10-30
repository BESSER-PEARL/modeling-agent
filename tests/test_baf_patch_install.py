"""Non-Docker runs must use the vendored BAF WebSocket platform.

Live finding (v4 acceptance, 2026-10): ``patches/websocket_platform.py`` was
installed only by the Dockerfile COPY. A local / on-prem ``python
modeling_agent.py`` ran stock BAF 4.3.2: the BYOK key was logged in plain text
(``Session variable user_api_key set to sk-...``) and never used, and there was
no reply outbox or ownership-guarded slot reclaim.

Each check runs in a fresh interpreter: the import order is the thing under test.
"""

import ast
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, ROOT]))
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300,
    )


def test_install_makes_the_agent_use_the_vendored_platform():
    proc = _run(
        """
        import inspect, baf_patch
        baf_patch.install()
        from baf.core import agent
        baf_patch.verify()
        cls = agent.WebSocketPlatform
        print("OK", hasattr(cls, "_flush_outbox"), hasattr(cls, "_buffer_reply"))
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK True True" in proc.stdout


def test_verify_refuses_an_unpatched_platform():
    proc = _run(
        """
        from baf.core import agent
        if hasattr(agent.WebSocketPlatform, "_flush_outbox"):
            print("PREPATCHED")  # Docker image: stock file already replaced
        else:
            import baf_patch
            try:
                baf_patch.verify()
            except RuntimeError:
                print("REFUSED")
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert ("REFUSED" in proc.stdout) or ("PREPATCHED" in proc.stdout)


def test_install_after_agent_import_is_an_error():
    proc = _run(
        """
        import baf.core.agent, baf_patch
        try:
            baf_patch.install()
        except RuntimeError:
            print("RAISED")
        """
    )
    assert "RAISED" in proc.stdout, proc.stderr[-2000:]


def test_entrypoint_installs_the_patch_before_importing_the_agent():
    tree = ast.parse(open(os.path.join(ROOT, "modeling_agent.py"), encoding="utf-8").read())
    install_line = verify_line = agent_import_line = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "baf.core.agent":
            agent_import_line = node.lineno
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and getattr(func.value, "id", None) == "baf_patch":
                if func.attr == "install":
                    install_line = node.lineno
                elif func.attr == "verify":
                    verify_line = node.lineno
    assert agent_import_line is not None
    assert install_line is not None and install_line < agent_import_line
    assert verify_line is not None and verify_line > agent_import_line
