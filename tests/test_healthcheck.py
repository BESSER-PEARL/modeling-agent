"""The Docker health check fails when the agent nears its open-file limit."""
from app import healthcheck


def test_well_under_the_limit_is_healthy():
    ok, detail = healthcheck.fd_verdict(97, 1024)
    assert ok
    assert detail == "97 of 1024 file descriptors open"


def test_near_the_limit_is_unhealthy():
    # Production on 2026-11-23: 1,013 of 1,024, and every new handshake hung.
    ok, detail = healthcheck.fd_verdict(1013, 1024)
    assert not ok
    assert detail == "1013 of 1024 file descriptors open"


def test_unknown_counts_do_not_fail_the_check():
    assert healthcheck.fd_verdict(None, 1024)[0]
    assert healthcheck.fd_verdict(500, None)[0]


def test_reads_the_soft_limit_from_proc(tmp_path, monkeypatch):
    limits = tmp_path / "limits"
    limits.write_text(
        "Limit                     Soft Limit           Hard Limit           Units     \n"
        "Max open files            1024                 524288               files     \n"
    )
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(limits if path == "/proc/1/limits" else path, *a, **k),
    )
    assert healthcheck.soft_fd_limit() == 1024


def test_main_reports_a_closed_port(monkeypatch, capsys):
    monkeypatch.setattr(healthcheck, "port_accepts", lambda *a, **k: False)
    assert healthcheck.main() == 1
    assert "does not accept" in capsys.readouterr().out
