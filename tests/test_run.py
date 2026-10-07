import fem_run as fr


def test_sweep_values():
    assert fr.sweep_values({"mode": "none"}) == []
    assert fr.sweep_values({"mode": "wavelength", "start": 400, "stop": 500, "n": 3}) == [400.0, 450.0, 500.0]


def test_progress_parsing():
    log = "PROGRESS 1/4\nSTEP 2/10\n"
    assert fr.progress(log) == (1, 4)
    frac, label = fr.progress_info(log)
    assert 0 < frac < 0.25 and "hp-Schritt 2" in label
    assert fr.progress_info("PROGRESS 4/4\nDONE")[0] == 1.0


def test_log_flags_and_library_warnings():
    text = "WARN x\nERROR y\n[2026-10-07] [warning] cell 3 bad\n[2026-10-07] [warning] cell 5 bad\nDONE"
    f = fr.log_flags(text)
    assert f["errors"] == ["ERROR y"] and f["warnings"] == ["WARN x"] and f["done"]
    assert fr.library_warnings(text) == [("cell 3 bad", 2)]
