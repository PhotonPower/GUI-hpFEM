"""Alle Module müssen zusammenpassen (die App prüft das beim Start selbst)."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "fem_gui"


def test_all_files_parse():
    for f in PKG.glob("*.py"):
        ast.parse(f.read_text(encoding="utf-8"), filename=str(f))


def test_required_modules_present():
    for name in ("fem_app", "fem_geometry", "fem_materials", "fem_post", "fem_run", "fem_worker"):
        assert (PKG / f"{name}.py").is_file()
