# Entwicklung

## Umgebung

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Unter Linux braucht Gmsh Systembibliotheken (`libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1`).

## Tests

`pytest` prüft Materialien, Modell/Vorlagen, Job-/Log-Auswertung und dass alle Dateien zusammenpassen. Es sind keine hpfem-Rechnungen enthalten. Ein Rechentest gegen die Referenzwerte der Anleitung (Ag-Lamellengitter, R₀ = 0,7800, R₋₁ = 0,0801) gehört in eine Umgebung mit gebautem hpfem und ist noch nicht automatisiert.

## Neue Vorlage anlegen

In `fem_geometry.presets()` ein Modell-Dict ergänzen. `tests/test_geometry.py` prüft alle Vorlagen automatisch.

## Neues Material anlegen

Neue Tabelle als CSV `wellenlänge[µm], n, k` in hp-FEM unter `python/hpfem/data` und in `hpfem.materials` registrieren. Danach den Namen in `fem_materials.LIBRARY`, `_TABULATED` bzw. `_SELLMEIER` und in `fem_app.LIB_CHOICES` eintragen.

## Versionsprüfung der Dateien

`fem_app.py` hat am Anfang eine Liste der erwarteten Funktionen je Modul (`_REQUIRED`). Wer eine öffentliche Funktion hinzufügt oder umbenennt, ergänzt sie dort.

## Continuous Integration

`docs/ci-workflow.yml` ist der GitHub-Actions-Workflow (ruff und pytest unter Python 3.10 und 3.12). Er liegt hier, weil der zum Einrichten benutzte Token keinen `workflow`-Scope hatte. Zum Aktivieren die Datei nach `.github/workflows/ci.yml` verschieben (Weboberfläche oder Token mit `workflow`-Scope).
