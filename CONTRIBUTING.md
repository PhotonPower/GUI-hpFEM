# Mitarbeit

* Python ≥ 3.10, Zeilenlänge bis 160 Zeichen, `ruff check .` und `pytest` müssen laufen (siehe [docs/ENTWICKLUNG.md](docs/ENTWICKLUNG.md)).
* Benutzeroberfläche und Anleitung sind deutsch; Code-Bezeichner und Docstrings dürfen deutsch oder englisch sein, je Datei einheitlich.
* Änderungen am Format von `job.json` oder `results.json` betreffen App und Worker: beide im selben Commit ändern und in `docs/ARCHITEKTUR.md` vermerken.
* Neue Rechenfunktionen aus hp-FEM nur mit Rückfall für ältere Builds einbauen oder die Mindestversion im README nennen.
* Einträge in `CHANGELOG.md` nicht vergessen.
