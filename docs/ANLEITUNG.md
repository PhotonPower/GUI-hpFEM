# FEM-Modellwerkstatt: Kurzanleitung

Periodische Struktur eingeben, mit Gmsh vernetzen, mit hp-FEM rechnen, Schaubilder und Karten ansehen. Keine BEM- oder RCWA-Daten nötig.

## Installation
Siehe README im Hauptordner. Die sechs Dateien (`fem_app.py`, `fem_geometry.py`, `fem_materials.py`, `fem_run.py`, `fem_post.py`, `fem_worker.py`) liegen zusammen im Ordner `fem_gui/`.

Für die App (Windows-Python oder MSYS2-Python):
```
pip install -r requirements.txt
```
(Im MSYS2-Python alternativ `pacman -S mingw-w64-ucrt-x86_64-gmsh`; prüfen mit `python -c "import gmsh"`.)

Die Rechnung selbst läuft in dem Python, in dem `hpfem` gebaut ist (MSYS2). Die App startet dieses Python als eigenen Prozess; die App braucht
`hpfem` also nicht. Start:
```
streamlit run fem_gui/fem_app.py
```
Seitenleiste: **Python mit hpfem** (z. B. `C:/msys64/ucrt64/bin/python3.exe`), **hp-FEM-Ordner** (enthält `python/`), Arbeitsordner, Threads.
Voraussetzung ist die Version von hp-FEM mit dem konischen Löser (`ConicalScattering`); für die hp-Adaptivität wird der In-Ebenen-Löser (`Scattering2D`) mit `AdaptiveMesh2D` benutzt.

## Ablauf (fünf Reiter)
1. **Modell:** Materialien (Bibliothek mit Si, Ag, Au, Al, GaAs, MAPbI3, SiO2, TiO2, Wasser, Luft; n + ik; ε; Drude; eigene Tabelle), Schichtaufbau
   (Einfallsmedium, Schichten, Substrat), Formen (Rechteck, Trapez, Kreis, Ellipse, Polygon; spätere überdecken frühere; Formen über dem Zellrand setzen sich
   periodisch fort), Beleuchtung (TE/TM, θ, φ, λ, Durchlauf über λ, θ oder φ) und Rechengebiet. Rechts steht die Vorschau, darunter die Prüfmeldungen.
   Mit **Vorlage laden** (Seitenleiste) starten, **Vorschlag aus der Wellenlänge übernehmen** setzt Höhen, PML und Substrattiefe.
2. **Netz:** Gmsh-Vernetzung mit Elementgröße nach Wellenlänge und Eindringtiefe, Verfeinerung an Grenzflächen, periodischem Rand, gekrümmten Elementen für
   Kreise und Ellipsen. Zeigt Statistik, Winkelqualität, Freiheitsgrade (Schätzung) und das Netz.
3. **Rechnung:** Polynomordnung p, PML-Zielfehler (angepasst an den größten Beugungswinkel), Löser, Ordnungen, Feldkarten (welche Punkte, Auflösung),
   optional Konvergenzstudie in p. Die Rechnung läuft im Hintergrund, die Seite zeigt Fortschritt und Protokoll; „Abbrechen“ ist möglich.
4. **Ergebnisse:** Spektrum und Beugungsordnungen (R, T, A, je Ordnung, ebener Stapel als Referenz), Feldkarten (|E|, |E|², Komponenten Re/Im/Betrag, absorbierte
   Leistungsdichte; 1 bis 3 Perioden; Geometrie und Netz einzeichenbar), Schnitte, Absorption je Material, Konvergenz in p, Tabelle und CSV-Export, Archiv.
5. **Info:** Koordinaten, Konventionen, Grenzen.

## Löser wählen und hp-Adaptivität testen (Reiter 3)
Drei Möglichkeiten:
* **Konischer Löser** (TE, TM, Azimut): feste Ordnung p auf dem vorhandenen Netz. Dieser Löser hat noch keine hp-Adaptivität.
* **In-Ebenen-Löser, gleichmäßig** (nur TM, φ = 0): derselbe Fall mit dem älteren Löser. Auf demselben Netz gerechnet vergleicht das beide Löser: R, T und die
  Ordnungen müssen übereinstimmen.
* **In-Ebenen-Löser, hp-adaptiv** (nur TM, φ = 0): Schleife aus Rechnen, Fehlerschätzer je Dreieck, Dörfler-Markierung und h- oder p-Verfeinerung nach der Glattheit.
  Einstellbar: Startordnung p₀, höchste Schrittzahl, höchste Freiheitsgrade, Dörfler-Anteil, Abbruch bei Änderung von R und T unter einer Schwelle (in zwei aufeinanderfolgenden
  Schritten), und ob bei einem Durchlauf nur am ersten Punkt adaptiert und das Endnetz für alle weiteren Punkte benutzt wird. Dreiecke am periodischen Rand und in der PML werden nie
  markiert. Ergebnisse im Reiter 4 „hp-Adaptivität“: Tabelle der Schritte, Konvergenzdiagramme (R, T und η über den Freiheitsgraden) und das Endnetz, nach der Polynomordnung gefärbt.

**Vorschlag für den Test:** Vorlage „Ag-Lamellengitter (TM, 50°: Test der hp-Adaptivität)“. Reiter 2: 2 Elemente pro Wellenlänge, 1 Element pro Eindringtiefe (etwa 450 bis 900
Dreiecke), gerade Elemente. Dann dreimal rechnen und im Reiter 4 vergleichen: (1) In-Ebenen-Löser hp-adaptiv mit p₀ = 3, 12 bis 20 Schritten, höchstens 150 000 Freiheitsgraden;
(2) In-Ebenen-Löser gleichmäßig mit p = 4 und 5 auf einem feineren Netz; (3) konischer Löser. RCWA-Referenz für die Silberdaten dieser Bibliothek (ε = −4,698 + 0,217 i bei 405 nm,
1/N-extrapoliert, Unsicherheit etwa 1e-4): **R₀ = 0,7800, R₋₁ = 0,0801, absorbiert 0,140.** Gleichmäßige Netze blieben in den früheren Tests bei Abweichungen von etwa 4e-3 bis 9e-3 stehen,
die hp-adaptive Rechnung erreichte etwa 2e-5 (R₋₁) bei etwa 66 000 Freiheitsgraden.
Gekrümmte Netze (Kreise, Ellipsen) mit hp-Verfeinerung sind ungetestet; die App warnt.


## Formen verschieben und zentrieren (Reiter 1)
Unter der Periode: **„Alle Formen in x verschieben um (nm)“** mit „Verschieben“, und **„Zentrieren“**, das die Mitte der x-Ausdehnung aller Formen in die Zellmitte P/2 legt. Die Schichten
hängen nicht von x ab; Formen, die über den Zellrand ragen, setzen sich periodisch fort. Mit **„Beim Ändern der Periode die Formen mitzentrieren“** wandern die Formen um die halbe Änderung
der Periode mit, die Struktur bleibt dann mittig.

## Neu mit hp-FEM M15 (Stand 7.10.2026): schnellere Karten, exakte Elemente, exakte Absorption
Der Worker nutzt, wenn der Build es kann, `problem.sample` (alle Kartenpunkte in einem Aufruf, parallel), `problem.triangulate` und `absorbed_power_by_tag`; bei einem älteren Build
rechnet er wie vorher (Punktschleife, Rasterintegration) und das Protokoll zeigt „field map … by loop“. Dafür hpfem aktualisieren (`git pull`, Python-Modul neu bauen und kopieren).
* **Feldkarten, „Elemente (exakt)“:** das Feld auf den unterteilten Dreiecken des Netzes. Sprünge der Normalkomponente an Materialgrenzen bleiben scharf, das Material ist je Dreieck bekannt,
  die absorbierte Leistungsdichte Q hat deshalb keine Pixelfehler. Umschalten auf „Raster“ ist möglich (nur dort lässt sich das Netz einzeichnen; Schnitte nutzen das Raster).
  Einstellungen in Reiter 3 unter Ausgabe: „Elementdarstellung speichern“ und die Unterteilung je Dreieck (3 bis 4 genügt für p = 4 bis 5).
* **Absorption je Material, exakt:** Volumenquadratur der Joule-Wärme Q = ½ ω ε₀ Im ε |E|² auf dem Netz (nur das physikalische Gebiet, PML-Zellen ausgenommen), bezogen auf die einfallende
  Leistung je Periode. Sie steht für **jeden Punkt eines Durchlaufs** zur Verfügung, ohne dass Karten gespeichert werden: Diagramm „Absorption je Material über den Durchlauf“, Tabelle im Reiter 4
  „Absorption“ und Spalten `A[Material]` in der Ergebnistabelle. Die Summe wird mit der Energiebilanz verglichen (Verhältnis nahe 1); die Rasterwerte bleiben als Vergleich, wenn Karten gespeichert sind.

## Wichtig für sinnvolle Ergebnisse
* **Netz zur Ordnung:** bei Polynomordnung p genügen etwa 12/p Elemente pro Wellenlänge im Material (p = 4: 3). Mehr Elemente erhöhen die Freiheitsgrade stark.
  Die App zeigt eine Schätzung und warnt über 400 000.
* **PML auslegen:** Die Bibliothek verlangt, dass das gedehnte Feld in der PML aufgelöst ist: k₀·n·|s|·h ≤ Grenze (3 für p ≥ 4, sonst 0,75·p), mit |s| = √(1 + σ̂²) und
  σ̂ = −(m+1)·ln R₀ / (2·k₀·n_ref·d). Ein strenger PML-Zielfehler bei großem Beugungswinkel (die PML wird für den größten Winkel der propagierenden Ordnungen ausgelegt)
  macht R₀ winzig und σ̂ groß; eine dünne PML braucht dann extrem feine Zellen, sonst meldet der Solver „the PML is under-resolved“. Die App zeigt die Auslegung im Reiter 3
  („PML-Auslegung“: Dicke, σ̂, |s|, nötige und vorhandene Zellgröße) und warnt vor der Rechnung. **„Vorschlag aus der Wellenlänge“ (Reiter 1)** legt die PML-Dicke aus
  Polynomordnung, PML-Zielfehler und „Elemente pro Wellenlänge in der PML“ (Reiter 2, Voreinstellung 6) aus; das PML-Netz wird entsprechend fein. Nach einer Änderung von p oder des
  Zielfehlers den Vorschlag erneut anwenden. Voreinstellung des PML-Zielfehlers: 1e-3; für genaue Vergleiche (z. B. hp-Test) 1e-4 oder kleiner, dann wird die PML dicker.
* **Periodische Ränder bei hp-Verfeinerung:** Seit hp-FEM M15 F16 Stufe 1 (Stand 7.10.2026) deklariert der Worker die Randpaare mit `AdaptiveMesh.set_periodic`, die Bibliothek spiegelt dann jede Verfeinerung auf die Gegenseite. Bei einem älteren Build gilt die Reparatur unten (das Protokoll zeigt „NOTE: this hpfem has no AdaptiveMesh.set_periodic“). Hintergrund: Der Löser verlangt, dass linke und rechte Seite Facette für Facette gleich fein sind. Die Verfeinerung eines Nachbarn kann eine Randzelle
  nur auf einer Seite mitverfeinern; die App gleicht die Seiten nach jedem Schritt an (die gröbere Seite wird nachverfeinert). Gelingt das nicht, endet die Schleife mit dem Ergebnis des
  letzten vollständigen Schritts und einem Hinweis im Reiter „hp-Adaptivität“.
* **Unterer Rand:** ein verlustfreies (oder fast verlustfreies) Substrat braucht unten eine PML. Eine Metallwand ist nur sicher, wenn das Feld bis dahin
  auf e⁻⁶ abgeklungen ist (sechs Eindringtiefen; bei Si und 405 nm sind das 1480 nm). Der Vorschlag aus der Wellenlänge beachtet das.
* **Runde Formen** nicht tangential an eine Grenzfläche legen (Kreis auf Substrat: 1 bis 2 nm einsinken lassen), sonst entstehen verzerrte Dreiecke. Die App warnt.
* **Bibliotheksdaten** gelten nur in ihrem Wellenlängenbereich (z. B. TiO2 430 bis 1530 nm); außerhalb meldet die App einen Fehler vor der Rechnung.
* **Absorption je Material** wird aus dem Feld auf dem Kartenraster integriert (Fehler der Größe eines Pixels an scharfen Materialgrenzen). Das Verhältnis zur
  Energiebilanz zeigt, ob die Kartenauflösung reicht; die normierte Spalte verteilt die verlässliche Energiebilanz nach dem Feld.
* **T und A:** bei verlustbehaftetem Substrat ist nur R messbar; A inkl. Substrat = 1 − R. Bei verlustfreiem Substrat sind T (Ordnungen im Substrat) und A = 1 − R − T verfügbar.

## Erster Test des Solvers
1. Vorlage **„Dünnschicht SiO2 auf Si (ebener Stapel …)“**: ohne Struktur. R und T müssen mit den Linien „ebener Stapel“ im Spektrum übereinstimmen.
2. Vorlage **„Si-Lamellengitter (Projektfall …)“**, TM, 50°, 405 nm: R0 und R−1 nahe den Referenzwerten des Projekts (R0 ≈ 0,143, R−1 ≈ 0,142). Die Bibliotheksdaten für Si
   weichen leicht von den BEM-Daten ab (ε = 29,87 + 2,86 i statt 29,63 + 2,77 i), kleine Unterschiede sind deshalb zu erwarten.
3. Konvergenzstudie in p anschalten: R, T, A sollten sich mit p schnell einem Wert nähern.

## Bei Problemen
* „Gmsh fehlt“ in der Seitenleiste: `pip install gmsh` in dem Python, das die App startet.
* Rechnung bricht ab: Protokoll im Reiter 3 öffnen (letzte Zeilen) und die Zeile mit `ERROR` lesen; der Ordner `fem_work/run/` enthält `log.txt`, `job.json` und `mesh.msh`.
* Rechnung läuft nicht an: „Python mit hpfem“ und „hp-FEM-Ordner“ in der Seitenleiste prüfen (`python -c "import hpfem"` im gewählten Python).
