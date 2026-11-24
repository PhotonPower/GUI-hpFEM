# Änderungen

## Unveröffentlicht

* Neu: **Nahfeld des Kreuzgitters** (Reiter 3 „Nahfeld“): |E|²/|E₀|² (Gesamtfeld, |E₀| = 1) über `CrossedResult.field` auf einer xy-, xz- und yz-Ebene. Schnittlagen und Punktzahl stellt man im Reiter 2 ein (Standard: Zellmitte, Mitte der ersten Schicht). Schichtgrenzen und Formen sind eingezeichnet, je Wellenlänge entsteht `field_<i>.npz`, nur bei der Hauptrechnung. An einem echten Lauf getestet (p = 2, 9212 Unbekannte), noch nicht gegen einen Referenzwert geprüft.
* Neu (hp-FEM M21 S4, Issue #16): **Kreuzgitter rechnen** mit `hpfem.crossed.solve` (`fem_xgrating_worker.py`, Reiter „2 Rechnung“ und „3 Ergebnisse“ des Modus „Kreuzgitter“).
  Ein Aufruf je Wellenlänge (s und p aus einer Faktorisierung), Fortschritt und Abbruch über Log und `cancel`-Datei wie bei den anderen Modi. Ergebnisse: R und T je Ordnung (m, n),
  ΣR, ΣT, Absorption und Energiebilanz ΣR + ΣT + A − 1, Spektrum, **Konvergenzstudie in p** (feste Elemente je Wellenlänge, Tabelle und Bild), **s/p-Gegenüberstellung** (Symmetrieprobe,
  größte Abweichung) und **Vergleich mit `efficiencies.csv` von FourierGrating** (Ordnungen (n, m) → (m, n) = (n, −m), δ = 0° s und 90° p, ψ = −φ; Abweichung je Ordnung, über p in der
  Konvergenztabelle). **Import einer FourierGrating-Konfiguration** über `hpfem.fouriergrating` im Python mit hpfem (`fem_xgrating_worker.py --import`): Rahmen, Schichtreihenfolge, p = −p, nur
  δ = 0° oder 90°. Neu im Modell: Polygon, Drehung von Ellipse und Rechteck, Netz (p, Elemente je Wellenlänge), Konvergenzstudie; Prüfung: verlustfreie Halbräume, keine Kristalle.
  Neues Merkmal `crossed` in der Bibliotheksprüfung (`fem_run.PROBE`: `hpfem.crossed`, `OrderPlane`, `diffraction_orders`); ohne es ist „Rechnen“ gesperrt und die Seite sagt warum.
  Tests mit einem Ersatz für hpfem (`tests/test_xgrating.py`: Abbildung des Modells auf `CrossedGrating`, Einheiten, Fehler je Punkt, Abbruch, Tabellen, CSV-Vergleich).
  **Gerechnet** (hpfem 0.5.1, neu gebautes `_hpfem`, GUI-Worker): Lochgitter r = 125 nm, Periode 500 nm, d = 150 nm, λ = 405 nm, Luft/Glas n = 1,5, p-Polarisation, 2,5 Elemente je Wellenlänge,
  gegen FourierGrating (RCWA, Nx = Nz = 20, `reference_rcwa_N20.csv`): größte Abweichung über alle Ordnungen und |ΔT₀|: p = 2: 1,6e-2, p = 3: 9,2e-4, p = 4: 2,9e-4 (T₀ = 0,8044 gegen 0,8041; R₀ = 0,02466
  gegen 0,02468); beugende Ordnungen (±1, 0), (0, ±1), (±1, ±1) bei p = 4 bis 5,2e-5. Energiebilanz ΣR + ΣT − 1: 7,9e-3, −8,6e-4, 7,0e-5 (p = 2, 3, 4). s gegen p bei senkrechtem Einfall: Summen auf 4e-5 (ΣR) und 7e-5 (ΣT) gleich,
  s bei (m, n) gleicht p bei (n, m) bis 2,9e-4 (p = 3). Schräger Einfall θ = 30°, φ = 20°, s, p = 3: Bilanz +9,3e-4; Ordnungen unsymmetrisch wie erwartet, 4 reflektierte und 11 transmittierte Ordnungen wie
  in der Vorschau aus der Gittergleichung. Kosten (bei parallel laufendem Benchmark): 19 s (p = 2, 14 052 Unbekannte), 77 s (p = 3, 39 375), 192 s (p = 4, 84 504). Nicht gerechnet: Vergleich bei schrägem Einfall mit FourierGrating
  (keine Referenzwerte), mehrere Wellenlängen, absorbierende Schichten.
* Neu (Vorbereitung hp-FEM M21, Issue #16): Modus **„Kreuzgitter (doppelt periodisch, Eingabemaske)“** (`fem_xgrating.py`, `fem_xgrating_app.py`). Perioden p_x, p_y; Einfallsmedium, Substrat;
  Schichten von oben nach unten mit Dicke, Hintergrundmaterial und Formen (Kreis, Ellipse, Rechteck; Loch oder Säule mit Material, Mittelpunkt, Maßen); Einfall (λ oder Durchlauf, θ, φ,
  Polarisation s/p); Aufsicht der Zelle und Seitenansicht des Stapels; Tabelle der ausbreitungsfähigen Ordnungen (m, n) in Reflexion und Transmission aus der Gittergleichung; Modell als JSON
  speichern und laden; Vorlage „Lochgitter in Glas auf Glas“ (FourierGrating-Beispiel). Konvention wie hpfem: Stapelnormale z, periodisch in x und y. Die Rechnung kam danach (nächster Eintrag). Tests: `tests/test_xgrating.py`.
* Behoben: `load_model()` (periodische Struktur, Rotationskörper) stürzte bei fremden Materialangaben ohne `type` mit `KeyError 'type'` in `fem_materials.describe()` ab. Neu
  `fem_materials.normalize_spec`: `{"n": 1.5}` bzw. `{"n": 1.5, "k": 0.01}` → `{"type": "index", "n": .., "k": ..}` (ebenso eine bloße Zahl, `{"eps": ..}`, `{"name": "Si"}`); nicht Umsetzbares
  (negatives k, kein n, unbekannter Typ) und nicht definierte Materialien in Einfallsmedium, Substrat, Schichten oder Formen werden mit verständlicher Meldung abgelehnt. FourierGrating-Dateien
  (`period_um`, `dimension`, `contour2d`) erkennt `load_model()` und verweist auf das Kreuzgitter; ihre Umsetzung liefert der Import in hpfem (M21 S3).
* Neu: eigener Seitenleisten-Eintrag **„3D-Wellenleiter und Ports (Streifen, Koppler, Ring)“**; er öffnet den 3D-Modus direkt mit der Aufgabe „Wellenleiter und Ports“ (vorher nur über Modus „3D-Strukturen“ → Aufgabe erreichbar).
* Neu: **Ringresonator in 3D** (Modus „3D-Strukturen“, Aufgabe „Wellenleiter und Ports“, Bauteil „Ringresonator (Add-Drop, Gmsh)“; hpfem 0.5, Vorlage nach
  `examples/ring_resonator_3d` der Bibliothek): Ring zwischen zwei geraden Zuleitungen gleichen Querschnitts (Parameter Ringradius, Breite, Höhe, Spalt, Ränder,
  Materialien Kern/Mantel), Gmsh-Netz mit gekrümmten Tetraedern (`fem_ring3d_mesh.py`, Port von `ring_mesh.py`; Zellgröße am Kern und fern, Staffelweg, Ordnung p am Kern
  innerhalb einer Reichweite und fern), vier modale Ports (1 Eingang, 2 Through, 3 Drop, 4 Add), Anregung an Port 1, optional das **halbe Modell** (magnetische Spiegelebene
  z = h/2, S-Parameter des ganzen Bauteils). Ergebnis: Through, Drop, Add und Reflexion über der Wellenlänge (feste Wellenlängen oder **automatische Suche**: neun Punkte um die
  Resonanz des isolierten Rings, dann neun innerhalb ±2 Halbwertsbreiten des angepassten Pols), die Resonanz als gemeinsamer komplexer Pol von Through und Drop (`fit_pole`:
  Lage, Halbwertsbreite, belastete Güte, Rest der Anpassung), die **Referenz** des isolierten Rings als Rotationskörper (`AxisymmetricCavity`, Azimutordnungen m0−1, m0, m0+1 aus
  n_eff der Zuleitung, nächste Resonanz zum Ziel) und die **FSR-Schätzung** λ²/(n_g·2πR) aus dem Gruppenindex der geraden Zuleitung (2D-Modenlöser, Mittendifferenz über ±5 nm,
  Materialdispersion enthalten) mit dem FSR des isolierten Rings daneben; optional |E|² in der x–y-Ebene durch die Kernmitte bei der Wellenlänge mit dem größten Drop; Speicher-
  und Zeitschätzung vor dem Lauf, harte Speichergrenze, wählbarer Löser (auto, SparseLU, MUMPS, cuDSS; nicht verfügbare Löser brechen vor der ersten Rechnung ab). Drei Vorlagen:
  „Test“ (R = 1,5 µm, 150-nm-Zellen, p = 1, halbes Modell: 16 808 Unbekannte, 386 MB, rund 9 s je Wellenlänge bei vier Threads), „mittel“ (R = 2 µm, 150 nm, p = 2/1: 79 740 Unbekannte,
  7,8 GB nach der Schätzung der Bibliothek, **nicht gelaufen**) und „Bibliotheksbeispiel“ (R = 2 µm, 100 nm, p = 3/2, 0,4 bis 1,7 Millionen Unbekannte, MUMPS und viel Speicher: **nicht gelaufen**).
  Gemessen: Referenz R = 2 µm, p = 2 auf 100-nm-Zellen: m = 19 bei 1556,7 nm (Bibliothek, p = 4 auf 50 nm: 1557,39 nm), m = 18 bei 1604,6 nm (1605,23), m = 20 bei 1511,5 nm (1512,21);
  Zuleitung 450 × 220 nm: n_eff = 2,354, n_g = 4,15 bei 1550 nm; FSR aus n_g·L für R = 2 µm rechnerisch 46,5 nm (die Bibliothek nennt 45 bis 48 nm für den isolierten Ring; hier nicht nachgerechnet). Der Testlauf (p = 1, R = 1,5 µm) zeigt nur die Rechenkette:
  Energiebilanz Σ|S|² = 1 auf 1e-6, aber Through 0,88, Drop 0,09, Reflexion 7·10⁻⁴ bei 1555 nm ohne Bedeutung als Spektrum (Scheinresonanzen, wie die Bibliothek für p = 1 meldet).
  Bei R = 1,5 µm weicht die FSR-Schätzung der geraden Zuleitung stark vom FSR des isolierten Rings ab (62 gegen 85 nm; vermutlich rückt die Mode im Bogen nach außen, nicht geprüft).
  **Nicht validiert:** das 3D-Spektrum, die 3D-Resonanz gegen die Bibliothek (die Produktionsläufe in `benchmarks/results/` sind hier nicht reproduziert), die Güte. Die Wellenlängen einer
  Rechnung teilen keine Faktorisierung (die Matrix hängt von ω ab); jede Wellenlänge kostet eine volle Lösung. Neue Module `fem_ring3d_mesh.py`; erweitert `fem_ports3d.py`, `fem_ports3d_app.py`,
  `fem_ports3d_worker.py`; Tests in `tests/test_ports3d.py` (10 neue: Vorlagen und Prüfung, Pol-Anpassung gegen bekannten Pol, Netzgruppen, Lauf mit Energiebilanz, automatische Suche mit
  synthetischem Pol, Speichergrenze und fehlender Löser, AppTest).

* Neu: **Wärme und Mechanik in 3D** (Modus „Wärme und Mechanik“, Schalter „Raumdimension“ 2D/3D; hpfem 0.5, `Scattering3D`, `Thermal3D`, `ThermoOptical3D`, `Elasticity3D`).
  Fall **Kugel**: ebene Welle (E längs x, Ausbreitung +z) auf eine Kugel in verlustfreier Matrix (`box_with_ball`, PML) → absorbierte Leistung → Temperatur (Wärmesenke T = T_ref an den
  Außenflächen der Box) → optional selbstkonsistent mit ε(T) (`ThermoOptical3D`, Relaxation, Toleranz, Konvergenzverlauf) → optional Wärmespannung (`Elasticity3D`, Wärmeausdehnung je
  Material, Schrumpfung der Kugel, Außenflächen eingespannt oder frei); wählbar das **Viertelmodell** (x > 0, y > 0, Leistung und Energie ×4, Felder gespiegelt). Fall **Quader**: Randbedingungen je
  Fläche (frei, eingespannt, Symmetrie, Zug/Druck), Temperaturänderung und Schrumpfung linear in z (damit auch das Schrumpfblock-Beispiel R7 der Bibliothek, Viertelmodell mit Substrat) oder eine
  gleichmäßige **Wärmequelle** (mW/µm³) mit Wärmesenken auf gewählten Flächen (`Thermal3D`, Rest adiabatisch), deren Temperaturfeld die Wärmedehnung treibt. Ergebnis: Kennzahlen, Schnitte
  senkrecht zu x, y oder z für T, |E|/E₀, u, σ-Komponenten und Von Mises, Linienschnitte, verformtes Raster eines Schnitts mit Überhöhung, Konvergenz. **Vor jedem Lauf** zeigt die App Zellen,
  Unbekannte, Speicher (`hpfem.estimate_memory`) und Zeit; der Worker bricht ab, wenn die Schätzung die einstellbare Speichergrenze (Vorgabe 8 GB) überschreitet. Eingebaute Prüfungen
  (gemessen): Quader mit Symmetrieebenen unter Zug, gleichmäßiger Temperatur und Schrumpfung gegen die geschlossene Lösung (exakt in 3D, relative Abweichung < 1e-12); Wärmequelle zwischen
  zwei Senken gegen T_ref + qW²/(8κ) (< 1e-12); Kugel gegen ΔT ≈ P/(4πκ)(1/R − 1/R_s) mit der Kapazität des Würfels als R_s (Vorlage n = 1, p = 2, 1500 Zellen: 30,8 K gegen 32,0 K, −3,7 %).
  Gemessene Kosten (hpfemenv, SparseLU, ein Thread): Kugel-Vorlage 14 s Optik und Wärme, 6 s Mechanik, 10 790 optische Unbekannte; selbstkonsistent mit 5 Iterationen 65 s; Elastizität am
  Quader 2 187 Unbekannte 2 s, 6 591 10 s, 14 739 48 s (590 MB), p = 3 mit 12 288 Unbekannten 60 s (800 MB) – daraus die Faustformel der App (für die Kugel zu hoch, etwa Faktor 3).
  Grenzen: die Temperatur gilt für das gewählte Gebiet (Senke an der Box); Standardnetz grob (die Bibliothek warnt vor unteraufgelöster PML); stationär, keine temperaturabhängigen Materialdaten
  außer ε(T), Optik auf dem unverformten Netz; Maxima sind Rasterwerte; Kette nicht gegen Literaturwerte für konkrete Partikel validiert; `absorbed_power_by_tag` und Lastbilanz weichen bei grobem
  Netz um etwa 2e-3 voneinander ab; die Kugelschrumpfung wird an Quadraturpunkten ausgewertet. Neue Module `fem_thermo3d.py`, `fem_thermo3d_app.py`, `fem_thermo3d_worker.py`, Tests
  `tests/test_thermo3d.py`; Eingriff in `fem_thermo_app.py` (Schalter 2D/3D), `fem_app.py` (Dateiliste, Modusname), `fem_run.py` (`feats["thermo3d"]`).
* Neu: Modus **„Wärme und Mechanik (2D)“** (Seitenleiste; hpfem 0.5, `Thermal2D`, `ThermoOptical2D`, `Elasticity2D`). Fall **Nanodraht**: ebene Welle auf einen Draht
  in verlustfreier Matrix (`square_with_disc`, PML) → absorbierte Leistung → Temperatur (Wärmesenke T = T_ref am äußeren Rand; Leitfähigkeit je Material) →
  optional selbstkonsistent mit ε(T) = ε + (dε/dT)(T − T_ref) (`ThermoOptical2D`, Relaxation, Toleranz, Konvergenzverlauf) → optional Wärmespannung in ebener
  Dehnung (Wärmeausdehnung α je Material aus dem Temperaturfeld, Schrumpfung des Drahts, Rand eingespannt oder frei). Fälle **Block** (ebene Dehnung) und
  **Zylinder** (rotationssymmetrisch, `axis_tag`): Randbedingungen je Seite (frei, eingespannt, Symmetrie, Zug/Druck), lineare Temperaturänderung, Schrumpfung.
  Ergebnis: Kennzahlen (absorbierte Leistung, Maximaltemperatur, größte Verschiebung und Von-Mises-Spannung, Dehnungsenergie), Feldkarten (T, |E|/E₀, u, σ-Komponenten,
  Von Mises), Schnitte, verformtes Netz mit Überhöhung, Konvergenz. Eingebaute Prüfungen: Verschiebungen von Block und Zylinder gegen die geschlossene Lösung (gemessen
  bis 1e-14 relativ), Temperatur des Drahts gegen die Linienquelle im quadratischen Senkengebiet (ΔT = P/(2πκ)·ln(1,0787 a/r); bei der Vorlage stimmen beide auf besser als 1 % überein).
  Die Bibliothek wendet Temperatur und Schrumpfung als Eigendehnung in der Ebene an (ε*_zz = 0); das steht in der Info. Materialvorgaben sind grobe Richtwerte;
  dε/dT der Vorlage ist ein Beispielwert. Nicht validiert gegen Literaturwerte für konkrete Nanodrähte, stationär, Materialdaten unabhängig von T, Optik auf dem
  unverformten Netz. Neue Module `fem_thermo.py`, `fem_thermo_app.py`, `fem_thermo_worker.py`; Tests `tests/test_thermo.py`.
* Neu: Modus **„3D-Strukturen“** (Seitenleiste; hpfem 0.5, `Scattering3D`, `Resonance3D`, `BandStructure3D`): (1) **Streuung** einer ebenen, x-polarisierten Welle
  längs +z an einer Kugel (`box_with_ball`, gekrümmte Elemente) oder einem Würfel (`box`, an das Gitter angepasst) in einer PML-Box: Streu-, Absorptions- und
  Extinktionsquerschnitt über `cross_sections`/`Surface3D.around_cells`, Effizienzen Q mit der **Mie-Reihe** (`mie_sphere`) als Referenz für die Kugel, Wellenlängendurchlauf
  (Material je Wellenlänge ausgewertet), |E|² auf zwei Schnittebenen, optional das Viertelmodell (Spiegelsymmetrie, Faktor 4, wie Validierung C der Bibliothek).
  (2) **Resonanzen** (Quasi-Normalmoden mit PML, λ, Q, Schnitte von |E|²; Material bei der Zielwellenlänge eingefroren). (3) **Bandstruktur** eines einfach kubischen
  Kristalls (Kugel oder Würfel, reelle ε, Pfad Γ–X–M–R–Γ; die statischen Nullmoden bei Γ werden bis auf die zwei akustischen entfernt; Referenz: leeres Gitter).
  Netzdarstellung als Schnitt, **Speicherschätzung** (`hpfem.estimate_memory`) vor jedem Lauf mit Warnung bei über 60 % des freien Hauptspeichers, einstellbare
  Speichergrenze (der Worker bricht bei Überschreitung ab) und Zeitrichtwert. Neue Module `fem_threed.py`, `fem_threed_app.py`, `fem_threed_worker.py`, Tests
  `tests/test_threed.py`. Gemessen (Wheel 0.5.0, nur SparseLU): Kugel n = 2, ka = 2, n = 2, p = 2, Viertelmodell: 18 264 Unbekannte, 72 s, Q_sca 4,90 gegen 4,77 (Mie, +2,7 %);
  n = 1, p = 2: 2 526 Unbekannte, 2 s, +5 %; n = 2, p = 3 braucht 5,5 GB und über 2 min. Resonanzen ε = 12, n = 1, p = 2: 9 324 Unbekannte, 72 s (λ ≈ 715 nm, Q ≈ 12,6, nicht
  validiert). Bandstruktur: das leere Gitter trifft bei X die exakten Werte (0,5 ×4, 1,118), aber ein k-Punkt kostet schon bei 2 936 Unbekannten 17 bis 75 s; die Bibliothek
  meldet bei den Gittern ihrer Generatoren, dass gegenüberliegende Facetten weder gleich noch geschachtelt seien (Hinweis im Protokoll, die Rechnung ist trotzdem richtig).
  Nicht dabei: 3D-Wärme und -Mechanik.

* Neu: **3D-Strukturen erweitert** (hp-FEM 0.5): (1) **Substrat und Schichtstapel** (Streuung, Gmsh-Geometrie): `LayerStack3D` als geschichteter Hintergrund der Streufeld-Formulierung (`ScatteringSetup3D.background`),
  ebene Welle unter beliebigem Einfallswinkel, s oder p, Azimut (`LayeredPlaneWave3D.field`), R, T, A des Stapels ohne Objekt, Objekt (Kugel, Zylinder, Quader) mit Abstand über der Oberkante z = 0.
  (2) **Gmsh-Netze**: eigene Geometrie (neues Modul `fem_threed_gmsh.py`: Box, jede Schicht ein Volumen, Objekt, Elemente zweiter Ordnung, Maschenfelder am Objekt und je Schicht; physikalische Gruppen für Volumina und
  die sechs Boxseiten) oder hochgeladene `.msh`-Datei mit Tabelle Zellmarke → Material, Wahl der Streukörper-Marken, Flächenmarken ohne Eintrag werden zu den Boxseiten (Worker-Aufruf `inspect` meldet die Marken).
  (3) **hp-adaptive Verfeinerung der Streuung**: `AdaptiveMesh3D` + `Scattering3D.estimate` + `dorfler_marking` + `hp_decide_by_prediction` + `hp_refine` + `predict_indicators`, Abbruch bei Schrittzahl, Obergrenze
  Unbekannte, Speichergrenze oder η/η₀; Konvergenzdiagramm (η, Q_sca gegen Unbekannte), Schritttabelle, Netz gefärbt nach Ordnung. (4) Neue Aufgabe **Wellenleiter und Ports (S-Parameter)**
  (`fem_ports3d.py`, `fem_ports3d_app.py`, `fem_ports3d_worker.py`): Streifenwellenleiter und Richtkoppler zwischen modalen Ports (`s_parameters`, `WaveguidePort`; Stirnflächen z = 0 und z = L, PEC-Seitenwände),
  Netze als Tensorgitter mit Gitterlinien auf den Kernkanten (`hpfem.box` auf Indexkoordinaten, dann Knoten verschoben; `hpfem.box` selbst ist gleichmäßig), S-Matrix, R, T, Verlust, n_eff der Portmoden, |E|² im Längs- und Querschnitt,
  Referenzen aus dem 2D-Modenlöser (β der Grundmode bzw. Supermoden; Theorie gekoppelter Moden) auf demselben und auf einem feineren Netz (halbe Zelle, p + 2); oder ein Gmsh-Netz mit Flächenmarken als PEC-Wand oder Port.
  Neue Tests `tests/test_threed_ext.py` (9) und `tests/test_ports3d.py` (11).
  **Gemessen (Wheel 0.5.0, SparseLU):** Gmsh-Kugel n = 2, ka = 1, p = 2, hw = 200 nm, PML 300 nm = 2 ferne Zellen (150 nm), 18 000 Unbekannte, 82 s: Q_sca = 0,7948 gegen 0,7969 (Mie, −0,3 %), Q_abs = −0,002;
  mit nur einer Zelle im PML (250 nm) 0,60 (−24 %), mit 1,6 Zellen 0,17: das PML braucht mindestens zwei bis drei Zellen. Derselbe Fall mit einem „Stapel“ aus Luft (Einfallsmedium = Substrat = Luft): Q_sca = 0,811 (+1,7 % gegen Mie,
  20 856 Unbekannte, 127 s); die Datei desselben Netzes liefert dieselben σ (1e-6). Stapel Luft/Glas bei 30°, p: R = 0,02525 stimmt mit der Fresnel-Formel überein. Silizium-Zylinder auf Glas (grobes Netz, 3 800 Unbekannte, 2,6 s):
  Q_sca = 6,05 bei p = 1 gegen 8,67 bei p = 2 (19 500 Unbekannte, 127 s): p = 1 ist hier nicht konvergiert. Gold-Kugel auf Schicht (Vorlage, p = 1, 14 000 Unbekannte, 18 s): nur ein Mechanik-Test, die Zahlen sind nicht konvergiert (feineres Netz: Q_sca 0,35 statt 1,10).
  hp-adaptiv (Kugel n = 2, ka = 2, Viertelmodell, n = 1, p = 2, 4 bis 5 Schritte, 16 s): η fällt von 0,133 auf 0,042 (2 526 → 10 638 Unbekannte), Q_sca 4,51 → 4,44 (Mie 4,77), nicht monoton; Q_abs ≈ −0,49 (PML unteraufgelöst, die Bibliothek warnt):
  der residuale Schätzer sieht den PML-Fehler nur teilweise. Streifen Si/SiO₂ 450 × 220 nm, λ = 1550 nm: n_eff der Portmode 2,136 bei p = 1 und 150-nm-Zellen, 2,343 bei p = 2, 2,356 bei p = 3 (2D, 150 nm): bei p = 1 9 % zu klein; 3D mit p = 1: T = 0,995, R = 0,005
  (8 s, 5 000 Unbekannte), mit p = 2 und 150 × 400 nm Zellen 10 000 Unbekannte, 198 s. Richtkoppler (Lücke 200 nm, L = 1 µm, p = 1, 150-nm-Zellen, Supermoden-Ports): cross = 0,038, Kopplungslänge 10,3 µm auf dem Netz der Rechnung gegen 24,6 µm auf dem feinen
  Referenznetz (cross_Ref = 0,004): das grobe Netz koppelt viel zu stark, cross ist bei kurzen Kopplern empfindlich auf den Phasenfehler (0,12 bis 0,20 rad). Ein Gmsh-Streifen (ungeordnetes Netz, p = 1) liefert n_eff = 2,27.
  **Grenzen:** das PML ist für den Index 1 ausgelegt (Substrat und Schichten höher: Richtwerte); der Querschnitt aus dem Fluss um das Objekt ist an verlustbehafteten, berührenden Schichten nicht eindeutig; Objekte, die eine Grenzfläche schneiden, gibt es nicht;
  ein Stapel geht nur mit Gmsh-Geometrie; Q der Datei beruht auf der Projektion der Oberfläche mit geraden Facetten; 3D-Wellenleiter sind mit p = 1 nicht konvergiert und mit p ≥ 2 teuer; keine Ringresonator-Vorlage (große Netze, `ring_mesh.py` der Bibliothek erzeugt die Datei);
  Resonanzen (auch mit Gmsh-Netzen) und 3D-Bänder bleiben nicht gegen analytische Lösungen validiert; kein Export; keine 3D-Wärme und -Mechanik.
* Neu: Modus **„Zeitbereich: Pulse, Sonden, Spektren“** (Seitenleiste; hpfem 0.5, `TimeDomain2D` und `TimeDomain3D`, Newmark-β, implizit): 2D-Schnitt (Rechteck mit den
  Formen des Wellenleiter-Modus, E in der Ebene, je Meter in z) oder 3D-Kasten mit Kugel (`box_with_ball`, sehr grobes Netz, Grenze 30 000 Freiheitsgrade). Quelle: Gauß-
  förmige Stromdichte mit Position, Größe und Polarisation; Zeitsignal modulierter Gauß-Puls (`modulated_gaussian`), Gauß-Puls (`gaussian_pulse`) oder Dauerstrich mit
  weichem Einschalten (`TimeSignal` mit eigener Ableitung). Ränder je Seite PEC, absorbierend (Silver-Müller erster Ordnung) oder PMC; konstante reelle Materialien (bei λ₀
  ausgewertet) mit Leitfähigkeit σ je Material, Verlust Im ε nur als σ (Umrechnung σ = ε₀ ω₀ Im ε wahlweise). Zeitschritt aus „Schritten je Periode der höchsten Frequenz“
  (Voreinstellung 40) oder von Hand, Endzeit automatisch, Newmark-Parameter mit Stabilitätsprüfung (β ≥ (γ + ½)²/4), Hinweis auf den Phasenfehler (ω Δt)²/12. Ergebnisse:
  Feldfilm (Schieber und Abspielen, feste oder je-Bild-Farbskala; 3D als Schnitt auf einem Gitter), Sonden E(t), Spektrum per FFT (Frequenz oder Wellenlänge, Hann-Fenster,
  Übertragung |E|/|G′| der Quelle), Energie über der Zeit, Tabellen und CSV. Sechs Vorlagen. Neue Module `fem_time.py`, `fem_time_app.py`, `fem_time_worker.py`; der
  Formeneditor des Wellenleiter-Moduls nimmt `bump` und `prefix` entgegen. Die Spektren sind nicht gegen eine Frequenzbereichsrechnung geprüft (getestet: Spitze der
  Quelle bei λ₀, Laufzeit eines Pulses im Streifen auf 3 %, Energieerhaltung im PEC-Kasten, Dämpfung durch σ).
* Neu: **Erweiterte Studien** (Modus „Studie“, Aufgabe Optimierung): Nebenbedingungen an Messgrößen (`optimize.constraints`, `hpfem.opt.OutcomeConstraint`,
  Verfahren bayesian) und an Parameter (`optimize.linear_constraints`, `hpfem.opt.LinearConstraint` auf dem Entwurfsraum, alle Verfahren), Mehrzieloptimierung
  (Verfahren `pareto`, `hpfem.opt.pareto_optimize` mit `objectives`, `maximize`, `ref_point`) und Mehrstufen-Optimierung (`multi_fidelity`, Stufen
  `fidelities`), beide mit BoTorch (`pip install "hpfem[opt-bo]"`; `FEATS["opt_bo"]`). Formular „Nebenbedingungen und Mehrziel“ in Reiter 1 (schreibt ins
  JSON, mit Prüfung), neue Vorlage „Optimierung mit Nebenbedingungen“. Neuer Studien-Worker `fem_study_worker.py`: der Job-Runner `hpfem.run` selbst (gleiche
  Ereignisse, `results.json` und Studienspeicher), nur mit ersetztem Optimierungsschritt (hpfem 0.5.x); nur Jobs mit den neuen Schlüsseln laufen darüber,
  alle anderen wie bisher über `python -m hpfem.run`. Verlauf und Pareto-Reiter markieren Auswertungen, die eine Nebenbedingung verletzen; die Front einer
  Mehrzieloptimierung steht mit Referenzpunkt als Tabelle, Diagramm und CSV im Ergebnis. Integer- und kategoriale Entwurfsvariablen gibt es nicht: der
  Evaluator `GratingEvaluator` kennt nur Geometrie- und Materialparameter.
* Neu: Modus **„Bandstruktur (photonischer Kristall)“** (Seitenleiste; hpfem 0.5, `BandStructure2D`, Floquet-Bloch): Bänder f = ω a / (2π c) eines 2D-Kristalls
  entlang Γ–X–M–Γ (quadratisch) bzw. Γ–M–K–Γ (hexagonal), Polarisation mit E in der Ebene (H_z; die Polarisation E_z deckt der Löser nicht ab). Zelle
  maßstabsfrei in Einheiten der Gitterkonstante, Formen (Scheibe, Rechteck, Ring, Polygon, Priorität nach Reihenfolge) werden im Gitter wiederholt (periodische
  Ränder im Gmsh-Netz), nur reelle konstante ε. Ergebnis: Bandstrukturdiagramm mit Lücken über den ganzen Pfad (Δf/Mitte in %), optional dem leeren Gitter
  |k + G| als Referenz, Tabellen und CSV, Modenfelder an den Eckpunkten. Der zweite Nullwert bei Γ (konstantes Feld) wird entfernt. Die Krylov-Dimension wird fest
  gesetzt (die Automatik der Bibliothek stockt bei Γ). Vorlagen: quadratisches Stabgitter (ε = 12, r = 0,2 a; keine Lücke für E in der Ebene), leere Gitter als
  Referenz, Dreiecksgitter aus Luftlöchern (ε = 13, r = 0,4 a). Neue Module `fem_bands.py`, `fem_bands_app.py`, `fem_bands_worker.py`.
* Neu: Aufgabe **„Hohlraum“** im Rotationskörper-Modus (hpfem 0.5, `AxisymmetricCavity`): Eigenmoden eines geschlossenen Zylinders mit PEC-Wand (0 ≤ r ≤ R,
  0 ≤ z ≤ h), gefüllt mit der „Umgebung“, Teile als Einsätze, verlustfrei (einachsige Kristalle mit Achse längs z, radial oder azimutal gehen). Je Azimutordnung
  m die niedrigsten Moden: k₀, λ = 2π/k₀, f = c/λ (Tabelle, CSV, Diagramm der Frequenzen über m), Modenfelder auf dem Meridianschnitt. Für den leeren Zylinder
  steht die analytische TM/TE-Lösung daneben. Eigenes Netz ohne PML (`fem_axi.build_cavity_mesh`), Vorlage „Zylinderhohlraum R = 1 µm, h = 1,5 µm“, Abschnitt
  „Hohlraum“ im Reiter 1. Gemessen mit dem groben Netz der Vorlage (3 Elemente je Wellenlänge, p = 3): größte Abweichung von der analytischen Wellenlänge 7·10⁻⁴
  über 12 Moden (m = 0, 1, 2), TM₀₁₀ auf 1·10⁻⁷.
* Neu: **Drude-Lorentz-Materialien** und **selbstkonsistente Resonanzen**. Neuer Materialtyp „Drude-Lorentz (Fit)“ (ε∞, ħωp, ħγ und Lorentz-Oszillatoren in eV,
  `hpfem.materials.DrudeLorentz`), in allen Modi. Im Materialeditor passt „An Messdaten anpassen“ ein Drude-Lorentz-Modell an Bibliotheks-, Tabellen- und
  Drude-Materialien über einen Wellenlängenbereich an (`hpfem.materials.fit_drude_lorentz`, Vergleich Re/Im ε, größter relativer Fehler; als neues Material
  oder als Ersatz). Aufgabe „Resonanzen“: Haken „Dispersion selbstkonsistent“ reicht die dispersiven Modelle an `hpfem.grating.resonances` und führt jede Mode
  mit `grating.refine_resonance` nach (ε bei der eigenen komplexen Frequenz); die Tabelle zeigt zusätzlich λ und Q der eingefrorenen Rechnung. Neues Modul
  `fem_disp.py` (läuft im hpfem-Python). Verkippte optische Achsen gab es im periodischen Modus schon (einachsig θ, φ; zweiachsig Euler-Winkel; voller Tensor).
* Neu: **Photovoltaik** im periodischen Modus (hpfem.pv; Reiter 3 Expander „Photovoltaik“, Reiter 4 Unterreiter „Photovoltaik“): je Streupunkt die Generationsrate
  im gewählten Absorber (absorbierte Leistung je Zelle / (ħω · Fläche), reine Nachbereitung ohne zusätzlichen Lösungsschritt) und ihr seitlich gemitteltes
  Tiefenprofil G(y) (`hpfem.pv.depth_profile`, ohne PML). Bei einem Wellenlängendurchlauf gewichtet die App die Punkte mit einem Spektrum (Schwarzkörper 5778 K als
  dokumentierte Näherung oder eigenes CSV, kein Normspektrum eingebaut) und zeigt J_sc (Sammelwirkungsgrad 1, obere Grenze), Absorption im Absorber, spektral
  gewichtetes G(y) und CSV. Neue Module `fem_pv.py` (rein), `fem_pv_app.py` (Oberfläche), `pv_profile` im Worker.
* Neu: Warnung in der Seitenleiste, wenn der Quell-Build im hp-FEM-Ordner älter ist als das im gewählten Python installierte hpfem (der Quell-Build
  kommt in den `PYTHONPATH` und verdeckt die neuere Version). `fem_run.older_source_build`, Eintrag `older_source` in der Bibliotheksprüfung.
* Neu: **Emitter in der Gitterzelle** (hp-FEM 0.5, M17; Reiter 1 „Emitter in der Zelle“, Aufgabe „Emitter“ im Reiter 3): ein Gauß-Dipol in der periodischen
  Zelle. Stufe B (einzelner Dipol per Array-Scan, `grating.dipole_emission`): Purcell-Faktor je Orientierung, Anteile nach oben, unten und nicht abgestrahlt
  (absorbiert oder geführt) und Leistung in Sammelaperturen (`grating.emission_cone`); Stufe A (Bloch-Array, `grating.emit`, Leistung je Ordnung); Stufe C
  (Abstrahlmuster dP/dΩ, `grating.emission_pattern`, Aperturen aus dem Raster). Vorab schätzt `grating.emission_cost` Zahl der Lösungen, Speicher und
  Rechenzeit (`COST` im Protokoll, „Nur die Kosten schätzen“); die Suche nach geführten Moden („Pole“) lässt sich abschalten. Das Netz wird um den Emitter
  verfeinert (σ-abhängig), die Position wird gegen Grenzflächen und Zellrand geprüft (6σ in einem verlustfreien Medium), Vorschau und Netz zeigen ihn.
  Neue Module `fem_emit.py` (Modell, Prüfung, Tabellen, Abbildungen), `fem_emit_app.py` (Oberfläche), `fem_emit_worker.py` (Rechnung im Worker).
* Neu: Modus **„Wellenleiter und Ports“** (Seitenleiste; ersetzt den Modus „Emitter über Schichtstapel“, dessen halbanalytische Rechnung die Bibliothek weiter
  bietet): Aufgabe **Ports** (`WaveguidePort`, `s_parameters`): Bauteil in der Chipebene (Effektivindex-Modell, H_z) zwischen modalen Ports bei x = 0 und x = L,
  Formen Streifen, Rechteck, Scheibe, Ring und Polygon mit Priorität, Gmsh-Netz mit Verfeinerung an Grenzflächen; Ergebnis: leistungsnormierte S-Matrix (|S|², Phase),
  Transmission, Reflexion und Verlust des Anregungskanals über der Wellenlänge, effektive Indizes der Portmoden, Feldkarte einer Anregung, CSV. Aufgabe **Moden**
  (`PropagatingMode`): effektive Indizes und Felder E_t, E_z der geführten Moden eines Querschnitts in einem PEC-Kasten. Vorlagen: Streifen, Stufe 450 → 900 nm,
  Querschnitt 450 × 220 nm. Module `fem_wg.py`, `fem_wg_app.py`, `fem_wg_worker.py`.
* Neu: **Emitter außerhalb der Achse** bei Rotationskörpern (Modus „Rotationskörper“, Emitter, „außerhalb der Achse“): Abstände ρ, alle Azimutordnungen,
  `dataports.axisymmetric_emitter_response(radii=…)`. Ergebnis: Purcell-Tensor (F_ρ, F_φ, F_z, A_xz), abgestrahlter Anteil, Detektoranteile (optional mit NA) über ρ,
  Tabelle und JSON-Umschlag. Das Netz bekommt einen Kasten (6σ) je Abstand; der Worker schiebt das Netz so, dass die Oberkante des Stapels bei z = 0 liegt.
* Neu: **Kristalle bei Rotationskörpern** (hp-FEM M19 S6): der Tensor wird in (r, φ, z) gelesen (x→r, y→φ). Erlaubt sind die optische Achse längs der Symmetrieachse, radial oder
  azimutal, zweiachsig ohne Drehung und diagonale Tensoren; eine verkippte Achse wird mit Hinweis abgelehnt. Resonanzen, Streuung und Emitter auf der Achse; Umgebung und Emitterwirt
  bleiben isotrop. Emitter außerhalb der Achse gehen in hp-FEM 0.5 nicht mit Kristallen (die Bibliothek bricht ab, die App meldet es vorab).
* Neu: **Kristall-Vorlagen** in der Materialauswahl aller Modi mit Kristallunterstützung (Saphir, Quarz, Kalkspat, LiNbO3, Rutil, MgF2; konstante Hauptindizes bei λ0).
* Neu: Materialien **Ti und Cr** (Johnson & Christy 1974, Haftschichten unter Gold) in der Bibliothek und in der Auswahl „aus der Bibliothek hinzufügen“.
* Geändert: Die Bibliotheksprüfung (`fem_run.probe_library`) meldet zusätzlich `emitter`, `dataports` und `offaxis`; die App bietet nur an, was vorhanden ist.

* Neu: **Studien** (hp-FEM M16 S9b, Seitenleiste „Art des Modells“ → „Studie“): Optimierung, Rekonstruktion von Parametern aus Messwerten und
  Unsicherheits-/Sensitivitätsanalyse mit dem Job-Runner der Bibliothek (`python -m hpfem.run`, Schema 2, Aufgaben `optimize`, `reconstruct`,
  `uq`). Job-Dokument aus Vorlagen (Lamellenzelle) oder Datei als JSON bearbeiten, im Hintergrund starten, abbrechen und fortsetzen (der
  Studienspeicher `*.study.jsonl` bleibt). Ansichten: Verlauf mit dem bisher besten Wert und den Gründen fehlgeschlagener Auswertungen, Parameter
  je Auswertung, abgetastete Punkte, Pareto-Front zweier Messgrößen, Optimum, Anpassung mit Laplace-Fehlern, Korrelationen und emcee-Posterior,
  Streuung der Messgrößen und Sobol'-Balken; Verlauf, Front, Ergebnisse und Speicher als Download.
* Neu: **Kristalle (anisotrope Materialien)** im Materialeditor (hp-FEM M19, ADR-0015): einachsig (n_o, n_e, Achse über θ gegen die Normale und φ
  gegen x), zweiachsig (drei Hauptindizes, Euler-Winkel z-x-z) oder voller symmetrischer Tensor, im Laborsystem (x quer zu den Linien, y längs, z die
  Normale); in Formen, Schichten und Substrat, das Einfallsmedium bleibt isotrop. Nur mit dem Löser `hpfem.grating` (Schichten oder Substrat aus
  Kristall: `AnisotropicStack`).
* Neu: **Jones- und Mueller-Matrizen** je ausbreitungsfähiger Ordnung (`hpfem.grating.jones`, Option „Jones- und Mueller-Matrizen berechnen“):
  Reiter „Jones / Mueller“ mit der Mueller-Matrix (Einzelpunkt als Karte, Durchlauf als 4×4 Kurven, normiert auf m_00) und der Jones-Tabelle
  (Betrag, Phase, Effizienz); bei einem Kristallsubstrat sind die Zeilen der transmittierten Matrix die beiden Moden. Gegen die analytische
  Jones-Matrix einer flachen Kristallplatte auf Kristallsubstrat: 3·10⁻¹⁶.

## 0.3.0 (2026-10-09)

* Neu: **Radial unendliche Schichten** bei Rotationskörpern (durch die PML bis zur Wand): planare Spiegel, Membranen, Schichtwellenleiter; DBR-Generator
  mit planarem unterem Spiegel; Vorlagen „GaAs-Membran mit Quantenpunkt“ und „Mikrosäule auf planarem unterem Spiegel“. Gegen die exakte
  Sommerfeld-Lösung: 1,4·10⁻³ (Dipol in der Membranebene) und 4·10⁻⁴ (axial) mit numerischer Normierung.
* Neu: Option „Purcell-Normierung numerisch auf demselben Netz“.

* Neu: **Isolierte Strukturen** im Modus „periodische Struktur“ (seitliche Ränder „isoliert“): PML links und rechts, Schichtstapel als Hintergrund,
  Streu-/Absorptions-/Extinktionsbreite (Mie-Vergleich für Zylinder auf 1e-4 bis 9e-4), Fernfeld, Detektoren für den Energiefluss, Messbox;
  Vorlagen „Isolierter Zylinder (Mie-Test)“ und „Slit-Groove-Benchmark“ (S/S₀ = 2,19923 bei p = 4 gegen 2,198826, +1,9e-4).
* Geändert: Kartenansichten in `fem_app` als Funktionen (`maps_view`, `cuts_view`); „Elemente pro Eindringtiefe“ ab 0,2.

* Neu: Modus **„Rotationskörper: Resonator, Emitter“** (Seitenleiste „Art des Modells“) auf dem zylindersymmetrischen Löser von hp-FEM
  (`AxisymmetricResonance`, `AxisymmetricScattering`): Teile im Querschnitt (Zylinder/Ring, Kegelstumpf, Kugel, Ellipsoid, Torus, Polygon), Substrat,
  Generator für Mikrosäulen mit Bragg-Spiegeln, Gmsh-Meridiannetz mit Achse, PML und Emitterkasten.
* Neu: **Resonanzen** der Ordnung m (λ_res, Q, Modenfelder E_r, E_φ, E_z) und **Emitter** auf der Achse (Purcell-Faktor, β nach oben, Anteil nach
  unten über der Wellenlänge, optional um die gefundene Resonanz zentriert, Feldkarte).
* Neu: **Streuung ebener Wellen** an Rotationskörpern (beliebiger Einfallswinkel, S/P): σ_sca, σ_abs, σ_ext über λ, Streudiagramm, Nahfeld in der
  Einfallsebene, Mie-Reihe als Referenz für Kugeln (gemessen: Au-Kugel und Si-Kugel unter 45° auf 10⁻⁴ bis 10⁻³).
* Neu: **Modenzerlegung** des Purcell-Spektrums (Riesz-Projektion): Anteil der Resonanz und Hintergrund.
* Neu: Vorlagen Mikrosäule GaAs/AlAs (wie `examples/micropillar_qd`), dielektrische Kugel, Gold-Nanokugel mit Emitter, Au-Kugel in Wasser
  (Streuung), Si-Nanokugel (Mie-Resonanzen, schräger Einfall), Mikroscheibe.
* Behoben: Splitterzellen zwischen DBR-Schichten (Rundung der Grenzflächen); Boolesche Toleranz in Gmsh.
* Geändert: Materialeditor und Hilfsfunktionen in `fem_ui.py` (von beiden Modi genutzt); `fem_run.start_worker(..., script=)`.

## 0.2.0 (2026-10-09)

Anpassung an hp-FEM 0.4 und den Stand M16 auf `main` (getestet mit einem Build von `main`, Commit 1c50a5b, Python 3.11, Windows).

* Neu: Löser **„hpfem.grating“** (Ein-Aufruf-Schnittstelle der Bibliothek) als Standard: Prüfungen der Bibliothek vor jedem Punkt, skalarer E_z-Pfad
  (TE bei φ = 0), exakte Absorption, Flussbilanz, Amplituden der Ordnungen, Rechenzeit je Phase.
* Neu: **hp-Adaptivität für TE, TM und konischen Einfall** („hpfem.grating, hp-adaptiv“) mit Residuen- oder zielorientiertem **DWR-Schätzer** für R₀;
  die periodischen Ränder dürfen mitverfeinert werden.
* Neu: **Ableitungen** (`hpfem.grating.jacobian`): dR/dε, dT/dε (Re, Im) je Material, dR/dλ, dR/dθ, dR/dφ; Reiter „Ableitungen“ mit Diagramm, Tabelle, CSV.
* Neu: Aufgabe **„Resonanzen“** (`hpfem.grating.resonances`): komplexe Eigenfrequenzen, λ_res, Q, Bandstruktur über θ/φ, Modenfelder.
* Neu: **„Mit hpfem prüfen“**: Diagnosen der Bibliothek und Speicherschätzung ohne Rechnung (`fem_worker.py --check`); streifende Ordnungen an allen
  Durchlaufpunkten.
* Neu: **Phasenanzeige** im Fortschritt (Assemblierung, Faktorisierung, Lösen …), Speicherschätzung und Meldungen während der Rechnung;
  **Abbruch zwischen den Phasen** über eine Abbruchdatei statt hartem Beenden („Sofort beenden“ bleibt).
* Neu: Feldkarten für **|H|, H-Komponenten und den Poynting-Vektor** (Raster und Elemente).
* Neu: Reiter „Prüfung, Bilanz, Zeit“ (Meldungen, R + T + A, Flussbilanz, Zeit je Phase).
* Neu: Seitenleiste fragt die Bibliothek ab (Version, Backends, Threads, Funktionen) und bietet nur an, was vorhanden ist.
* Geändert: Rand-Tags des Gmsh-Netzes 1 bis 4 (`hpfem.box_tag`) statt 101 bis 104; ältere Netze neu erzeugen.
* Geändert: PYTHONPATH des Workers nur noch für einen Quellordner mit gebautem Modul (ein Quellordner ohne Modul verdeckte ein installiertes Wheel);
  „Python mit hpfem“ ist mit dem Python der App vorbelegt, wenn es hpfem hat; Materialdaten auch aus dem installierten hpfem.
* Geändert: `start_windows.bat` nimmt `.venv`, wenn vorhanden. README: Einrichtung mit einer venv, Bau von hpfem mit MSYS2-GCC, Hinweis zu `pip.ini`.
* Tests für die neuen Teile (ohne hpfem).

## 0.1.0 (2026-10-07)

* Erste Aufnahme des Prototyps ins Repository (Streamlit-App, Geometrie/Gmsh, Worker, Auswertung, Anleitung).
* Neu: `fem_materials.py` (Materialbibliothek ohne hpfem-Import; Werte wie `hpfem.materials`).
* Neu: Vorbelegung des hp-FEM-Ordners (`HPFEM_REPO`, `../hp-FEM`), Tests, CI, Startskripte, Dokumentation.

* CI-Workflow (GitHub Actions) aktiviert.
