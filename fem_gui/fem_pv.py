"""Photovoltaik-Auswertung der FEM-Modellwerkstatt (reines Modul, ohne hpfem).

Der Worker (``fem_worker.pv_profile``) liefert je Wellenlängenpunkt die Absorption im gewählten Absorber und das Tiefenprofil der Generationsrate,
jeweils für die einfallende Intensität 1 W/m² (``g_per_intensity`` = G / S_inc in 1/(m³ s) je W/m²). Dieses Modul gewichtet die Punkte mit einem
Sonnenspektrum und liefert daraus

* die Kurzschlussstromdichte ``J_sc = q ∫ A_abs(λ) Φ(λ) dλ`` mit dem Photonenfluss ``Φ = I(λ) λ / (h c)`` (jedes absorbierte Photon erzeugt ein
  Elektron-Loch-Paar, Sammelwirkungsgrad 1: das ist die obere Grenze des Bauteils, keine Ladungsträgersimulation),
* die spektral gewichtete Generationsrate ``G(z) = Σ_k w_k g_k(z)`` (Gewicht ``w_k = I(λ_k) Δλ_k``, Bandbreiten aus den Mittelpunkten wie bei
  ``hpfem.pv.spectral_weights``), die ein 1D-Bauteilsimulator übernehmen kann.

Spektren: ein Schwarzkörper (5778 K, auf eine Gesamtbestrahlungsstärke skaliert, eine dokumentierte Näherung an das Sonnenspektrum ohne
Absorptionslinien) oder ein eigenes Spektrum als CSV (``λ [nm], Irradianz [W/(m² nm)]``). Es ist kein Normspektrum (AM1.5) eingebaut: für
belastbare Zahlen bitte das Normspektrum als CSV laden.
"""
from __future__ import annotations

import io

import numpy as np

H_PLANCK = 6.62607015e-34
C0 = 299792458.0
K_B = 1.380649e-23
Q_E = 1.602176634e-19

BLACKBODY_T = 5778.0
BLACKBODY_RANGE_NM = (280.0, 4000.0)          # the total irradiance is the integral over this range


def blackbody_irradiance(lam_nm, temperature=BLACKBODY_T, total=1000.0, band_nm=BLACKBODY_RANGE_NM):
    """Spectral irradiance [W/(m² nm)] ∝ Planck(λ, T), scaled so that its integral over ``band_nm`` is ``total`` [W/m²]. Without absorption lines and
    without the air mass: an approximation of the solar spectrum for estimates."""
    def planck(l_nm):
        lam = np.asarray(l_nm, dtype=float) * 1e-9
        return lam**-5 / np.expm1(H_PLANCK * C0 / (lam * K_B * temperature))

    grid = np.linspace(band_nm[0], band_nm[1], 4001)
    pl = planck(grid)
    norm = float(np.sum(0.5 * (pl[1:] + pl[:-1]) * np.diff(grid)))
    return planck(lam_nm) * (float(total) / norm)


def parse_spectrum_csv(text):
    """(λ [nm], Irradianz [W/(m² nm)]) aus CSV-Text: zwei Spalten, getrennt durch Semikolon, Komma, Tab oder Leerraum; Zeilen mit # und eine
    Kopfzeile werden übersprungen; Dezimalkomma ist erlaubt, wenn das Semikolon trennt. ``ValueError`` mit Hinweis bei unbrauchbaren Daten."""
    lam, irr = [], []
    for raw in io.StringIO(text):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line:
            parts = [p.strip().replace(",", ".") for p in line.split(";")]
        else:
            parts = line.replace("\t", " ").replace(",", " ").split()
        try:
            lam.append(float(parts[0]))
            irr.append(float(parts[1]))
        except (ValueError, IndexError):
            if lam:
                raise ValueError(f"Spektrum: Zeile nicht lesbar: '{line[:60]}'") from None
            continue                                                  # header line
    if len(lam) < 2:
        raise ValueError("Spektrum: mindestens zwei Zeilen „λ [nm]; Irradianz [W/(m² nm)]“ nötig")
    lam, irr = np.array(lam), np.array(irr)
    if np.any(np.diff(lam) <= 0):
        raise ValueError("Spektrum: die Wellenlängen müssen streng aufsteigend sein")
    if np.any(irr < 0):
        raise ValueError("Spektrum: die Irradianz darf nicht negativ sein")
    return lam, irr


def band_widths(lam_nm):
    """Bandbreiten [nm] je Wellenlänge aus den Mittelpunkten (Randbänder mit dem vollen Abstand, wie ``hpfem.pv.spectral_weights``); ein einzelner
    Punkt bekommt 1 nm (der Aufrufer skaliert)."""
    lam = np.asarray(lam_nm, dtype=float)
    if len(lam) == 1:
        return np.ones(1)
    edges = np.concatenate([[lam[0]], 0.5 * (lam[1:] + lam[:-1]), [lam[-1]]])
    w = np.diff(edges)
    w[0] *= 2
    w[-1] *= 2
    return w


def irradiance_at(lam_nm, spectrum):
    """Irradianz [W/(m² nm)] des Spektrums ``(λ, I)`` an den Wellenlängen (linear interpoliert, außerhalb 0)."""
    return np.interp(np.asarray(lam_nm, dtype=float), spectrum[0], spectrum[1], left=0.0, right=0.0)


def coverage(lam_nm, spectrum):
    """Anteil der Wellenlängenpunkte, die im Wellenlängenbereich des Spektrums liegen."""
    lam = np.asarray(lam_nm, dtype=float)
    return float(np.mean((lam >= spectrum[0][0]) & (lam <= spectrum[0][-1]))) if len(lam) else 0.0


def band_weights(lam_nm, spectrum):
    """Einfallende Leistung je Band [W/m²] = I(λ_k) Δλ_k: bei einfallender Intensität 1 W/m² je Punkt das Gewicht des Punktes im Spektrum."""
    return irradiance_at(lam_nm, spectrum) * band_widths(lam_nm)


def photon_flux(lam_nm, power):
    """Photonen je s und m² einer Leistung ``power`` [W/m²] bei der Wellenlänge ``lam_nm``: P λ / (h c)."""
    return np.asarray(power, dtype=float) * np.asarray(lam_nm, dtype=float) * 1e-9 / (H_PLANCK * C0)


def jsc(lam_nm, a_abs, spectrum):
    """Kurzschlussstromdichte [mA/cm²] bei Sammelwirkungsgrad 1: q Σ_k A_abs,k Φ_k, mit den Bandgewichten des Spektrums."""
    a = np.asarray(a_abs, dtype=float)
    phi = photon_flux(lam_nm, band_weights(lam_nm, spectrum))
    return float(Q_E * np.sum(a * phi) * 0.1)                     # A/m² -> mA/cm²


def jsc_limit(lam_nm, spectrum):
    """J_sc [mA/cm²] bei vollständiger Absorption (A = 1) in demselben Wellenlängenbereich."""
    return jsc(lam_nm, np.ones(len(np.atleast_1d(lam_nm))), spectrum)


def collect(points, key="pv"):
    """Aus den Punkten des Ergebnisses: (λ [nm], A_abs, y [nm], g (n_λ × n_bins)) für alle Punkte mit PV-Daten, nach λ sortiert. None, wenn
    keine Punkte PV-Daten haben oder die Profile nicht zusammenpassen (verschiedene Bins)."""
    rows = [(p["lam_nm"], p[key]) for p in points if "error" not in p and p.get(key)]
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    n = len(rows[0][1]["y_nm"])
    rows = [r for r in rows if len(r[1]["y_nm"]) == n]
    lam = np.array([r[0] for r in rows])
    a = np.array([r[1]["A_abs"] for r in rows])
    y = np.array(rows[0][1]["y_nm"])
    g = np.array([r[1]["g_per_intensity"] for r in rows])
    return lam, a, y, g


def spectral_profile(lam_nm, g, spectrum):
    """Spektral gewichtete Generationsrate G(z) = Σ_k w_k g_k(z) [1/(m³ s)] (g: n_λ × n_bins in 1/(m³ s) je W/m²)."""
    return np.einsum("k,kz->z", band_weights(lam_nm, spectrum), np.asarray(g, dtype=float))


def integrated_generation(y_nm, profile):
    """∫ G dz [1/(m² s)] des Profils (Mittelpunktsregel auf den Bins)."""
    y = np.asarray(y_nm, dtype=float)
    dz = (y[1] - y[0]) * 1e-9 if len(y) > 1 else 0.0
    return float(np.sum(profile) * dz)


def spectrum_table(lam_nm, a_abs, spectrum):
    """Tabelle je Wellenlänge: Absorption im Absorber, einfallende Bandleistung, Photonenfluss und Stromdichte-Beitrag."""
    import pandas as pd

    w = band_weights(lam_nm, spectrum)
    phi = photon_flux(lam_nm, w)
    return pd.DataFrame({"λ (nm)": np.asarray(lam_nm, dtype=float), "A Absorber": np.asarray(a_abs, dtype=float), "Bandleistung (W/m²)": w,
                         "Photonenfluss (1/(m² s))": phi, "J Beitrag (mA/cm²)": Q_E * np.asarray(a_abs, dtype=float) * phi * 0.1})


def profile_table(y_nm, profile):
    import pandas as pd

    return pd.DataFrame({"y (nm)": np.asarray(y_nm, dtype=float), "G (1/(m³ s))": np.asarray(profile, dtype=float)})


def fig_absorption(lam_nm, a_abs, spectrum=None, absorber=""):
    """Absorption im Absorber über der Wellenlänge (EQE bei Sammelwirkungsgrad 1), mit dem Spektrum (zweite Achse), wenn gegeben."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(lam_nm, a_abs, "o-", color="C3", label=f"Absorption in {absorber}" if absorber else "Absorption")
    ax.set_xlabel("Wellenlänge (nm)")
    ax.set_ylabel("A im Absorber (≙ EQE)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    if spectrum is not None:
        ax2 = ax.twinx()
        ax2.fill_between(spectrum[0], spectrum[1], color="0.7", alpha=0.4, label="Spektrum")
        ax2.set_ylabel("Irradianz (W/(m² nm))")
        ax2.set_xlim(min(np.min(lam_nm), spectrum[0][0]), max(np.max(lam_nm), spectrum[0][-1]))
        ax2.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_profile(y_nm, profile, label="G(z)", unit="1/(m³ s)"):
    """Generationsrate über der Höhe y (Modellkoordinate, nach oben)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(profile, y_nm, "-", color="C3", label=label)
    ax.set_xlabel(f"G ({unit})")
    ax.set_ylabel("y (nm)")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig
