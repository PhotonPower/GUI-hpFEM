"""Dispersive Materialmodelle für hpfem (läuft im Python mit hpfem; vom Worker importiert, als Skript für den Drude-Lorentz-Fit).

* ``dispersive_model(spec)``: hpfem.materials-Modell (Dispersive) zu einem Material-Dict aus fem_materials, damit hpfem.grating die
  Permittivität bei der komplexen Eigenfrequenz auswerten kann (analytische Fortsetzung für Drude-Lorentz und Konstanten, Realteil von ω für Tabellen).
* ``fit(spec, lam_lo_nm, lam_hi_nm, oscillators)``: hpfem.materials.fit_drude_lorentz an Bibliotheks- oder Tabellendaten, Ergebnis als
  ``drude_lorentz``-Spec (Energien in eV) und größter relativer Fehler von ε.

    python fem_disp.py fit <job.json>     job.json: {"spec": .., "lam_lo_nm": .., "lam_hi_nm": .., "oscillators": n, "count": 60}
                                          Ausgabe: eine Zeile ``FIT{json}``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_materials as fm  # noqa: E402


def dispersive_model(spec):
    import hpfem.materials as hm

    t = spec["type"]
    ev = 1.0 / fm.HBAR_EV_S                                   # rad/s per eV
    if t == "library":
        if spec["name"] in ("air", "vacuum"):
            return hm.Constant(complex(fm.eps_at(spec, 500.0)))
        return hm.get(spec["name"])
    if t in ("index", "eps"):
        return hm.Constant(complex(fm.eps_at(spec, 500.0)))
    if t == "drude":
        return hm.DrudeLorentz(float(spec["eps_inf"]), float(spec["omega_p_eV"]) * ev, float(spec["gamma_eV"]) * ev)
    if t == "drude_lorentz":
        osc = tuple((float(f), float(ej) * ev, float(gj) * ev) for f, ej, gj in spec.get("osc", []))
        return hm.DrudeLorentz(float(spec["eps_inf"]), float(spec["omega_p_eV"]) * ev, float(spec["gamma_eV"]) * ev, osc)
    if t == "table":
        rows = sorted(spec["rows"])
        w, n, k = (np.array(c, dtype=float) for c in zip(*rows))
        return hm.Tabulated(w * 1e-9, n, k, name="Tabelle", out_of_range="clamp")
    raise ValueError(f"Material vom Typ '{t}' hat kein dispersives Modell (Kristalle sind nicht dispersiv)")


def fit(spec, lam_lo_nm, lam_hi_nm, oscillators=1, count=60):
    import hpfem.materials as hm

    if spec["type"] not in ("library", "table", "drude", "drude_lorentz"):
        raise ValueError("Angepasst werden können Bibliotheks-, Tabellen- und Drude-Materialien (konstante Werte haben keine Dispersion).")
    mat = dispersive_model(spec)
    lam = np.geomspace(float(lam_lo_nm), float(lam_hi_nm), int(count)) * 1e-9
    if hasattr(mat, "range") and mat.range:
        lo, hi = mat.range
        if lam[0] < lo * (1 - 1e-9) or lam[-1] > hi * (1 + 1e-9):
            raise ValueError(f"Der Bereich {lam_lo_nm:g}–{lam_hi_nm:g} nm liegt außerhalb der Daten ({lo * 1e9:.0f}–{hi * 1e9:.0f} nm)")
    model, err = hm.fit_drude_lorentz(mat, lam, oscillators=int(oscillators))
    hbar_ev = fm.HBAR_EV_S
    out = {"type": "drude_lorentz", "eps_inf": float(model.eps_inf), "omega_p_eV": float(model.omega_p) * hbar_ev, "gamma_eV": float(model.gamma) * hbar_ev,
           "osc": [[float(f), float(w) * hbar_ev, float(g) * hbar_ev] for f, w, g in model.oscillators]}
    lam_nm = lam * 1e9
    ref = [complex(fm.eps_at(spec, float(x))) if spec["type"] != "library" or spec["name"] not in ("air", "vacuum") else 1.0 + 0j for x in lam_nm]
    return dict(spec=out, max_rel_error=float(err), lam_nm=lam_nm.tolist(), eps_ref_re=[v.real for v in ref], eps_ref_im=[v.imag for v in ref],
                eps_fit_re=[float(fm.eps_at(out, float(x)).real) for x in lam_nm], eps_fit_im=[float(fm.eps_at(out, float(x)).imag) for x in lam_nm])


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if len(argv) != 2 or argv[0] != "fit":
        print(__doc__)
        return 2
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        out = fit(job["spec"], job["lam_lo_nm"], job["lam_hi_nm"], job.get("oscillators", 1), job.get("count", 60))
    except Exception as exc:
        out = dict(error=f"{type(exc).__name__}: {exc}")
    print("FIT" + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
