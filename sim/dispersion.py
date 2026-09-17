"""Analytic and tabulated dispersion models for IR-reflective-film stacks.

Every model returns the complex refractive index N = n + i*k with the
convention Im(N) >= 0. Wavelengths are in metres.

**Defaults are PVD-oriented** for high-index / metals (``tio2`` → NIST
RF-sputter, ``ag`` → McPeak evaporated Ag, ``ito`` → König commercial film).
``sio2`` defaults to **fused silica** (Franta); use ``sio2_pvd`` for sputtered
SiO₂. Crystal / annealed TiO₂ remain under ``tio2_rutile``, ``tio2_a``. See
``materials/SOURCES.md``. Do not extrapolate far outside each table's range
without checking the original record.
"""

from __future__ import annotations

import cmath
import csv
import math
from functools import lru_cache
from pathlib import Path

H_EV_M = 1.23984193e-6  # h*c in eV·m → E[eV] = H_EV_M / λ[m]
_MATERIALS_DIR = Path(__file__).resolve().parent / "materials"

# Anatase Sellmeier extension beyond Jolivet table (λ > ~0.827 µm).
# Fit of n² = A + B / (λ² − C) to Jolivet transparent points (k < 1e-4).
_TIO2_A_SEL_A = 5.581208
_TIO2_A_SEL_B = 0.166074
_TIO2_A_SEL_C = 0.081000  # µm²
_TIO2_A_TABLE_MAX_UM = 0.82656

# Amorphous ALD TiO2 (Jolivet @ 200 °C) NIR extension.
_TIO2_AM_SEL_A = 5.369819
_TIO2_AM_SEL_B = 0.142209
_TIO2_AM_SEL_C = 0.075500
_TIO2_AM_TABLE_MAX_UM = 0.821

# Rutile ordinary ray: Jellison 2024 Sellmeier (VIS) + Bond 1965 (NIR).
# n² = 1 + A λ²/(λ² − λo²), λ in µm; Ao=4.8393, λoo=241.89 nm.
_TIO2_R_JELL_A = 4.8393
_TIO2_R_JELL_LO_UM = 0.24189
_TIO2_R_BLEND_LO_UM = 0.85
_TIO2_R_BLEND_HI_UM = 0.95
# Urbach tail (ordinary): α(cm⁻¹)=100·exp((E−Ego)/Eu); Ego at α=100 cm⁻¹.
_TIO2_R_EG_EV = 3.023
_TIO2_R_EU_EV = 0.0302
_TIO2_R_ALPHA_AT_EG_CM = 100.0
_TIO2_R_K_CAP = 3.0


def ev(wl: float) -> float:
    return H_EV_M / wl


def _eps_to_n(eps: complex) -> complex:
    n = cmath.sqrt(eps)
    return n if n.imag >= 0 else -n


def _drude_lorentz(wl, eps_inf, e_plasma, gamma):
    """eps(E) = eps_inf − Ep² / (E² + i E γ), energies in eV."""
    e = ev(wl)
    eps = eps_inf - e_plasma**2 / (e**2 + 1j * e * gamma)
    return _eps_to_n(eps)


def _sellmeier(wl, terms):
    """n² = 1 + Σ Bⱼ λ² / (λ² − Cⱼ), λ in micrometres."""
    l2 = (wl * 1e6) ** 2
    n2 = 1.0
    for b, c in terms:
        n2 += b * l2 / (l2 - c)
    return complex(n2**0.5, 0.0)


def _formula2(wl, coefficients):
    """refractiveindex.info formula 2 (Schott / Sellmeier-2).

    n² − 1 = C1 + C2 λ²/(λ²−C3) + C4 λ²/(λ²−C5) + …
    """
    l2 = (wl * 1e6) ** 2
    n2 = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        b = coefficients[i]
        c = coefficients[i + 1]
        n2 += b * l2 / (l2 - c)
    return n2**0.5


def _formula4_devore(wl, a, b, c):
    """Devore / formula-4 special case: n² = A + B / (λ² − C), λ in µm."""
    l2 = (wl * 1e6) ** 2
    return (a + b / (l2 - c)) ** 0.5


def _formula6_ciddor(wl, coefficients):
    """refractiveindex.info formula 6 (Ciddor air).

    n − 1 = C1 + C2/(C3 − λ⁻²) + C4/(C5 − λ⁻²) + …
    """
    sigma2 = 1.0 / (wl * 1e6) ** 2
    n = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n += coefficients[i] / (coefficients[i + 1] - sigma2)
    return n


# Extra aliases after hyphen→underscore normalisation.
_ALIASES = {
    "tio2a": "tio2_a",
    "tio2anatase": "tio2_a",
    "tio2rutile": "tio2_rutile",
    "tio2sputter": "tio2_pvd",
    "tio2pvd": "tio2_pvd",
    "sio2pvd": "sio2_pvd",
    "sio2fused": "sio2_fused",
}


def normalize_material_name(name: str) -> str:
    """Canonical library key: lower-case, hyphen → underscore."""
    key = str(name).strip().lower().replace("-", "_")
    return _ALIASES.get(key, key)


@lru_cache(maxsize=16)
def _load_nk_table(filename: str) -> tuple[tuple[float, float, float], ...]:
    """Load ``wavelength_um,n,k`` CSV (comment lines starting with #)."""
    path = _MATERIALS_DIR / filename
    rows: list[tuple[float, float, float]] = []
    with path.open(newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 3:
                continue
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if len(rows) < 2:
        raise RuntimeError(f"nk table {path} has fewer than 2 points")
    return tuple(rows)


@lru_cache(maxsize=8)
def _load_k_table(filename: str) -> tuple[tuple[float, float], ...]:
    path = _MATERIALS_DIR / filename
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    if len(rows) < 2:
        raise RuntimeError(f"k table {path} has fewer than 2 points")
    return tuple(rows)


@lru_cache(maxsize=8)
def _load_n_table(filename: str) -> tuple[tuple[float, float], ...]:
    """Load ``wavelength_um,n`` CSV (comment lines starting with #)."""
    path = _MATERIALS_DIR / filename
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    if len(rows) < 2:
        raise RuntimeError(f"n table {path} has fewer than 2 points")
    return tuple(rows)


def _interp_nk(table: tuple[tuple[float, float, float], ...], wl_um: float) -> complex:
    if wl_um <= table[0][0]:
        n, k = table[0][1], table[0][2]
    elif wl_um >= table[-1][0]:
        n, k = table[-1][1], table[-1][2]
    else:
        lo, hi = 0, len(table) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if table[mid][0] <= wl_um:
                lo = mid
            else:
                hi = mid
        w0, n0, k0 = table[lo]
        w1, n1, k1 = table[hi]
        t = (wl_um - w0) / (w1 - w0)
        n = n0 + t * (n1 - n0)
        k = k0 + t * (k1 - k0)
    if k < 0.0:
        k = 0.0
    return complex(n, k)


def _interp_k(table: tuple[tuple[float, float], ...], wl_um: float) -> float:
    if wl_um <= table[0][0]:
        k = table[0][1]
    elif wl_um >= table[-1][0]:
        k = table[-1][1]
    else:
        lo, hi = 0, len(table) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if table[mid][0] <= wl_um:
                lo = mid
            else:
                hi = mid
        w0, k0 = table[lo]
        w1, k1 = table[hi]
        t = (wl_um - w0) / (w1 - w0)
        k = k0 + t * (k1 - k0)
    return max(0.0, k)


def _interp_n(table: tuple[tuple[float, float], ...], wl_um: float) -> float:
    if wl_um <= table[0][0]:
        return table[0][1]
    if wl_um >= table[-1][0]:
        return table[-1][1]
    lo, hi = 0, len(table) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if table[mid][0] <= wl_um:
            lo = mid
        else:
            hi = mid
    w0, n0 = table[lo]
    w1, n1 = table[hi]
    t = (wl_um - w0) / (w1 - w0)
    return n0 + t * (n1 - n0)


def _rutile_n_jellison(wl_um: float) -> float:
    """Ordinary-ray Sellmeier, Jellison et al. 2024 (fit window 0.45–0.85 µm)."""
    l2 = wl_um * wl_um
    lo2 = _TIO2_R_JELL_LO_UM * _TIO2_R_JELL_LO_UM
    return math.sqrt(1.0 + _TIO2_R_JELL_A * l2 / (l2 - lo2))


def _rutile_n_combined(wl_um: float) -> float:
    """Jellison n (≤0.85 µm) blended to Bond n (≥0.95 µm)."""
    n_j = _rutile_n_jellison(wl_um)
    if wl_um <= _TIO2_R_BLEND_LO_UM:
        return n_j
    n_b = _interp_n(_load_n_table("tio2_rutile_bond_o.csv"), wl_um)
    if wl_um >= _TIO2_R_BLEND_HI_UM:
        return n_b
    t = (wl_um - _TIO2_R_BLEND_LO_UM) / (
        _TIO2_R_BLEND_HI_UM - _TIO2_R_BLEND_LO_UM
    )
    return (1.0 - t) * n_j + t * n_b


def _rutile_k_urbach(wl: float) -> float:
    """Ordinary-ray Urbach k from Jellison et al. 2024 transmission edge.

    α(cm⁻¹) = 100·exp((E−Ego)/Eu) with Ego=3.023 eV, Eu=0.0302 eV.
    Valid near and below the edge; above-gap values are a steep continuation
    capped at ``_TIO2_R_K_CAP`` (not a substitute for full UV ellipsometry).
    """
    e_ev = ev(wl)
    alpha_cm = _TIO2_R_ALPHA_AT_EG_CM * math.exp(
        (e_ev - _TIO2_R_EG_EV) / _TIO2_R_EU_EV
    )
    alpha_m = alpha_cm * 100.0  # cm⁻¹ → m⁻¹
    k = alpha_m * wl / (4.0 * math.pi)
    if k < 1e-12:
        return 0.0
    return min(k, _TIO2_R_K_CAP)


# ---------------------------------------------------------------------------
# Material models
# ---------------------------------------------------------------------------


def air(wl: float) -> complex:
    """Standard dry air (Ciddor 1996); k = 0.

    Valid roughly 0.23–1.69 µm. For coating TMM, n ≈ 1.00027 @ 550 nm.
    """
    n = _formula6_ciddor(wl, (0.0, 0.05792105, 238.0185, 0.00167917, 57.362))
    return complex(n, 0.0)


def silver(wl: float) -> complex:
    """Thermally evaporated Ag (McPeak et al. 2015); PVD-relevant.

    Template-stripped evaporated film. Table ~0.30–1.70 µm; outside the
    table the nearest endpoint is used. Prefer this over the old Drude toy
    model for coating stacks that include Ag.
    """
    wl_um = wl * 1e6
    return _interp_nk(_load_nk_table("ag_mcpeak.csv"), wl_um)


def ito(wl: float) -> complex:
    """Commercial ITO thin film (König et al. 2014) with Drude NIR tail.

    Tabulated ~0.25–1.0 µm (72 nm ITO on BK7, Delta Technology).
    For λ beyond the table a moderate-carrier Drude model is used, offset so
    N is continuous at the join (sheet resistance still process-dependent).
    """
    wl_um = wl * 1e6
    table = _load_nk_table("ito_konig.csv")
    wl_join = table[-1][0]
    if wl_um <= wl_join * (1.0 + 1e-12):
        return _interp_nk(table, wl_um)
    n_join = complex(table[-1][1], table[-1][2])
    n_drude_join = _drude_lorentz(
        wl_join * 1e-6, eps_inf=3.9, e_plasma=0.95, gamma=0.12
    )
    n_drude = _drude_lorentz(wl, eps_inf=3.9, e_plasma=0.95, gamma=0.12)
    return n_drude + (n_join - n_drude_join)


def sio2(wl: float) -> complex:
    """Fused silica (Franta et al. 2016) — default low-index layer.

    Bulk/fused reference matching Malitson n to ~2×10⁻⁴ in the visible.
    For magnetron-sputtered film data use ``sio2_pvd``.
    """
    return _interp_nk(_load_nk_table("sio2_franta.csv"), wl * 1e6)


def sio2_fused(wl: float) -> complex:
    """Alias for default ``sio2`` (Franta fused silica)."""
    return sio2(wl)


def sio2_pvd(wl: float) -> complex:
    """Magnetron-sputtered SiO₂ film (Lemarchand / Gao 2012–13).

    580 nm sputtered monolayer on BK7; tabulated ~0.25–2.5 µm.
    n(550 nm) ≈ 1.475 (slightly above fused silica).
    """
    return _interp_nk(_load_nk_table("sio2_pvd_lemarchand.csv"), wl * 1e6)


def tfcalc_sio2(wl: float) -> complex:
    """TFCalc SiO₂ tabulated optical constants (0.31–2.066 µm)."""
    return _interp_nk(_load_nk_table("tfcalc_sio2.csv"), wl * 1e6)


def tio2_pvd(wl: float) -> complex:
    """RF-sputtered TiO₂ film (NIST Wang 2014) — default PVD high-index.

    Denton Discovery 550, 400 W, 7 mTorr Ar. VIS–NIR from sputter ellipsometry;
    λ > 1.35 µm extended with Franta e-beam n scaled for continuity.
    n(550 nm) ≈ 2.36 (typical dense sputtered oxide, below crystal rutile).
    """
    return _interp_nk(_load_nk_table("tio2_pvd_sputter.csv"), wl * 1e6)


def tio2_eb(wl: float) -> complex:
    """E-beam evaporated TiO₂ (Franta 2015), amorphous / fine polycrystalline.

    Alternative PVD route (evaporation). n(550 nm) ≈ 2.35; wide IR coverage.
    """
    return _interp_nk(_load_nk_table("tio2_pvd_franta.csv"), wl * 1e6)


def tio2_amorphous(wl: float) -> complex:
    """ALD amorphous TiO₂ @ 200 °C (Jolivet et al. 2023).

    Tabulated to ~0.82 µm; longer λ uses a Sellmeier fit, k → 0.
    """
    wl_um = wl * 1e6
    table = _load_nk_table("tio2_amorphous_jolivet.csv")
    if wl_um <= _TIO2_AM_TABLE_MAX_UM:
        return _interp_nk(table, wl_um)
    n = _formula4_devore(wl, _TIO2_AM_SEL_A, _TIO2_AM_SEL_B, _TIO2_AM_SEL_C)
    return complex(n, 0.0)


def tio2_a(wl: float) -> complex:
    """TiO₂ anatase thin film (Jolivet et al. 2023, ALD @ 300 °C).

    Use for annealed / crystalline anatase-like coatings (higher n than
    as-deposited PVD). Tabulated ~0.25–0.827 µm; NIR Sellmeier extension.
    """
    wl_um = wl * 1e6
    table = _load_nk_table("tio2_a_jolivet.csv")
    if wl_um <= _TIO2_A_TABLE_MAX_UM:
        return _interp_nk(table, wl_um)
    n = _formula4_devore(wl, _TIO2_A_SEL_A, _TIO2_A_SEL_B, _TIO2_A_SEL_C)
    return complex(n, 0.0)


def tio2_rutile(wl: float) -> complex:
    """TiO₂ rutile ordinary ray: Jellison 2024 + Bond 1965 (crystal upper bound).

    Not representative of typical as-deposited PVD; use for annealed/dense
    crystalline reference or optimistic index contrast.
    """
    n = _rutile_n_combined(wl * 1e6)
    k = _rutile_k_urbach(wl)
    return complex(n, k)


def tfcalc_tio2_a(wl: float) -> complex:
    """TFCalc anatase TiO₂ tabulated optical constants (0.30–1.80 µm)."""
    return _interp_nk(_load_nk_table("tfcalc_tio2_a.csv"), wl * 1e6)


def tfcalc_tio2_rutile(wl: float) -> complex:
    """TFCalc rutile TiO₂ tabulated optical constants (0.30–1.80 µm)."""
    return _interp_nk(_load_nk_table("tfcalc_tio2_rutile.csv"), wl * 1e6)


def tio2(wl: float) -> complex:
    """Default TiO₂ for PVD stacks → ``tio2_pvd`` (RF-sputtered)."""
    return tio2_pvd(wl)


def glass(wl: float) -> complex:
    """Phone / display cover-glass stand-in: Schott AF32eco.

    Alkali-free alumino-borosilicate (SCHOTT Zemax catalog 2017).
    n_d ≈ 1.5115; Corning Gorilla Glass core is ≈ 1.50 @ 590 nm — similar
    class, slightly lower index. Includes catalog tabulated k.
    """
    n = _formula2(
        wl,
        (
            0.0,
            1.11332955,
            0.00707493977,
            0.141527282,
            0.0225358637,
            0.860743147,
            90.1552652,
        ),
    )
    k = _interp_k(_load_k_table("glass_af32eco_k.csv"), wl * 1e6)
    return complex(n, k)


def pet(wl: float) -> complex:
    """Weakly dispersive fit for biaxially drawn PET, n(550 nm) ~ 1.65."""
    return _sellmeier(wl, [(1.6483, 0.01575)])


MATERIALS = {
    "air": air,
    "ag": silver,
    "ito": ito,
    "sio2": sio2,
    "sio2_fused": sio2_fused,
    "sio2_pvd": sio2_pvd,
    "tfcalc_sio2": tfcalc_sio2,
    "tio2": tio2,
    "tio2_pvd": tio2_pvd,
    "tio2_sputter": tio2_pvd,
    "tio2_eb": tio2_eb,
    "tio2_amorphous": tio2_amorphous,
    "tio2_a": tio2_a,
    "tio2_rutile": tio2_rutile,
    "tfcalc_tio2_a": tfcalc_tio2_a,
    "tfcalc_tio2_rutile": tfcalc_tio2_rutile,
    "glass": glass,
    "pet": pet,
}


def material_n(name: str, wl: float) -> complex:
    """Look up N(λ) with hyphen/underscore normalisation."""
    key = normalize_material_name(name)
    if key not in MATERIALS:
        raise KeyError(f"unknown material '{name}'; known: {sorted(MATERIALS)}")
    return MATERIALS[key](wl)


_DEFAULT_PRINT_NM = (300, 400, 450, 550, 700, 800, 1000, 1300, 1550, 1800)


def library_material_names() -> list[str]:
    """Registered names, aliases that share a model listed once after the primary."""
    seen: dict[int, str] = {}
    names: list[str] = []
    for name, fn in MATERIALS.items():
        key = id(fn)
        if key not in seen:
            seen[key] = name
            names.append(name)
    return names


def print_library_nk(wavelengths_nm: tuple[int, ...] = _DEFAULT_PRINT_NM) -> None:
    """Print n and k of every unique library material at the given wavelengths."""
    names = library_material_names()
    wls_m = [w * 1e-9 for w in wavelengths_nm]
    print("Material library  N = n + i k")
    print(
        f"  {len(names)} materials, wavelengths (nm): "
        + ", ".join(str(w) for w in wavelengths_nm)
    )
    hdr = f"  {'material':<16}" + "".join(f"  {w:>18}nm" for w in wavelengths_nm)
    sub = f"  {'':16}" + "".join(f"  {'n':>8}  {'k':>8}" for _ in wavelengths_nm)
    print(hdr)
    print(sub)
    for name in names:
        row = f"  {name:<16}"
        for wl in wls_m:
            try:
                z = MATERIALS[name](wl)
                row += f"  {z.real:8.4f}  {z.imag:8.4f}"
            except (ValueError, ZeroDivisionError, OverflowError):
                row += f"  {'n/a':>8}  {'n/a':>8}"
        print(row)
    aliases = [n for n in sorted(MATERIALS) if n not in names]
    if aliases:
        print(f"\n  aliases: {', '.join(aliases)}")
    print()


def write_library_nk_csv(
    path: str,
    *,
    wl_lo_nm: float = 300.0,
    wl_hi_nm: float = 1800.0,
    step_nm: float = 5.0,
) -> None:
    """Write n(λ), k(λ) for every unique library material."""
    names = library_material_names()
    n_pts = max(1, int(round((wl_hi_nm - wl_lo_nm) / step_nm)))
    wls_nm = [wl_lo_nm + i * (wl_hi_nm - wl_lo_nm) / n_pts for i in range(n_pts + 1)]
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        header = ["wavelength_nm"]
        for name in names:
            header += [f"{name}_n", f"{name}_k"]
        fh.write(",".join(header) + "\n")
        for w_nm in wls_nm:
            wl = w_nm * 1e-9
            row = [f"{w_nm:.2f}"]
            for name in names:
                try:
                    z = MATERIALS[name](wl)
                    row += [f"{z.real:.6f}", f"{z.imag:.6f}"]
                except (ValueError, ZeroDivisionError, OverflowError):
                    row += ["", ""]
            fh.write(",".join(row) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Print n and k from the thin-film material library."
    )
    ap.add_argument(
        "--wl",
        default=",".join(str(w) for w in _DEFAULT_PRINT_NM),
        help="comma-separated wavelengths in nm (table print)",
    )
    ap.add_argument(
        "--csv",
        default="",
        help="optional path to write a dense n,k CSV (default: no file)",
    )
    ap.add_argument("--lo", type=float, default=300.0, help="CSV start wavelength (nm)")
    ap.add_argument("--hi", type=float, default=1800.0, help="CSV end wavelength (nm)")
    ap.add_argument("--step", type=float, default=5.0, help="CSV step (nm)")
    args = ap.parse_args()
    wls = tuple(int(round(float(x))) for x in args.wl.split(",") if x.strip())
    print_library_nk(wls or _DEFAULT_PRINT_NM)
    if args.csv:
        write_library_nk_csv(args.csv, wl_lo_nm=args.lo, wl_hi_nm=args.hi, step_nm=args.step)
        print(f"  wrote {args.csv}")

