"""Analytic and tabulated dispersion models for IR-reflective-film stacks.

Every model returns the complex refractive index N = n + i*k with the
convention Im(N) >= 0. Wavelengths are in metres.

Sources (see ``materials/SOURCES.md``) are taken from the public-domain
refractiveindex.info database unless noted. Models cover the VIS–NIR band
used by this project; do not extrapolate far outside the documented range
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
    """Drude model for magnetron-sputtered Ag (order-of-magnitude)."""
    return _drude_lorentz(wl, eps_inf=5.0, e_plasma=9.5, gamma=0.0987)


def ito(wl: float) -> complex:
    """Drude model for moderate-carrier-density ITO (n(550 nm) ~ 1.93)."""
    return _drude_lorentz(wl, eps_inf=3.9, e_plasma=0.95, gamma=0.12)


def sio2(wl: float) -> complex:
    """Fused silica n,k from Franta et al. 2016 (tabulated).

    n agrees with Malitson 1965 to ~2×10⁻⁴ in the visible. Table spans
    ~0.20–5.0 µm (downsampled from refractiveindex.info).
    """
    return _interp_nk(_load_nk_table("sio2_franta.csv"), wl * 1e6)


def tio2_a(wl: float) -> complex:
    """TiO₂ anatase thin film (Jolivet et al. 2023, ALD @ 300 °C).

    Tabulated n,k for ~0.25–0.827 µm. For longer wavelengths, n is extended
    with a Devore-style Sellmeier fit to the transparent region and k → 0.
    """
    wl_um = wl * 1e6
    table = _load_nk_table("tio2_a_jolivet.csv")
    if wl_um <= _TIO2_A_TABLE_MAX_UM:
        return _interp_nk(table, wl_um)
    n = _formula4_devore(wl, _TIO2_A_SEL_A, _TIO2_A_SEL_B, _TIO2_A_SEL_C)
    return complex(n, 0.0)


def tio2_rutile(wl: float) -> complex:
    """TiO₂ rutile ordinary ray: Jellison 2024 + Bond 1965.

    * **n (λ ≤ 0.85 µm):** Jellison et al. Sellmeier (Ao=4.8393, λo=241.89 nm),
      fit window 0.45–0.85 µm.
    * **n (λ ≥ 0.95 µm):** Bond 1965 tabulated n_o (to 2.4 µm).
    * **n (0.85–0.95 µm):** linear blend between the two.
    * **k:** Jellison polarized-transmission Urbach tail (ordinary);
      ≈0 through the VIS–NIR; rises only near/below ~410 nm.

    Polycrystalline coatings are usually closer to n_o than n_e.
    """
    n = _rutile_n_combined(wl * 1e6)
    k = _rutile_k_urbach(wl)
    return complex(n, k)


def tio2(wl: float) -> complex:
    """Backward-compatible alias for anatase ``tio2_a``."""
    return tio2_a(wl)


def glass(wl: float) -> complex:
    """Phone / display cover-glass stand-in: Schott AF32eco.

    Alkali-free alumino-borosilicate (SCHOTT Zemax catalog 2017).
    n_d ≈ 1.5115; Corning Gorilla Glass core is ≈ 1.50 @ 590 nm — similar
    class, slightly lower index. Includes catalog tabulated k.
    """
    # formula 2 coefficients from AF32ECO.yml
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
    "tio2": tio2,  # alias → anatase
    "tio2_a": tio2_a,
    "tio2_rutile": tio2_rutile,
    "glass": glass,
    "pet": pet,
}


def material_n(name: str, wl: float) -> complex:
    """Look up N(λ) with hyphen/underscore normalisation."""
    key = normalize_material_name(name)
    if key not in MATERIALS:
        raise KeyError(f"unknown material '{name}'; known: {sorted(MATERIALS)}")
    return MATERIALS[key](wl)
