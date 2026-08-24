"""Analytic dispersion models for IR-reflective-film stacks.

Every model returns the complex refractive index N = n + i*k with the
convention Im(N) >= 0, wavelengths in metres.

The Drude-Lorentz parameters below are order-of-magnitude representative of
sputtered films, not a substitute for measured data.  Before committing a
design to process, replace these with ellipsometry or refractiveindex.info
values -- the near-infrared response of Ag and ITO in particular varies
strongly with deposition conditions and carrier density.
"""

import cmath

H_EV_M = 1.23984193e-6  # h*c in eV*m, so E[eV] = H_EV_M / lambda[m]


def ev(wl):
    return H_EV_M / wl


def _drude_lorentz(wl, eps_inf, e_plasma, gamma):
    """eps(E) = eps_inf - Ep^2 / (E^2 + i*E*gamma), all energies in eV."""
    e = ev(wl)
    eps = eps_inf - e_plasma**2 / (e**2 + 1j * e * gamma)
    return _eps_to_n(eps)


def _eps_to_n(eps):
    n = cmath.sqrt(eps)
    return n if n.imag >= 0 else -n


def _sellmeier(wl, terms):
    """n^2 = 1 + sum_j B_j * l^2 / (l^2 - C_j), with l in micrometres."""
    l2 = (wl * 1e6) ** 2
    n2 = 1.0
    for b, c in terms:
        n2 += b * l2 / (l2 - c)
    return complex(n2**0.5, 0.0)


def air(wl):
    return complex(1.0, 0.0)


def silver(wl):
    """Drude model for magnetron-sputtered Ag.

    Slightly lossier than Johnson & Christy in the NIR, so predicted visible
    transmission is on the conservative side.
    """
    return _drude_lorentz(wl, eps_inf=5.0, e_plasma=9.5, gamma=0.0987)


def ito(wl):
    """Drude model for a moderate-carrier-density ITO (n(550 nm) ~ 1.93).

    A higher carrier density (lower sheet resistance) pushes the screened
    plasma edge to shorter wavelengths and adds free-carrier absorption
    across the whole NIR band, which directly degrades the stop band.
    """
    return _drude_lorentz(wl, eps_inf=3.9, e_plasma=0.95, gamma=0.12)


def tio2(wl):
    """Single-oscillator Sellmeier fit to sputtered TiO2 (n = 2.35 @ 550 nm).

    Transparent above ~400 nm; the k values below the band edge are not
    modelled, so do not trust results at wavelengths under 380 nm.
    """
    return _sellmeier(wl, [(4.0719, 0.030143)])


def sio2(wl):
    """Malitson Sellmeier coefficients for fused silica."""
    return _sellmeier(
        wl,
        [
            (0.6961663, 0.0684043**2),
            (0.4079426, 0.1162414**2),
            (0.8974794, 9.896161**2),
        ],
    )


def glass(wl):
    """Schott N-BK7, a stand-in for display cover glass."""
    return _sellmeier(
        wl,
        [
            (1.03961212, 0.00600069867),
            (0.231792344, 0.0200179144),
            (1.01046945, 103.560653),
        ],
    )


def pet(wl):
    """Weakly dispersive fit for biaxially drawn PET, n(550 nm) ~ 1.65."""
    return _sellmeier(wl, [(1.6483, 0.01575)])


MATERIALS = {
    "air": air,
    "ag": silver,
    "ito": ito,
    "tio2": tio2,
    "sio2": sio2,
    "glass": glass,
    "pet": pet,
}
