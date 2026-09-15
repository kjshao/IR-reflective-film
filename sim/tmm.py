"""Pure-stdlib transfer matrix method for planar multilayer stacks.

Supports oblique incidence and s/p polarisation, which OghmaNano's built-in
TMM solver does not.  Used here as the angle-resolved cross-check for stacks
that are otherwise designed and inspected inside OghmaNano.

Sign convention: N = n + i*k with k >= 0, fields ~ exp(i*(k*z - w*t)).
"""

import cmath
import math

Layer = tuple  # (complex refractive index, thickness in metres)


def _cos_theta(n, n0_sin0):
    """Cosine of the propagation angle inside a layer, forward branch."""
    c = cmath.sqrt(1.0 - (n0_sin0 / n) ** 2)
    # The forward-propagating wave must decay, i.e. Im(n*cos) >= 0.  For a
    # transparent layer Im is ~0 and we fall back on requiring Re(n*cos) >= 0.
    probe = n * c
    if abs(probe.imag) > 1e-12:
        if probe.imag < 0:
            c = -c
    elif probe.real < 0:
        c = -c
    return c


def _fresnel(pol, n_i, c_i, n_j, c_j):
    if pol == "s":
        num = n_i * c_i - n_j * c_j
        den = n_i * c_i + n_j * c_j
        return num / den, 2.0 * n_i * c_i / den
    num = n_j * c_i - n_i * c_j
    den = n_j * c_i + n_i * c_j
    return num / den, 2.0 * n_i * c_i / den


def coherent_rt(stack, wl, theta0=0.0, pol="s"):
    """Reflectance and transmittance of a coherent stack.

    stack -- [(N, d), ...]; the first and last entries are the semi-infinite
             incident and exit media and their thickness is ignored.
    theta0 -- angle of incidence in the first medium, radians.

    Returns (R, T).  R + T + A = 1 with A the absorptance of the stack.
    """
    n = [layer[0] for layer in stack]
    d = [layer[1] for layer in stack]
    n0_sin0 = n[0] * math.sin(theta0)
    c = [_cos_theta(ni, n0_sin0) for ni in n]

    # M = interface(0,1) * prod_j [ propagate(j) * interface(j,j+1) ]
    r01, t01 = _fresnel(pol, n[0], c[0], n[1], c[1])
    m00, m01, m10, m11 = 1.0 / t01, r01 / t01, r01 / t01, 1.0 / t01

    for j in range(1, len(stack) - 1):
        delta = 2.0 * math.pi * n[j] * c[j] * d[j] / wl
        p_fwd, p_bwd = cmath.exp(-1j * delta), cmath.exp(1j * delta)
        rij, tij = _fresnel(pol, n[j], c[j], n[j + 1], c[j + 1])
        # layer matrix L = P * I, folded straight into the running product
        l00, l01 = p_fwd / tij, p_fwd * rij / tij
        l10, l11 = p_bwd * rij / tij, p_bwd / tij
        m00, m01, m10, m11 = (
            m00 * l00 + m01 * l10,
            m00 * l01 + m01 * l11,
            m10 * l00 + m11 * l10,
            m10 * l01 + m11 * l11,
        )

    r = m10 / m00
    t = 1.0 / m00
    R = abs(r) ** 2
    if pol == "s":
        flux = (n[-1] * c[-1]).real / (n[0] * c[0]).real
    else:
        flux = (n[-1] * c[-1].conjugate()).real / (n[0] * c[0].conjugate()).real
    return R, abs(t) ** 2 * flux


def with_incoherent_substrate(front, wl, theta0, pol, n_sub, d_sub, n_exit):
    """Add a thick substrate and its back surface as an incoherent slab.

    `front` is the coherent stack from the incident medium down to (and
    including) the semi-infinite substrate.  This is what a spectrophotometer
    actually measures: interference inside the thin films, but none across the
    millimetre-thick substrate.
    """
    R_f, T_f = coherent_rt(front, wl, theta0, pol)

    reverse = [(layer[0], layer[1]) for layer in reversed(front)]
    n0 = front[0][0]
    theta_sub = cmath.asin(n0 * math.sin(theta0) / n_sub)
    R_b, T_b = coherent_rt(reverse, wl, theta_sub.real, pol)

    c_sub = _cos_theta(n_sub, n0 * math.sin(theta0))
    alpha = 4.0 * math.pi * n_sub.imag / wl
    a = math.exp(-alpha * d_sub / abs(c_sub.real)) if alpha > 0 else 1.0

    back = [(n_sub, 0.0), (n_exit, 0.0)]
    R_sa, T_sa = coherent_rt(back, wl, theta_sub.real, pol)

    denom = 1.0 - R_sa * R_b * a * a
    R = R_f + T_f * a * a * R_sa * T_b / denom
    T = T_f * a * T_sa / denom
    return R, T


def unpolarised(fn, *args, **kwargs):
    """Average s and p results of any function returning (R, T)."""
    rs, ts = fn(*args, pol="s", **kwargs)
    rp, tp = fn(*args, pol="p", **kwargs)
    return 0.5 * (rs + rp), 0.5 * (ts + tp)
