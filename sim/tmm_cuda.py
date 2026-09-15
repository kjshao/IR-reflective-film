"""Optional CuPy/CUDA wavelength-batched coherent TMM.

The stdlib ``tmm.coherent_rt`` evaluates one wavelength at a time. Optimisers
call ``spectrum`` many times over long grids, so batching wavelengths on a
GPU can help when an NVIDIA device + CuPy are available.

This module does **not** use cuBLAS/cuFFT as a drop-in for TMM: the transfer
matrix product is still sequential over layers, but independent wavelengths
are vectorised on the device.

Requires: ``pip install cupy-cuda12x`` (or the wheel matching your CUDA toolkit).
macOS has no NVIDIA CUDA — use ``use_cuda: false`` (default) there.
"""

from __future__ import annotations

from typing import Sequence

_CUPY_ERR: Exception | None = None
try:
    import cupy as cp

    _HAS_CUPY = True
except Exception as exc:  # noqa: BLE001 — surface later when CUDA is requested
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False
    _CUPY_ERR = exc


def cuda_available() -> bool:
    """True when CuPy imports and at least one CUDA device is visible."""
    if not _HAS_CUPY:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def require_cupy():
    """Return the cupy module or raise a clear SystemExit-style error."""
    if not _HAS_CUPY:
        raise ImportError(
            "use_cuda/rt_engine=tmm_cuda requires CuPy "
            "(pip install cupy-cuda12x). "
            f"Import failed: {_CUPY_ERR}"
        )
    if not cuda_available():
        raise RuntimeError(
            "CuPy is installed but no CUDA device was found. "
            "NVIDIA GPU + CUDA drivers are required; macOS cannot use CUDA."
        )
    return cp


def _cos_theta_batch(n, n0_sin0):
    """Batched cos(theta) with the same branch choice as ``tmm._cos_theta``."""
    c = cp.sqrt(1.0 - (n0_sin0 / n) ** 2)
    probe = n * c
    flip_imag = (cp.abs(probe.imag) > 1e-12) & (probe.imag < 0)
    flip_real = (cp.abs(probe.imag) <= 1e-12) & (probe.real < 0)
    return cp.where(flip_imag | flip_real, -c, c)


def _fresnel_batch(pol: str, n_i, c_i, n_j, c_j):
    if pol == "s":
        num = n_i * c_i - n_j * c_j
        den = n_i * c_i + n_j * c_j
        return num / den, 2.0 * n_i * c_i / den
    num = n_j * c_i - n_i * c_j
    den = n_j * c_i + n_i * c_j
    return num / den, 2.0 * n_i * c_i / den


def coherent_rt_batch(
    n_layers: "cp.ndarray",
    d_layers: "cp.ndarray",
    wavelengths: "cp.ndarray",
    theta0: float,
    pol: str = "s",
) -> tuple["cp.ndarray", "cp.ndarray"]:
    """Batched coherent R,T over wavelengths.

    Parameters
    ----------
    n_layers:
        Complex index array shaped ``(n_layer, n_wl)`` (incident … exit).
    d_layers:
        Thicknesses shaped ``(n_layer,)`` metres; first/last ignored.
    wavelengths:
        Wavelengths shaped ``(n_wl,)`` metres.
    """
    if pol not in ("s", "p"):
        raise ValueError(f"pol must be 's' or 'p', got {pol!r}")
    n = n_layers
    d = d_layers
    wl = wavelengths
    n0_sin0 = n[0] * cp.sin(theta0)
    c = _cos_theta_batch(n, n0_sin0)

    r01, t01 = _fresnel_batch(pol, n[0], c[0], n[1], c[1])
    m00 = 1.0 / t01
    m01 = r01 / t01
    m10 = r01 / t01
    m11 = 1.0 / t01

    n_lay = int(n.shape[0])
    for j in range(1, n_lay - 1):
        delta = 2.0 * cp.pi * n[j] * c[j] * d[j] / wl
        p_fwd = cp.exp(-1j * delta)
        p_bwd = cp.exp(1j * delta)
        rij, tij = _fresnel_batch(pol, n[j], c[j], n[j + 1], c[j + 1])
        l00 = p_fwd / tij
        l01 = p_fwd * rij / tij
        l10 = p_bwd * rij / tij
        l11 = p_bwd / tij
        m00, m01, m10, m11 = (
            m00 * l00 + m01 * l10,
            m00 * l01 + m01 * l11,
            m10 * l00 + m11 * l10,
            m10 * l01 + m11 * l11,
        )

    r = m10 / m00
    t = 1.0 / m00
    R = cp.abs(r) ** 2
    if pol == "s":
        flux = (n[-1] * c[-1]).real / (n[0] * c[0]).real
    else:
        flux = (n[-1] * cp.conj(c[-1])).real / (n[0] * cp.conj(c[0])).real
    T = cp.abs(t) ** 2 * flux
    return R, T


def unpolarised_batch(n_layers, d_layers, wavelengths, theta0: float):
    rs, ts = coherent_rt_batch(n_layers, d_layers, wavelengths, theta0, "s")
    rp, tp = coherent_rt_batch(n_layers, d_layers, wavelengths, theta0, "p")
    return 0.5 * (rs + rp), 0.5 * (ts + tp)


def spectrum_constant_n(
    n_list: Sequence[complex],
    d_list: Sequence[float],
    wavelengths: Sequence[float],
    theta0: float = 0.0,
    polarization: str = "unpolarized",
) -> tuple[list[float], list[float]]:
    """GPU spectrum for a stack with wavelength-independent N."""
    xp = require_cupy()
    wl = xp.asarray(list(wavelengths), dtype=xp.float64)
    n_vec = xp.asarray([complex(v) for v in n_list], dtype=xp.complex128)
    # Broadcast (L,) -> (L, W)
    n = xp.broadcast_to(n_vec[:, None], (len(n_list), int(wl.size))).copy()
    d = xp.asarray(list(d_list), dtype=xp.float64)
    if polarization in ("unpolarized", "avg", "average"):
        R, T = unpolarised_batch(n, d, wl, theta0)
    elif polarization in ("s", "p"):
        R, T = coherent_rt_batch(n, d, wl, theta0, polarization)
    else:
        raise ValueError(
            f"polarization must be s/p/unpolarized, got {polarization!r}"
        )
    return xp.asnumpy(R).tolist(), xp.asnumpy(T).tolist()


def spectrum_dispersive(
    n_by_layer: Sequence[Sequence[complex]],
    d_list: Sequence[float],
    wavelengths: Sequence[float],
    theta0: float = 0.0,
    polarization: str = "unpolarized",
) -> tuple[list[float], list[float]]:
    """GPU spectrum; ``n_by_layer[i][k]`` is N of layer i at wavelengths[k]."""
    xp = require_cupy()
    wl = xp.asarray(list(wavelengths), dtype=xp.float64)
    n = xp.asarray(n_by_layer, dtype=xp.complex128)
    if n.shape != (len(d_list), wl.size):
        raise ValueError(
            f"n_by_layer shape {n.shape} != ({len(d_list)}, {wl.size})"
        )
    d = xp.asarray(list(d_list), dtype=xp.float64)
    if polarization in ("unpolarized", "avg", "average"):
        R, T = unpolarised_batch(n, d, wl, theta0)
    elif polarization in ("s", "p"):
        R, T = coherent_rt_batch(n, d, wl, theta0, polarization)
    else:
        raise ValueError(
            f"polarization must be s/p/unpolarized, got {polarization!r}"
        )
    return xp.asnumpy(R).tolist(), xp.asnumpy(T).tolist()
