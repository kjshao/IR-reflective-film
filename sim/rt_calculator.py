"""Reflectance / transmittance calculators for multilayer film stacks.

Default engine is the in-repo TMM solver. An ExternalRTCalculator hook is
provided for plugging in another program that returns R(lambda) and T(lambda).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import dispersion as dsp
import tmm

# Semi-infinite media use thickness 0; substrate may be treated as incoherent.
DEFAULT_SUBSTRATE_THICKNESS = 0.7e-3


class RTCalculator(ABC):
    """Common interface: R and T arrays for a coherent coating on a substrate."""

    @abstractmethod
    def spectrum(
        self,
        layers: Sequence[tuple[str, float]],
        wavelengths: Sequence[float],
        theta0: float = 0.0,
        *,
        incident: str = "air",
        substrate: str = "glass",
        substrate_thickness: float = DEFAULT_SUBSTRATE_THICKNESS,
        exit_medium: str = "air",
        polarization: str = "unpolarized",
        substrate_model: str = "semi_infinite",
    ) -> tuple[list[float], list[float]]:
        """Return (R, T) lists matching `wavelengths` (metres)."""


def _n(name: str, wl: float) -> complex:
    if name not in dsp.MATERIALS:
        raise KeyError(f"unknown material '{name}'; known: {sorted(dsp.MATERIALS)}")
    return dsp.MATERIALS[name](wl)


def _coherent_pair(
    stack: list,
    wl: float,
    theta0: float,
    polarization: str,
) -> tuple[float, float]:
    if polarization in ("unpolarized", "avg", "average"):
        return tmm.unpolarised(tmm.coherent_rt, stack, wl, theta0)
    if polarization not in ("s", "p"):
        raise ValueError(f"polarization must be s/p/unpolarized, got {polarization!r}")
    return tmm.coherent_rt(stack, wl, theta0, polarization)


def _eval_one(
    layers: Sequence[tuple[str, float]],
    wl: float,
    theta0: float,
    incident: str,
    substrate: str,
    substrate_thickness: float,
    exit_medium: str,
    polarization: str,
    substrate_model: str = "semi_infinite",
) -> tuple[float, float]:
    n_sub = _n(substrate, wl)
    front = [(_n(incident, wl), 0.0)]
    front += [(_n(m, wl), d) for m, d in layers]
    front.append((n_sub, 0.0))

    model = (substrate_model or "semi_infinite").lower()
    if model in ("semi_infinite", "semi-infinite", "infinite"):
        # Standard coating-design metric: ignore the far side of a thick substrate.
        return _coherent_pair(front, wl, theta0, polarization)

    if model not in ("incoherent", "incoherent_slab", "slab"):
        raise ValueError(
            f"substrate_model must be semi_infinite or incoherent_slab, got {substrate_model!r}"
        )

    kwargs = dict(
        front=front,
        wl=wl,
        theta0=theta0,
        n_sub=n_sub,
        d_sub=substrate_thickness,
        n_exit=_n(exit_medium, wl),
    )
    if polarization in ("unpolarized", "avg", "average"):
        return tmm.unpolarised(tmm.with_incoherent_substrate, **kwargs)
    return tmm.with_incoherent_substrate(**kwargs, pol=polarization)


class TMMCalculator(RTCalculator):
    """Built-in transfer-matrix calculator.

    ``substrate_model='semi_infinite'`` (default) matches usual coating design
    software.  ``'incoherent_slab'`` folds in the thick substrate back surface.
    """

    def spectrum(
        self,
        layers: Sequence[tuple[str, float]],
        wavelengths: Sequence[float],
        theta0: float = 0.0,
        *,
        incident: str = "air",
        substrate: str = "glass",
        substrate_thickness: float = DEFAULT_SUBSTRATE_THICKNESS,
        exit_medium: str = "air",
        polarization: str = "unpolarized",
        substrate_model: str = "semi_infinite",
    ) -> tuple[list[float], list[float]]:
        rs, ts = [], []
        for wl in wavelengths:
            r, t = _eval_one(
                layers,
                wl,
                theta0,
                incident,
                substrate,
                substrate_thickness,
                exit_medium,
                polarization,
                substrate_model,
            )
            rs.append(r)
            ts.append(t)
        return rs, ts


class CudaTMMCalculator(RTCalculator):
    """Wavelength-batched coherent TMM on NVIDIA GPU via CuPy.

    Only ``substrate_model='semi_infinite'`` is accelerated. Other models
    fall back to the CPU ``TMMCalculator`` path.
    """

    def __init__(self):
        import tmm_cuda

        tmm_cuda.require_cupy()
        self._cpu = TMMCalculator()

    def spectrum(
        self,
        layers: Sequence[tuple[str, float]],
        wavelengths: Sequence[float],
        theta0: float = 0.0,
        *,
        incident: str = "air",
        substrate: str = "glass",
        substrate_thickness: float = DEFAULT_SUBSTRATE_THICKNESS,
        exit_medium: str = "air",
        polarization: str = "unpolarized",
        substrate_model: str = "semi_infinite",
    ) -> tuple[list[float], list[float]]:
        model = (substrate_model or "semi_infinite").lower()
        if model not in ("semi_infinite", "semi-infinite", "infinite"):
            return self._cpu.spectrum(
                layers,
                wavelengths,
                theta0,
                incident=incident,
                substrate=substrate,
                substrate_thickness=substrate_thickness,
                exit_medium=exit_medium,
                polarization=polarization,
                substrate_model=substrate_model,
            )
        import tmm_cuda

        wls = list(wavelengths)
        names = [incident, *[m for m, _ in layers], substrate]
        d_list = [0.0, *[d for _, d in layers], 0.0]
        n_by_layer = [[_n(name, wl) for wl in wls] for name in names]
        return tmm_cuda.spectrum_dispersive(
            n_by_layer, d_list, wls, theta0, polarization
        )


class ExternalRTCalculator(RTCalculator):
    """Call an external program that reads a stack JSON and writes R/T CSV.

    The command template may contain ``{input}`` and ``{output}`` placeholders.
    Input JSON schema::

        {
          "wavelengths_m": [...],
          "theta0_rad": 0.0,
          "polarization": "unpolarized",
          "incident": "air",
          "substrate": "glass",
          "substrate_thickness_m": 7e-4,
          "exit_medium": "air",
          "layers": [{"material": "tio2", "thickness_m": 1e-7}, ...]
        }

    Output CSV (header optional): wavelength_m,R,T  — one row per wavelength,
    same order as the input list.
    """

    def __init__(self, command: str, timeout_s: float = 120.0):
        if "{input}" not in command or "{output}" not in command:
            raise ValueError("external command must contain {input} and {output}")
        self.command = command
        self.timeout_s = timeout_s

    def spectrum(
        self,
        layers: Sequence[tuple[str, float]],
        wavelengths: Sequence[float],
        theta0: float = 0.0,
        *,
        incident: str = "air",
        substrate: str = "glass",
        substrate_thickness: float = DEFAULT_SUBSTRATE_THICKNESS,
        exit_medium: str = "air",
        polarization: str = "unpolarized",
        substrate_model: str = "semi_infinite",
    ) -> tuple[list[float], list[float]]:
        payload = {
            "wavelengths_m": list(wavelengths),
            "theta0_rad": theta0,
            "polarization": polarization,
            "incident": incident,
            "substrate": substrate,
            "substrate_thickness_m": substrate_thickness,
            "exit_medium": exit_medium,
            "substrate_model": substrate_model,
            "layers": [{"material": m, "thickness_m": d} for m, d in layers],
        }
        with tempfile.TemporaryDirectory(prefix="film_rt_") as tmp:
            inp = os.path.join(tmp, "stack.json")
            out = os.path.join(tmp, "rt.csv")
            with open(inp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            cmd = self.command.format(input=inp, output=out)
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                timeout=self.timeout_s,
            )
            return _read_rt_csv(out, len(wavelengths))


def _read_rt_csv(path: str, n_expect: int) -> tuple[list[float], list[float]]:
    rs, ts = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("wave"):
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            if len(parts) < 3:
                continue
            rs.append(float(parts[1]))
            ts.append(float(parts[2]))
    if len(rs) != n_expect:
        raise RuntimeError(
            f"external RT output length {len(rs)} != expected {n_expect}"
        )
    return rs, ts


def make_calculator(
    engine: str = "tmm",
    external_command: str | None = None,
    *,
    use_cuda: bool = False,
) -> RTCalculator:
    engine = (engine or "tmm").lower()
    want_cuda = bool(use_cuda) or engine in ("tmm_cuda", "cuda", "cupy")
    if want_cuda:
        if engine in ("external", "ext"):
            raise ValueError("use_cuda is incompatible with rt_engine=external")
        return CudaTMMCalculator()
    if engine == "tmm":
        return TMMCalculator()
    if engine in ("external", "ext"):
        if not external_command:
            raise ValueError("rt_engine=external requires external_command")
        return ExternalRTCalculator(external_command)
    raise ValueError(
        f"unknown rt_engine {engine!r}; use 'tmm', 'tmm_cuda', or 'external'"
    )


def material_index(
    name: str,
    wavelengths: Sequence[float],
) -> list[complex]:
    """Complex refractive index n+ik from the material library."""
    return [_n(name, wl) for wl in wavelengths]


def absorption_coefficient(n_complex: complex, wl: float) -> float:
    """Absorption coefficient alpha [1/m] from N = n + i*k."""
    return 4.0 * math.pi * n_complex.imag / wl


# Callable type alias for custom engines registered by user code.
RTFactory = Callable[..., RTCalculator]
