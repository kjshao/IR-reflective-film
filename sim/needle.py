"""Layer-by-layer (needle-style) synthesis for a variable number of films.

Outer loop: append high/low-index quarter-wave pairs (chirped across the
longest target band), then re-optimise all thicknesses with damped least
squares until band specs are met or the layer budget is exhausted.  Among
feasible stacks, keep the one with smallest total physical thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from lm_optimizer import BandSpec, LMThicknessOptimizer, OptimResult
from rt_calculator import RTCalculator


@dataclass
class NeedleResult:
    layers: list[tuple[str, float]]
    before_layers: list[tuple[str, float]]
    cost: float
    specs_ok: bool
    n_added: int
    lm: OptimResult
    report: list[dict]


def _qw_thickness(material: str, wl0: float, n_fn) -> float:
    """Quarter-wave physical thickness at design wavelength wl0."""
    n = n_fn(material, wl0).real
    n = max(n, 1.01)
    return 0.25 * wl0 / n


def _design_wavelengths(bands: Sequence[BandSpec], n_centres: int = 4) -> list[float]:
    """Centres spanning the longest band (typically the IR stop)."""
    if not bands:
        return [1000e-9]
    b = max(bands, key=lambda x: x.wl_hi - x.wl_lo)
    if n_centres <= 1:
        return [0.5 * (b.wl_lo + b.wl_hi)]
    return [
        b.wl_lo + (i + 0.5) * (b.wl_hi - b.wl_lo) / n_centres
        for i in range(n_centres)
    ]


def _alternate_material(
    existing: Sequence[tuple[str, float]],
    hi: str = "tio2",
    lo: str = "sio2",
) -> str:
    if not existing:
        return hi
    last = existing[-1][0].lower()
    return lo if last == hi else hi


class NeedleSynthesizer:
    """逐层增加法: append H/L pairs and LM-optimise after each addition."""

    def __init__(
        self,
        optimizer: LMThicknessOptimizer,
        *,
        high_index: str = "tio2",
        low_index: str = "sio2",
        max_layers: int = 24,
        max_add_rounds: int = 20,
        refine_after_add: bool = True,
        n_design_centres: int = 4,
        add_pair: bool = True,
        candidate_mode: str = "append",
    ):
        self.opt = optimizer
        self.high_index = high_index
        self.low_index = low_index
        self.max_layers = max_layers
        self.max_add_rounds = max_add_rounds
        self.refine_after_add = refine_after_add
        self.n_design_centres = n_design_centres
        self.add_pair = add_pair
        self.candidate_mode = candidate_mode

    def _new_layers(self, layers, wl0, n_of):
        """One layer or one H/L pair at quarter-wave thickness for wl0."""
        first = _alternate_material(layers, self.high_index, self.low_index)
        d1 = _qw_thickness(first, wl0, n_of)
        added = [(first, d1)]
        if self.add_pair:
            second = self.low_index if first == self.high_index else self.high_index
            d2 = _qw_thickness(second, wl0, n_of)
            added.append((second, d2))
        return added

    def run(
        self,
        initial_layers: Sequence[tuple[str, float]],
        *,
        verbose: bool = True,
        force_add_until: int | None = None,
    ) -> NeedleResult:
        import dispersion as dsp

        before = [(m, float(d)) for m, d in initial_layers]
        layers = [tuple(l) for l in before]
        centres = _design_wavelengths(self.opt.bands, self.n_design_centres)
        n_added = 0
        best_ok: list[tuple[str, float]] | None = None
        best_ok_thick = float("inf")
        last_lm: OptimResult | None = None
        # Fewer LM iterations while searching; full refine at the end.
        search_iter = min(self.opt.max_iter, 18)
        final_iter = self.opt.max_iter

        def n_of(mat: str, wl: float):
            return dsp.MATERIALS[mat](wl)

        all_bands = list(self.opt.bands)
        # Phase-1 bands: those that mainly need a stop (R_min) — grow the stack
        # for these first so the IR reflector exists before visible AR tuning.
        stop_bands = [b for b in all_bands if b.R_min is not None]
        if not stop_bands:
            stop_bands = all_bands

        if verbose:
            print("  Needle / layer-addition synthesis", flush=True)
            print(
                f"    design centres (nm) = {[round(c * 1e9) for c in centres]}, "
                f"max_layers = {self.max_layers}",
                flush=True,
            )

        # --- Phase 1: meet stop-band (e.g. IR R_min) while adding pairs ---
        self.opt.bands = stop_bands
        self.opt.max_iter = search_iter
        if verbose:
            print("    phase 1: stop-band synthesis", flush=True)

        _, _, stop_ok, _ = self.opt.evaluate(layers)
        best_stop = list(layers) if stop_ok else None
        best_stop_cost = self.opt.cost(layers) if stop_ok else float("inf")
        if stop_ok:
            if verbose:
                print(
                    f"    phase 1: initial stack already meets stop-band "
                    f"(Σd={sum(d for _, d in layers)*1e9:.1f} nm); skip growth",
                    flush=True,
                )
            last_lm = OptimResult(
                list(layers), best_stop_cost, [], 0, True, "stop_ok_initial"
            )
        else:
            last_lm = self.opt.optimize(layers, verbose=verbose)
            layers = last_lm.layers
            _, _, stop_ok, _ = self.opt.evaluate(layers)
            if stop_ok:
                best_stop, best_stop_cost = list(layers), last_lm.cost
            if verbose:
                print(
                    f"    phase 1 after LM: stop_ok={stop_ok}  cost={last_lm.cost:.6e}  "
                    f"Σd={sum(d for _, d in layers)*1e9:.1f} nm",
                    flush=True,
                )

            for round_i in range(1, self.max_add_rounds + 1):
                if stop_ok and force_add_until is None:
                    break
                if len(layers) >= self.max_layers:
                    if verbose:
                        print("    reached max_layers; stop adding", flush=True)
                    break

                wl0 = centres[(round_i - 1) % len(centres)]
                chunk = self._new_layers(layers, wl0, n_of)
                if len(layers) + len(chunk) > self.max_layers:
                    chunk = chunk[: max(0, self.max_layers - len(layers))]
                if not chunk:
                    break

                cand = list(layers) + chunk
                if self.refine_after_add:
                    lm = self.opt.optimize(cand, verbose=False)
                    layers = lm.layers
                    last_lm = lm
                else:
                    layers = cand
                    last_lm = OptimResult(
                        layers, self.opt.cost(layers), [], 0, False, ""
                    )
                n_added += len(chunk)
                _, _, stop_ok, _ = self.opt.evaluate(layers)
                thick = sum(d for _, d in layers)
                if stop_ok and last_lm.cost < best_stop_cost:
                    best_stop, best_stop_cost = list(layers), last_lm.cost
                if verbose:
                    status = "OK" if stop_ok else "miss"
                    print(
                        f"    add #{round_i}: +{len(chunk)} -> {len(layers)} layers  "
                        f"λ0={wl0*1e9:.0f} nm  cost={last_lm.cost:.6e}  "
                        f"Σd={thick*1e9:.1f} nm  stop[{status}]",
                        flush=True,
                    )

        if best_stop is not None:
            layers = best_stop
            if verbose:
                print(
                    f"    phase 1 best stop stack: {len(layers)} layers  "
                    f"Σd={sum(d for _, d in layers)*1e9:.1f} nm",
                    flush=True,
                )

        # --- Phase 2: all band specs, fixed layer count, thickness shrink ---
        self.opt.bands = all_bands
        self.opt.max_iter = final_iter
        if verbose:
            print("    phase 2: full-spec LM refine", flush=True)
        tw = self.opt.thickness_weight
        # Keep thickness pressure mild until all specs are met.
        self.opt.thickness_weight = 0.0
        # Large-step coordinate descent first (Bragg harmonics need big moves).
        coarse = self.opt.coarse_descent(layers, rounds=10, step0=12e-9, verbose=verbose)
        layers = coarse.layers
        last_lm = self.opt.optimize(layers, verbose=verbose)
        layers = last_lm.layers
        _, _, ok, report = self.opt.evaluate(layers)
        if ok:
            self.opt.thickness_weight = max(tw, 0.05)
            last_lm = self.opt.optimize(layers, verbose=verbose)
            layers = last_lm.layers
            _, _, ok, report = self.opt.evaluate(layers)
            thick = sum(d for _, d in layers)
            if ok and thick < best_ok_thick:
                best_ok, best_ok_thick = list(layers), thick
        self.opt.thickness_weight = tw

        if best_ok is not None:
            layers = best_ok
            last_lm = self.opt.optimize(layers, verbose=False)
            layers = last_lm.layers
            _, _, ok, report = self.opt.evaluate(layers)

        return NeedleResult(
            layers=list(layers),
            before_layers=before,
            cost=last_lm.cost,
            specs_ok=ok,
            n_added=n_added,
            lm=last_lm,
            report=report,
        )


def make_optimizer_from_config(
    calculator: RTCalculator,
    bands: Sequence[BandSpec],
    cfg: dict,
) -> LMThicknessOptimizer:
    return LMThicknessOptimizer(
        calculator,
        bands,
        theta0=cfg.get("theta0", 0.0),
        incident=cfg.get("incident", "air"),
        substrate=cfg.get("substrate", "glass"),
        substrate_thickness=cfg.get("substrate_thickness", 0.7e-3),
        exit_medium=cfg.get("exit_medium", "air"),
        polarization=cfg.get("polarization", "unpolarized"),
        substrate_model=cfg.get("substrate_model", "semi_infinite"),
        wavelength_step=cfg.get("wavelength_step", 10e-9),
        thickness_weight=cfg.get("thickness_weight", 0.02),
        fd_step=cfg.get("fd_step", 0.5e-9),
        lambda0=cfg.get("lambda0", 1e-2),
        max_iter=cfg.get("max_iter", 40),
    )
