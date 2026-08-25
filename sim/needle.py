"""Layer-by-layer (needle-style) synthesis for a variable number of films.

Phase 1: grow / keep an IR stop stack (R_min bands).
Phase 2: improve visible pass (R_max / T_min) by prepending AR layers,
         optimising front layers with the IR core frozen, then joint refine.
"""

from __future__ import annotations

import math
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


def _qw_thickness(material: str, wl0: float, n_fn, fraction: float = 0.25) -> float:
    """Optical-thickness fraction of wl0 as physical thickness."""
    n = n_fn(material, wl0).real
    n = max(n, 1.01)
    return fraction * wl0 / n


def _design_wavelengths(bands: Sequence[BandSpec], n_centres: int = 8) -> list[float]:
    """Log-spaced centres over the union of bands (denser at short λ)."""
    if not bands:
        return [1000e-9]
    lo = min(b.wl_lo for b in bands)
    hi = max(b.wl_hi for b in bands)
    if n_centres <= 1:
        return [math.sqrt(lo * hi)]
    log_lo, log_hi = math.log(lo), math.log(hi)
    return [
        math.exp(log_lo + (i + 0.5) * (log_hi - log_lo) / n_centres)
        for i in range(n_centres)
    ]


def _pass_band_centre(bands: Sequence[BandSpec]) -> float:
    """Centre of the shortest pass-like band (R_max / T_min without R_min)."""
    pass_bands = [
        b for b in bands if b.R_min is None and (b.R_max is not None or b.T_min is not None)
    ]
    if not pass_bands:
        return 550e-9
    b = min(pass_bands, key=lambda x: x.wl_lo)
    return 0.5 * (b.wl_lo + b.wl_hi)


def _alternate_material(
    existing: Sequence[tuple[str, float]],
    hi: str = "tio2",
    lo: str = "sio2",
) -> str:
    if not existing:
        return hi
    last = existing[-1][0].lower()
    return lo if last == hi else hi


def _ar_matching_stack(
    n_of,
    *,
    hi: str,
    lo: str,
    wl_vis: float,
    n_layers: int = 3,
) -> list[tuple[str, float]]:
    """Incident-side matching layers for visible AR (start with low index)."""
    # Fractions of λ/4 that work as a broadband front matcher on a Bragg stack.
    fracs = [0.20, 0.35, 0.20, 0.30, 0.15, 0.25]
    out: list[tuple[str, float]] = []
    for i in range(n_layers):
        mat = lo if i % 2 == 0 else hi
        out.append((mat, _qw_thickness(mat, wl_vis, n_of, fracs[i % len(fracs)])))
    return out


class NeedleSynthesizer:
    """逐层增加法 + 可见光增透相位。"""

    def __init__(
        self,
        optimizer: LMThicknessOptimizer,
        *,
        high_index: str = "tio2",
        low_index: str = "sio2",
        max_layers: int = 24,
        max_add_rounds: int = 20,
        refine_after_add: bool = True,
        n_design_centres: int = 8,
        add_pair: bool = True,
        candidate_mode: str = "append",
        ar_layers: int = 4,
        ar_add_rounds: int = 4,
        front_free_extra: int = 6,
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
        self.ar_layers = ar_layers
        self.ar_add_rounds = ar_add_rounds
        self.front_free_extra = front_free_extra

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

    def _stop_ok(self, layers: Sequence[tuple[str, float]], stop_bands: Sequence[BandSpec]) -> bool:
        saved = self.opt.bands
        self.opt.bands = list(stop_bands)
        _, _, ok, _ = self.opt.evaluate(layers)
        self.opt.bands = saved
        return ok

    def _refine_front(
        self,
        layers: list[tuple[str, float]],
        n_front: int,
        *,
        stop_bands: Sequence[BandSpec],
        verbose: bool,
        coarse_rounds: int = 8,
        step0: float = 15e-9,
        require_stop: bool = True,
    ) -> tuple[list[tuple[str, float]], OptimResult]:
        """Optimise front layers; optionally reject moves that break the IR stop."""
        n_front = max(1, min(n_front, len(layers)))
        free = list(range(n_front))
        base = list(layers)
        base_cost = self.opt.cost(base)

        coarse = self.opt.coarse_descent(
            base,
            rounds=coarse_rounds,
            step0=step0,
            free_indices=free,
            accept_fn=(lambda L: self._stop_ok(L, stop_bands)) if require_stop and stop_bands else None,
            verbose=verbose,
        )
        trial = coarse.layers
        if require_stop and stop_bands and not self._stop_ok(trial, stop_bands):
            trial = base
        lm = self.opt.optimize(trial, free_indices=free, verbose=verbose)
        trial = lm.layers
        if require_stop and stop_bands and not self._stop_ok(trial, stop_bands):
            # Fall back to best stop-feasible front tweak via small coarse steps.
            trial = base
            lm = OptimResult(base, base_cost, [], 0, False, "reject_ir_break")
            saved_step = step0
            for scale in (0.5, 0.25, 0.1):
                coarse = self.opt.coarse_descent(
                    trial,
                    rounds=4,
                    step0=saved_step * scale,
                    free_indices=free,
                    accept_fn=lambda L: self._stop_ok(L, stop_bands),
                    verbose=False,
                )
                if self._stop_ok(coarse.layers, stop_bands) and self.opt.cost(
                    coarse.layers
                ) < self.opt.cost(trial):
                    trial = coarse.layers
                    lm = coarse
        return trial, lm

    def _global_scale_search(
        self,
        layers: list[tuple[str, float]],
        stop_bands: Sequence[BandSpec],
        *,
        verbose: bool,
    ) -> list[tuple[str, float]]:
        """Uniformly scale all thicknesses to shift Bragg harmonics out of the visible."""
        best = list(layers)
        best_cost = self.opt.cost(best)
        for s in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15):
            trial = [(m, max(5e-9, d * s)) for m, d in layers]
            if stop_bands and not self._stop_ok(trial, stop_bands):
                continue
            c = self.opt.cost(trial)
            if c < best_cost:
                best, best_cost = trial, c
        if verbose and best != list(layers):
            print(
                f"      global scale search: cost {self.opt.cost(layers):.6e} -> {best_cost:.6e}",
                flush=True,
            )
        return best

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
        wl_vis = _pass_band_centre(self.opt.bands)
        n_added = 0
        best_ok: list[tuple[str, float]] | None = None
        best_ok_thick = float("inf")
        last_lm: OptimResult | None = None
        search_iter = min(self.opt.max_iter, 18)
        final_iter = self.opt.max_iter

        def n_of(mat: str, wl: float):
            return dsp.MATERIALS[mat](wl)

        all_bands = list(self.opt.bands)
        stop_bands = [b for b in all_bands if b.R_min is not None]
        if not stop_bands:
            stop_bands = all_bands
        has_pass = any(
            b.R_min is None and (b.R_max is not None or b.T_min is not None)
            for b in all_bands
        )

        if verbose:
            print("  Needle / layer-addition synthesis", flush=True)
            print(
                f"    IR centres (nm) = {[round(c * 1e9) for c in centres]}, "
                f"vis λ0 = {wl_vis * 1e9:.0f} nm, max_layers = {self.max_layers}",
                flush=True,
            )

        # --- Phase 1: meet stop-band (e.g. IR R_min) ---
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

        # --- Phase 2: visible AR + full-spec refine ---
        self.opt.bands = all_bands
        self.opt.max_iter = final_iter
        tw = self.opt.thickness_weight
        self.opt.thickness_weight = 0.0

        if verbose:
            print(
                "    phase 2a: visible AR matching layers"
                if has_pass
                else "    phase 2a: skip AR (all bands are high-reflector)",
                flush=True,
            )

        # Shift harmonics first while keeping the stop.
        layers = self._global_scale_search(layers, stop_bands, verbose=verbose)

        if has_pass:
            n_core = len(layers)
            ar_candidates = []
            for n_ar in range(2, self.ar_layers + 1):
                if n_core + n_ar > self.max_layers:
                    break
                ar = _ar_matching_stack(
                    n_of,
                    hi=self.high_index,
                    lo=self.low_index,
                    wl_vis=wl_vis,
                    n_layers=n_ar,
                )
                ar_candidates.append(ar + list(layers))
            ar_candidates.insert(0, list(layers))

            best_cand = None
            best_cand_cost = float("inf")
            for cand in ar_candidates:
                n_ar = len(cand) - n_core
                n_front = min(len(cand), max(n_ar + self.front_free_extra, 4))
                trial, lm = self._refine_front(
                    cand,
                    n_front,
                    stop_bands=stop_bands,
                    verbose=False,
                    coarse_rounds=8,
                    step0=18e-9,
                    require_stop=True,
                )
                if stop_bands and not self._stop_ok(trial, stop_bands):
                    continue
                c = self.opt.cost(trial)
                if verbose:
                    _, _, _, rep = self.opt.evaluate(trial)
                    vis = next(
                        (b for b in rep if b["targets"].get("R_max") is not None),
                        None,
                    )
                    vis_s = ""
                    if vis:
                        vis_s = (
                            f"  vis Rmax={100*vis['R_max']:.1f}% "
                            f"Tmin={100*vis['T_min']:.1f}%"
                        )
                    ir = next(
                        (b for b in rep if b["targets"].get("R_min") is not None),
                        None,
                    )
                    ir_s = ""
                    if ir:
                        ir_s = f"  IR Rmin={100*ir['R_min']:.1f}%"
                    print(
                        f"      AR trial n={n_ar}: cost={c:.6e}  "
                        f"layers={len(trial)}{vis_s}{ir_s}",
                        flush=True,
                    )
                if c < best_cand_cost:
                    best_cand_cost = c
                    best_cand = trial
                    last_lm = lm
                    n_added += max(0, n_ar)

            layers = best_cand if best_cand is not None else layers

            for round_i in range(1, self.ar_add_rounds + 1):
                _, _, ok_now, _ = self.opt.evaluate(layers)
                if ok_now or len(layers) >= self.max_layers:
                    break
                mat = self.low_index if round_i % 2 == 1 else self.high_index
                d0 = _qw_thickness(mat, wl_vis, n_of, 0.15)
                cand = [(mat, d0)] + list(layers)
                if len(cand) > self.max_layers:
                    break
                n_front = min(len(cand), self.ar_layers + self.front_free_extra)
                trial, lm = self._refine_front(
                    cand,
                    n_front,
                    stop_bands=stop_bands,
                    verbose=False,
                    coarse_rounds=5,
                    step0=12e-9,
                    require_stop=True,
                )
                if stop_bands and not self._stop_ok(trial, stop_bands):
                    break
                if self.opt.cost(trial) < self.opt.cost(layers) - 1e-9:
                    layers = trial
                    last_lm = lm
                    n_added += 1
                    if verbose:
                        print(
                            f"      AR insert #{round_i}: +{mat}  "
                            f"cost={last_lm.cost:.6e}  layers={len(layers)}",
                            flush=True,
                        )
                else:
                    break

        if verbose:
            print("    phase 2b: IR-constrained joint coarse + LM", flush=True)

        stop_filter = (
            (lambda L: self._stop_ok(L, stop_bands)) if stop_bands else None
        )
        coarse = self.opt.coarse_descent(
            layers,
            rounds=16,
            step0=12e-9,
            accept_fn=stop_filter,
            verbose=verbose,
        )
        layers = coarse.layers
        last_lm = self.opt.optimize(layers, verbose=verbose)
        layers = last_lm.layers
        if stop_bands and not self._stop_ok(layers, stop_bands):
            if verbose:
                print("      joint LM broke IR stop; reverting to coarse result", flush=True)
            layers = coarse.layers
            last_lm = coarse

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

        assert last_lm is not None
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
        method=cfg.get("method", "lm"),
        adam_lr=cfg.get("adam_lr", 2e-9),
        adam_beta1=cfg.get("adam_beta1", 0.9),
        adam_beta2=cfg.get("adam_beta2", 0.999),
        adam_eps=cfg.get("adam_eps", 1e-8),
        adam_max_step=cfg.get("adam_max_step", 10e-9),
        mini_batch=bool(cfg.get("mini_batch", False)),
        batch_size=int(cfg.get("batch_size", 8)),
        n_batches=cfg.get("n_batches"),
        n_epochs=cfg.get("n_epochs"),
        shuffle_seed=cfg.get("shuffle_seed"),
    )
