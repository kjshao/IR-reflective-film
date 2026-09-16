"""Pareto-beam search over coating layer count and material topology.

The outer search changes the discrete material sequence.  Each fixed topology
is optimized by the existing thickness optimizer: a cheap optical-QW/Sobol
multistart is used during beam expansion, then the original optimizer settings
are restored for full-fidelity refinement of the best topologies.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from lm_optimizer import (
    BandSpec,
    LMThicknessOptimizer,
    OptimResult,
    band_merit_values,
    wavelength_grid,
)


@dataclass
class LayerSearchConfig:
    min_layers: int
    max_layers: int
    materials: tuple[str, ...]
    beam_width: int = 12
    offspring_per_parent: int = 10
    max_generations: int = 8
    stagnation_generations: int = 3
    improvement_tol: float = 1e-4
    coarse_wavelength_step: float = 40e-9
    coarse_optical_starts: int = 8
    coarse_sobol_starts: int = 8
    coarse_local_method: str = "trf"
    coarse_local_max_iter: int = 20
    pareto_archive_size: int = 64
    preserve_per_band: int = 2
    preserve_per_layer_count: int = 1
    final_top_k: int = 4
    initial_random_topologies: int = 4
    seed: int = 0
    enable_insert: bool = True
    enable_delete: bool = True
    enable_replace: bool = True
    enable_split: bool = True
    enable_pair_mutation: bool = True

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping,
        initial_layers: Sequence[tuple[str, float]],
        *,
        default_materials: Sequence[str],
    ) -> "LayerSearchConfig":
        n0 = len(initial_layers)
        materials_raw = raw.get("materials", default_materials)
        if not isinstance(materials_raw, (list, tuple)) or not materials_raw:
            raise ValueError("layer_search.materials must be a non-empty array")
        materials = tuple(str(value).strip() for value in materials_raw)
        if any(not value for value in materials):
            raise ValueError("layer_search.materials must not contain empty names")
        min_layers = int(raw.get("min_layers", max(1, n0 - 4)))
        max_layers = int(raw.get("max_layers", max(min_layers, n0 + 6)))
        if min_layers < 1 or max_layers < min_layers:
            raise ValueError(
                "layer_search requires 1 <= min_layers <= max_layers"
            )
        cfg = cls(
            min_layers=min_layers,
            max_layers=max_layers,
            materials=materials,
            beam_width=int(raw.get("beam_width", 12)),
            offspring_per_parent=int(raw.get("offspring_per_parent", 10)),
            max_generations=int(raw.get("max_generations", 8)),
            stagnation_generations=int(
                raw.get("stagnation_generations", 3)
            ),
            improvement_tol=float(raw.get("improvement_tol", 1e-4)),
            coarse_wavelength_step=1e-9
            * float(raw.get("coarse_wavelength_step_nm", 40.0)),
            coarse_optical_starts=int(raw.get("coarse_optical_starts", 8)),
            coarse_sobol_starts=int(raw.get("coarse_sobol_starts", 8)),
            coarse_local_method=str(
                raw.get("coarse_local_method", "trf")
            ).lower(),
            coarse_local_max_iter=int(
                raw.get("coarse_local_max_iter", 20)
            ),
            pareto_archive_size=int(raw.get("pareto_archive_size", 64)),
            preserve_per_band=int(raw.get("preserve_per_band", 2)),
            preserve_per_layer_count=int(
                raw.get("preserve_per_layer_count", 1)
            ),
            final_top_k=int(raw.get("final_top_k", 4)),
            initial_random_topologies=int(
                raw.get("initial_random_topologies", 4)
            ),
            seed=int(raw.get("seed", 0)),
            enable_insert=bool(raw.get("enable_insert", True)),
            enable_delete=bool(raw.get("enable_delete", True)),
            enable_replace=bool(raw.get("enable_replace", True)),
            enable_split=bool(raw.get("enable_split", True)),
            enable_pair_mutation=bool(
                raw.get("enable_pair_mutation", True)
            ),
        )
        positive = {
            "beam_width": cfg.beam_width,
            "offspring_per_parent": cfg.offspring_per_parent,
            "max_generations": cfg.max_generations,
            "stagnation_generations": cfg.stagnation_generations,
            "coarse_local_max_iter": cfg.coarse_local_max_iter,
            "pareto_archive_size": cfg.pareto_archive_size,
            "final_top_k": cfg.final_top_k,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(
                "layer_search values must be >= 1: " + ", ".join(invalid)
            )
        if cfg.coarse_optical_starts + cfg.coarse_sobol_starts < 1:
            raise ValueError(
                "layer_search coarse start counts must sum to at least 1"
            )
        if cfg.coarse_wavelength_step <= 0.0:
            raise ValueError(
                "layer_search.coarse_wavelength_step_nm must be positive"
            )
        return cfg


@dataclass
class TopologyCandidate:
    candidate_id: int
    layers: list[tuple[str, float]]
    generation: int
    operation: str
    parent_id: int | None = None
    cost: float = math.inf
    band_errors: list[float] = field(default_factory=list)
    max_violation: float = math.inf
    feasible: bool = False
    total_thickness: float = math.inf
    pareto_rank: int = 0
    crowding: float = 0.0
    fidelity: str = "coarse"
    result: OptimResult | None = None

    @property
    def topology(self) -> tuple[str, ...]:
        return tuple(material.lower() for material, _ in self.layers)


@dataclass
class LayerSearchResult:
    result: OptimResult
    ranking: list[TopologyCandidate]
    archive: list[TopologyCandidate]
    generations: int
    evaluated_topologies: int


def _band_max_violation(
    bands: Sequence[BandSpec],
    wavelengths: Sequence[float],
    R: Sequence[float],
    T: Sequence[float],
) -> float:
    violation = 0.0
    for band in bands:
        samples = [
            (r, t)
            for wl, r, t in zip(wavelengths, R, T)
            if band.wl_lo - 1e-15 <= wl <= band.wl_hi + 1e-15
        ]
        for r, t in samples:
            if band.R_min is not None:
                violation = max(violation, band.R_min - r)
            if band.R_max is not None:
                violation = max(violation, r - band.R_max)
            if band.T_min is not None:
                violation = max(violation, band.T_min - t)
            if band.T_max is not None:
                violation = max(violation, t - band.T_max)
    return max(0.0, violation)


class ParetoBeamLayerSearch:
    """Coarse-to-fine variable-layer search around an existing optimizer."""

    def __init__(
        self,
        optimizer: LMThicknessOptimizer,
        config: LayerSearchConfig,
    ):
        self.opt = optimizer
        self.config = config
        self._next_id = 1
        self._seen_topologies: set[tuple[str, ...]] = set()
        self._rng = random.Random(config.seed)

    def _new_candidate(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        generation: int,
        operation: str,
        parent_id: int | None = None,
    ) -> TopologyCandidate | None:
        repaired = self._repair(layers)
        if repaired is None:
            return None
        key = tuple(material.lower() for material, _ in repaired)
        if key in self._seen_topologies:
            return None
        self._seen_topologies.add(key)
        candidate = TopologyCandidate(
            candidate_id=self._next_id,
            layers=repaired,
            generation=generation,
            operation=operation,
            parent_id=parent_id,
        )
        self._next_id += 1
        return candidate

    def _repair(
        self,
        layers: Sequence[tuple[str, float]],
    ) -> list[tuple[str, float]] | None:
        merged: list[tuple[str, float]] = []
        for material, thickness in layers:
            material = str(material)
            if merged and merged[-1][0].lower() == material.lower():
                previous_material, previous_thickness = merged[-1]
                merged[-1] = (
                    previous_material,
                    previous_thickness + float(thickness),
                )
            else:
                merged.append((material, float(thickness)))
        if not (
            self.config.min_layers
            <= len(merged)
            <= self.config.max_layers
        ):
            return None
        materials = [material for material, _ in merged]
        values = []
        for material, thickness in merged:
            lo, hi = self.opt.bounds_for(material)
            values.append(min(max(float(thickness), lo), hi))
        try:
            values = self.opt._project(
                materials, values, list(range(len(values)))
            )
        except ValueError:
            return None
        return list(zip(materials, values))

    def _qw_thickness(self, material: str, wavelength: float) -> float:
        n_value = self.opt._sampling_refractive_index(material, wavelength)
        lo, hi = self.opt.bounds_for(material)
        return min(max(wavelength / (4.0 * n_value), lo), hi)

    def _design_wavelength(self) -> float:
        lo = min(band.wl_lo for band in self.opt.bands)
        hi = max(band.wl_hi for band in self.opt.bands)
        return math.sqrt(lo * hi)

    def _alternating_layers(
        self,
        count: int,
        *,
        first: str,
        second: str,
    ) -> list[tuple[str, float]]:
        wavelength = self._design_wavelength()
        return [
            (
                first if index % 2 == 0 else second,
                self._qw_thickness(
                    first if index % 2 == 0 else second, wavelength
                ),
            )
            for index in range(count)
        ]

    def _initial_candidates(
        self,
        initial_layers: Sequence[tuple[str, float]],
    ) -> list[TopologyCandidate]:
        candidates: list[TopologyCandidate] = []

        def add(layers, operation):
            candidate = self._new_candidate(
                layers, generation=0, operation=operation
            )
            if candidate is not None:
                candidates.append(candidate)

        add(initial_layers, "input")
        if len(self.config.materials) >= 2:
            high, low = self.config.materials[:2]
            for count in range(
                self.config.min_layers, self.config.max_layers + 1
            ):
                add(
                    self._alternating_layers(
                        count, first=high, second=low
                    ),
                    "seed_hlh",
                )
                add(
                    self._alternating_layers(
                        count, first=low, second=high
                    ),
                    "seed_lhl",
                )
        wavelength = self._design_wavelength()
        for random_index in range(self.config.initial_random_topologies):
            count = self._rng.randint(
                self.config.min_layers, self.config.max_layers
            )
            sequence: list[str] = []
            for _ in range(count):
                options = [
                    material
                    for material in self.config.materials
                    if not sequence
                    or material.lower() != sequence[-1].lower()
                ]
                sequence.append(self._rng.choice(options or self.config.materials))
            add(
                [
                    (material, self._qw_thickness(material, wavelength))
                    for material in sequence
                ],
                f"seed_random_{random_index}",
            )
        return candidates

    def _mutation_buckets(
        self,
        parent: TopologyCandidate,
        generation: int,
    ) -> list[list[tuple[list[tuple[str, float]], str]]]:
        layers = parent.layers
        wavelength = self._design_wavelength()
        buckets: list[list[tuple[list[tuple[str, float]], str]]] = []
        if self.config.enable_delete and len(layers) > self.config.min_layers:
            buckets.append(
                [
                    (layers[:index] + layers[index + 1 :], f"delete@{index}")
                    for index in range(len(layers))
                ]
            )
        if self.config.enable_insert and len(layers) < self.config.max_layers:
            insertions = []
            for position in range(len(layers) + 1):
                for material in self.config.materials:
                    insertion = (
                        material,
                        self._qw_thickness(material, wavelength),
                    )
                    insertions.append(
                        (
                            layers[:position]
                            + [insertion]
                            + layers[position:],
                            f"insert@{position}:{material}",
                        )
                    )
            buckets.append(insertions)
        if self.config.enable_replace:
            replacements = []
            for index, (old_material, thickness) in enumerate(layers):
                for material in self.config.materials:
                    if material.lower() == old_material.lower():
                        continue
                    trial = list(layers)
                    trial[index] = (material, thickness)
                    replacements.append(
                        (trial, f"replace@{index}:{material}")
                    )
            buckets.append(replacements)
        if (
            self.config.enable_split
            and len(layers) < self.config.max_layers
            and len(self.config.materials) >= 2
        ):
            splits = []
            for index, (old_material, thickness) in enumerate(layers):
                alternatives = [
                    material
                    for material in self.config.materials
                    if material.lower() != old_material.lower()
                ]
                for material in alternatives:
                    other = self._qw_thickness(material, wavelength)
                    first = (old_material, max(thickness * 0.5, 1e-12))
                    splits.append(
                        (
                            layers[:index]
                            + [first, (material, other)]
                            + layers[index + 1 :],
                            f"split@{index}:{old_material}/{material}",
                        )
                    )
                    splits.append(
                        (
                            layers[:index]
                            + [(material, other), first]
                            + layers[index + 1 :],
                            f"split@{index}:{material}/{old_material}",
                        )
                    )
            buckets.append(splits)
        if self.config.enable_pair_mutation and len(self.config.materials) >= 2:
            high, low = self.config.materials[:2]
            pairs = []
            if len(layers) + 2 <= self.config.max_layers:
                for position in range(len(layers) + 1):
                    for first, second in ((high, low), (low, high)):
                        pair = [
                            (first, self._qw_thickness(first, wavelength)),
                            (second, self._qw_thickness(second, wavelength)),
                        ]
                        pairs.append(
                            (
                                layers[:position] + pair + layers[position:],
                                f"insert_pair@{position}:{first}/{second}",
                            )
                        )
            if len(layers) - 2 >= self.config.min_layers:
                for position in range(len(layers) - 1):
                    pairs.append(
                        (
                            layers[:position] + layers[position + 2 :],
                            f"delete_pair@{position}",
                        )
                    )
            buckets.append(pairs)
        for index, bucket in enumerate(buckets):
            random.Random(
                self.config.seed
                + generation * 1_000_003
                + parent.candidate_id * 101
                + index
            ).shuffle(bucket)
        return [bucket for bucket in buckets if bucket]

    def _offspring(
        self,
        beam: Sequence[TopologyCandidate],
        generation: int,
    ) -> list[TopologyCandidate]:
        children: list[TopologyCandidate] = []
        for parent in beam:
            buckets = self._mutation_buckets(parent, generation)
            positions = [0] * len(buckets)
            while (
                len(
                    [
                        child
                        for child in children
                        if child.parent_id == parent.candidate_id
                    ]
                )
                < self.config.offspring_per_parent
                and any(
                    positions[i] < len(bucket)
                    for i, bucket in enumerate(buckets)
                )
            ):
                for bucket_index, bucket in enumerate(buckets):
                    if positions[bucket_index] >= len(bucket):
                        continue
                    layers, operation = bucket[positions[bucket_index]]
                    positions[bucket_index] += 1
                    child = self._new_candidate(
                        layers,
                        generation=generation,
                        operation=operation,
                        parent_id=parent.candidate_id,
                    )
                    if child is not None:
                        children.append(child)
                    parent_children = sum(
                        item.parent_id == parent.candidate_id
                        for item in children
                    )
                    if parent_children >= self.config.offspring_per_parent:
                        break
        return children

    def _snapshot_optimizer(self) -> dict[str, object]:
        names = (
            "method",
            "multistart_method",
            "multistart_n",
            "multistart_sampler",
            "optical_sampler_fraction",
            "max_iter",
            "multistart_final_polish_method",
            "wavelengths",
            "on_best",
        )
        return {name: getattr(self.opt, name) for name in names}

    def _restore_optimizer(self, state: Mapping[str, object]) -> None:
        for name, value in state.items():
            if name == "wavelengths":
                self.opt._set_wavelengths(value)  # type: ignore[arg-type]
            else:
                setattr(self.opt, name, value)

    def _configure_coarse(self) -> None:
        optical = max(0, self.config.coarse_optical_starts)
        sobol = max(0, self.config.coarse_sobol_starts)
        total = max(1, optical + sobol)
        self.opt.method = "multistart" if total > 1 else self.config.coarse_local_method
        self.opt.multistart_method = self.config.coarse_local_method
        self.opt.multistart_n = total
        self.opt.multistart_sampler = "optical_qw"
        self.opt.optical_sampler_fraction = optical / total
        self.opt.max_iter = self.config.coarse_local_max_iter
        self.opt.multistart_final_polish_method = "none"
        self.opt.on_best = None
        self.opt._set_wavelengths(
            wavelength_grid(self.opt.bands, self.config.coarse_wavelength_step)
        )

    def _measure(self, candidate: TopologyCandidate) -> None:
        R, T = self.opt._rt(candidate.layers)
        merits = band_merit_values(
            self.opt.bands, self.opt.wavelengths, R, T
        )
        candidate.cost = self.opt.cost(candidate.layers)
        candidate.band_errors = [
            math.sqrt(max(0.0, value)) for value in merits
        ]
        candidate.max_violation = _band_max_violation(
            self.opt.bands, self.opt.wavelengths, R, T
        )
        candidate.feasible = candidate.max_violation <= 1e-4
        candidate.total_thickness = sum(
            thickness for _, thickness in candidate.layers
        )

    def _evaluate_candidates(
        self,
        candidates: Sequence[TopologyCandidate],
        *,
        fidelity: str,
        verbose: bool,
    ) -> list[TopologyCandidate]:
        evaluated: list[TopologyCandidate] = []
        for index, candidate in enumerate(candidates, 1):
            if verbose:
                print(
                    f"    {fidelity} topology {index}/{len(candidates)}: "
                    f"id={candidate.candidate_id} "
                    f"layers={len(candidate.layers)} "
                    f"op={candidate.operation}",
                    flush=True,
                )
            try:
                result = self.opt.optimize(candidate.layers, verbose=False)
                candidate.layers = list(result.layers)
                candidate.result = result
                candidate.fidelity = fidelity
                self._measure(candidate)
                evaluated.append(candidate)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                if verbose:
                    print(
                        f"      rejected topology id={candidate.candidate_id}: "
                        f"{exc}",
                        flush=True,
                    )
        return evaluated

    @staticmethod
    def _objectives(candidate: TopologyCandidate) -> tuple[float, ...]:
        return (
            candidate.max_violation,
            *candidate.band_errors,
            candidate.total_thickness,
            float(len(candidate.layers)),
        )

    @classmethod
    def _dominates(
        cls,
        left: TopologyCandidate,
        right: TopologyCandidate,
    ) -> bool:
        if left.feasible != right.feasible:
            return left.feasible
        left_values = cls._objectives(left)
        right_values = cls._objectives(right)
        return all(a <= b for a, b in zip(left_values, right_values)) and any(
            a < b for a, b in zip(left_values, right_values)
        )

    def _assign_pareto(self, candidates: Sequence[TopologyCandidate]) -> None:
        remaining = set(range(len(candidates)))
        rank = 0
        fronts: list[list[int]] = []
        while remaining:
            front = [
                index
                for index in remaining
                if not any(
                    self._dominates(candidates[other], candidates[index])
                    for other in remaining
                    if other != index
                )
            ]
            if not front:
                front = [min(remaining)]
            for index in front:
                candidates[index].pareto_rank = rank
            fronts.append(front)
            remaining.difference_update(front)
            rank += 1

        for front in fronts:
            for index in front:
                candidates[index].crowding = 0.0
            if len(front) <= 2:
                for index in front:
                    candidates[index].crowding = math.inf
                continue
            objective_count = len(self._objectives(candidates[front[0]]))
            for objective_index in range(objective_count):
                ordered = sorted(
                    front,
                    key=lambda i: self._objectives(candidates[i])[
                        objective_index
                    ],
                )
                candidates[ordered[0]].crowding = math.inf
                candidates[ordered[-1]].crowding = math.inf
                lo = self._objectives(candidates[ordered[0]])[objective_index]
                hi = self._objectives(candidates[ordered[-1]])[objective_index]
                if hi <= lo:
                    continue
                for position in range(1, len(ordered) - 1):
                    candidate = candidates[ordered[position]]
                    if math.isinf(candidate.crowding):
                        continue
                    before = self._objectives(
                        candidates[ordered[position - 1]]
                    )[objective_index]
                    after = self._objectives(
                        candidates[ordered[position + 1]]
                    )[objective_index]
                    candidate.crowding += (after - before) / (hi - lo)

    @staticmethod
    def _base_key(candidate: TopologyCandidate):
        return (
            not candidate.feasible,
            candidate.pareto_rank,
            candidate.cost,
            candidate.total_thickness,
            len(candidate.layers),
        )

    def _select(
        self,
        candidates: Sequence[TopologyCandidate],
        limit: int,
    ) -> list[TopologyCandidate]:
        if not candidates or limit <= 0:
            return []
        self._assign_pareto(candidates)
        selected: list[TopologyCandidate] = []

        def add(candidate: TopologyCandidate) -> None:
            if (
                len(selected) < limit
                and all(
                    existing.candidate_id != candidate.candidate_id
                    for existing in selected
                )
            ):
                selected.append(candidate)

        add(min(candidates, key=self._base_key))
        n_bands = max(
            (len(candidate.band_errors) for candidate in candidates),
            default=0,
        )
        for band_index in (
            range(n_bands) if self.config.preserve_per_band > 0 else ()
        ):
            ranked = sorted(
                candidates,
                key=lambda item: (
                    not item.feasible,
                    item.band_errors[band_index],
                    item.cost,
                ),
            )
            kept = 0
            for candidate in ranked:
                before = len(selected)
                add(candidate)
                kept += len(selected) > before
                if kept >= self.config.preserve_per_band:
                    break

        by_count: dict[int, list[TopologyCandidate]] = {}
        for candidate in candidates:
            by_count.setdefault(len(candidate.layers), []).append(candidate)
        count_groups = sorted(
            by_count.values(),
            key=lambda group: self._base_key(min(group, key=self._base_key)),
        )
        for group in (
            count_groups
            if self.config.preserve_per_layer_count > 0
            else ()
        ):
            kept = 0
            for candidate in sorted(group, key=self._base_key):
                before = len(selected)
                add(candidate)
                kept += len(selected) > before
                if kept >= self.config.preserve_per_layer_count:
                    break

        remaining = sorted(
            candidates,
            key=lambda item: (
                not item.feasible,
                item.pareto_rank,
                -item.crowding,
                item.cost,
                item.total_thickness,
            ),
        )
        for candidate in remaining:
            add(candidate)
        return selected

    def run(
        self,
        initial_layers: Sequence[tuple[str, float]],
        *,
        verbose: bool = True,
    ) -> LayerSearchResult:
        original = self._snapshot_optimizer()
        archive: list[TopologyCandidate] = []
        evaluated_count = 0
        generations_run = 0
        try:
            self._configure_coarse()
            pending = self._initial_candidates(initial_layers)
            if verbose:
                print(
                    "  Pareto-beam variable-layer search\n"
                    f"    layers={self.config.min_layers}.."
                    f"{self.config.max_layers}  "
                    f"beam={self.config.beam_width}  "
                    f"generations={self.config.max_generations}  "
                    f"initial_topologies={len(pending)}",
                    flush=True,
                )
            evaluated = self._evaluate_candidates(
                pending, fidelity="coarse", verbose=verbose
            )
            evaluated_count += len(evaluated)
            if not evaluated:
                raise RuntimeError("layer search found no feasible topology")
            archive = self._select(
                evaluated, self.config.pareto_archive_size
            )
            beam = self._select(archive, self.config.beam_width)
            best_cost = min(candidate.cost for candidate in beam)
            stale = 0

            for generation in range(1, self.config.max_generations + 1):
                children = self._offspring(beam, generation)
                if not children:
                    break
                evaluated = self._evaluate_candidates(
                    children, fidelity="coarse", verbose=verbose
                )
                evaluated_count += len(evaluated)
                if not evaluated:
                    break
                archive = self._select(
                    [*archive, *evaluated],
                    self.config.pareto_archive_size,
                )
                beam = self._select(archive, self.config.beam_width)
                generations_run = generation
                generation_best = min(
                    candidate.cost for candidate in beam
                )
                relative_improvement = (
                    (best_cost - generation_best) / max(abs(best_cost), 1e-15)
                )
                if generation_best < best_cost:
                    best_cost = generation_best
                if relative_improvement > self.config.improvement_tol:
                    stale = 0
                else:
                    stale += 1
                if verbose:
                    feasible_count = sum(
                        candidate.feasible for candidate in archive
                    )
                    print(
                        f"    generation {generation}: "
                        f"children={len(evaluated)}  "
                        f"archive={len(archive)}  feasible={feasible_count}  "
                        f"best={best_cost:.6e}  stale={stale}",
                        flush=True,
                    )
                if stale >= self.config.stagnation_generations:
                    break
        finally:
            self._restore_optimizer(original)

        finalists = self._select(archive, self.config.final_top_k)
        if verbose:
            print(
                f"  Full-fidelity refinement: {len(finalists)} topologies",
                flush=True,
            )
        original_callback = self.opt.on_best
        if original_callback is not None:
            checkpoint_state = {"cost": math.inf}

            def guarded_callback(layers, cost, info):
                if float(cost) < checkpoint_state["cost"] - 1e-15:
                    checkpoint_state["cost"] = float(cost)
                    original_callback(layers, cost, info)

            self.opt.on_best = guarded_callback
        try:
            refined = self._evaluate_candidates(
                finalists, fidelity="full", verbose=verbose
            )
        finally:
            self.opt.on_best = original_callback
        if not refined:
            raise RuntimeError("full-fidelity layer refinement failed")
        ranking = self._select(refined, len(refined))
        ranking.sort(key=self._base_key)
        best = ranking[0]
        assert best.result is not None
        best.result.layers = list(best.layers)
        best.result.cost = best.cost
        best.result.residuals = self.opt.residuals(best.layers)
        best.result.final_layers = list(best.layers)
        best.result.final_cost = best.cost
        best.result.message = (
            f"pareto_beam(N={len(best.layers)}, "
            f"topologies={evaluated_count}, generations={generations_run})"
            f"+{best.result.message}"
        )
        return LayerSearchResult(
            result=best.result,
            ranking=ranking,
            archive=archive,
            generations=generations_run,
            evaluated_topologies=evaluated_count,
        )


def write_layer_search_results(
    output_prefix: str,
    result: LayerSearchResult,
) -> tuple[str, str]:
    """Write ranked full-fidelity candidates as CSV and JSON."""
    csv_path = output_prefix + ".csv"
    json_path = output_prefix + ".json"
    fieldnames = [
        "rank",
        "candidate_id",
        "layers",
        "topology",
        "cost",
        "feasible",
        "max_violation",
        "total_thickness_nm",
        "pareto_rank",
        "crowding",
        "generation",
        "operation",
        "parent_id",
        "band_errors",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(result.ranking, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "layers": len(candidate.layers),
                    "topology": "/".join(candidate.topology),
                    "cost": f"{candidate.cost:.12e}",
                    "feasible": int(candidate.feasible),
                    "max_violation": f"{candidate.max_violation:.12e}",
                    "total_thickness_nm": (
                        f"{candidate.total_thickness * 1e9:.6f}"
                    ),
                    "pareto_rank": candidate.pareto_rank,
                    "crowding": candidate.crowding,
                    "generation": candidate.generation,
                    "operation": candidate.operation,
                    "parent_id": candidate.parent_id,
                    "band_errors": ";".join(
                        f"{value:.12e}" for value in candidate.band_errors
                    ),
                }
            )
    payload = {
        "generations": result.generations,
        "evaluated_topologies": result.evaluated_topologies,
        "ranking": [
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "topology": list(candidate.topology),
                "thickness_nm": [
                    thickness * 1e9 for _, thickness in candidate.layers
                ],
                "cost": candidate.cost,
                "feasible": candidate.feasible,
                "max_violation": candidate.max_violation,
                "total_thickness_nm": candidate.total_thickness * 1e9,
                "pareto_rank": candidate.pareto_rank,
                "band_errors": candidate.band_errors,
                "generation": candidate.generation,
                "operation": candidate.operation,
                "parent_id": candidate.parent_id,
            }
            for rank, candidate in enumerate(result.ranking, 1)
        ],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return csv_path, json_path
