"""Compute reflectance / transmittance from a JSON stack and plot the spectra.

Usage::

    python3 sim/plot_rt.py sim/examples/example_plot_rt.json
    python3 sim/plot_rt.py sim/out/optimize_example/stack_optimised.json \\
        --bands-from sim/examples/example_vis_pass_ir_reflect.json

Input JSON fields:
  - layers: [{material, thickness_nm}, ...]  (required unless ``seed`` is set)
  - bands:  [{wavelength_nm: [lo, hi], ...}, ...]  (optional; shaded on plot)
  - plot_wavelength_nm, plot_step_nm, incident_angle_deg, substrate, ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dispersion as dsp
from lm_optimizer import BandSpec
from rt_calculator import make_calculator, material_index

_NM = 1e-9

# Soft band shading (low-chroma, easy on the eyes).
BAND_BG_COLORS = (
    "#d9e8f5",
    "#f5e6d4",
    "#ddebd8",
    "#e8dff0",
    "#f3ecd4",
    "#f0dde3",
    "#d8ebe7",
    "#e8e4dc",
)

# Spectrum curve colors (muted, colorblind-friendlier).
COLOR_R = "#3d6f9c"  # soft slate blue
COLOR_T = "#c17a3a"  # soft amber
COLOR_A = "#6a8f6a"  # sage green

# Layer bar / legend fills: same soft blue / warm yellow as band windows.
# One stable color per material so stack bars and legends always match.
_LAYER_FACE_COLORS = {
    "tio2": BAND_BG_COLORS[0],
    "tio2_pvd": BAND_BG_COLORS[0],
    "tio2_eb": "#cfe0f0",
    "tio2_amorphous": "#e0eef8",
    "tio2_a": BAND_BG_COLORS[0],
    "tio2_rutile": "#c5d8ea",
    "sio2": BAND_BG_COLORS[1],
    "sio2_fused": BAND_BG_COLORS[1],
    "sio2_pvd": "#efdcc8",
    "ito": BAND_BG_COLORS[2],
    "ag": BAND_BG_COLORS[3],
    "glass": BAND_BG_COLORS[7],
    "pet": BAND_BG_COLORS[4],
    "air": "#f7f7f5",
}
_LAYER_FALLBACK = BAND_BG_COLORS

# Stronger strokes for n/k curves (pastel bar fills are too light as lines).
_LAYER_LINE_COLORS = {
    "tio2": "#3d6f9c",
    "tio2_pvd": "#3d6f9c",
    "tio2_eb": "#355f88",
    "tio2_amorphous": "#4a7ca8",
    "tio2_a": "#3d6f9c",
    "tio2_rutile": "#2f587c",
    "sio2": "#c17a3a",
    "sio2_fused": "#c17a3a",
    "sio2_pvd": "#a86830",
    "ito": "#5f7f58",
    "ag": "#7a5f84",
    "glass": "#6e6558",
    "pet": "#8a7a48",
    "air": "#8a8a8a",
}
_LAYER_LINE_FALLBACK = (
    "#3d6f9c",
    "#c17a3a",
    "#5f7f58",
    "#7a5f84",
    "#8a7a48",
    "#8a5f5f",
    "#4f7a72",
    "#6e6558",
)

# Typography — keep readable on saved PNGs (dpi=150).
FONT_TITLE = 15
FONT_LABEL = 13
FONT_TICK = 12
FONT_LEGEND = 12
FONT_LAYER = FONT_TICK  # stack thickness labels match R/T legend/tick size
FONT_LAYER_SMALL = FONT_TICK
LINEWIDTH = 2.15
# Fraction of total thickness below which the number is drawn outside with an arrow.
_STACK_OUTSIDE_FRAC = 0.045

_MPL_CONFIGURED = False


def configure_matplotlib(*, backend: str = "Agg") -> None:
    """Force a non-GUI backend before pyplot loads.

    Saving PNGs never needs Qt/Tk. Interactive backends leave
    ``QThreadStorage: entry N destroyed before end of thread`` noise on
    process exit (CPU and GPU alike; CUDA teardown can make it more visible).
    """
    global _MPL_CONFIGURED
    import os

    # Prefer Agg unless the user explicitly set MPLBACKEND.
    env_backend = os.environ.get("MPLBACKEND")
    chosen = env_backend or backend
    import matplotlib as mpl

    current = ""
    try:
        current = str(mpl.get_backend())
    except Exception:
        current = ""
    # Switch away from Qt/Tk/macOS GUI backends used by default on many installs.
    gui_markers = ("qt", "tk", "macosx", "gtk", "wx", "webagg", "nbagg")
    needs_switch = any(m in current.lower() for m in gui_markers) or not current
    if needs_switch or not _MPL_CONFIGURED:
        try:
            mpl.use(chosen, force=True)
        except Exception:
            try:
                mpl.use("Agg", force=True)
            except Exception:
                pass
    _MPL_CONFIGURED = True


def apply_plot_style() -> None:
    """Global matplotlib defaults for softer, readable figures."""
    configure_matplotlib()
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": FONT_TICK,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfaf8",
            "axes.edgecolor": "#6e6e6e",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#bdbdbd",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def close_all_figures() -> None:
    """Drop figure state so Qt/Agg thread locals tear down cleanly."""
    try:
        configure_matplotlib()
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def band_bg_color(index: int) -> str:
    return BAND_BG_COLORS[index % len(BAND_BG_COLORS)]


def shade_bands(ax, bands: list[BandSpec]) -> None:
    for i, b in enumerate(bands):
        ax.axvspan(
            b.wl_lo / _NM,
            b.wl_hi / _NM,
            color=band_bg_color(i),
            alpha=0.55,
            lw=0,
            zorder=0,
        )
    edges_nm = sorted(
        {b.wl_lo / _NM for b in bands} | {b.wl_hi / _NM for b in bands}
    )
    for x in edges_nm:
        ax.axvline(
            x,
            color="#8a8a8a",
            linestyle="--",
            lw=0.9,
            alpha=0.55,
            zorder=1,
        )


def style_axes(ax, *, title: str | None = None, xlabel: str | None = None,
               ylabel: str | None = None) -> None:
    """Apply shared font sizes, grid, and spine styling."""
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE, pad=10, color="#222222")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL, labelpad=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL, labelpad=6)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.35, color="#bdbdbd", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#6e6e6e")
        spine.set_linewidth(0.9)


def style_percent_yaxis(ax) -> None:
    """R/T axis: labels at 0/20/…/100; unlabeled dashed lines every 10%."""
    ax.set_ylim(-2, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks(list(range(10, 100, 10)), minor=True)
    ax.tick_params(axis="y", which="minor", labelleft=False, length=0)
    ax.yaxis.grid(False)
    for y in range(0, 101, 10):
        ax.axhline(
            y,
            color="#9a9a9a",
            linestyle="--",
            lw=0.75,
            alpha=0.5,
            zorder=0,
        )


def annotate_band_means(
    ax,
    wavelengths_m: Sequence[float],
    R: Sequence[float],
    T: Sequence[float] | None,
    bands: list[BandSpec],
    *,
    R_before: Sequence[float] | None = None,
    T_before: Sequence[float] | None = None,
) -> None:
    """Label each wavelength window with mean R (and T) in percent."""
    if not bands:
        return
    after = band_stats(list(wavelengths_m), list(R), list(T or [0.0] * len(R)), bands)
    before = (
        band_stats(list(wavelengths_m), list(R_before), list(T_before or [0.0] * len(R)), bands)
        if R_before is not None
        else None
    )
    show_t = T is not None
    for i, row in enumerate(after):
        lo, hi = row["wl_nm"]
        x = 0.5 * (lo + hi)
        if before is not None and i < len(before):
            lines = [
                f"R̄ {100 * before[i]['R_mean']:.1f}→{100 * row['R_mean']:.1f}%"
            ]
            if show_t:
                lines.append(
                    f"T̄ {100 * before[i]['T_mean']:.1f}→{100 * row['T_mean']:.1f}%"
                )
        else:
            lines = [f"R̄={100 * row['R_mean']:.1f}%"]
            if show_t:
                lines.append(f"T̄={100 * row['T_mean']:.1f}%")
        ax.text(
            x,
            102,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=10,
            color="#2a2a2a",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#c8c8c8",
                "linewidth": 0.6,
                "alpha": 0.88,
            },
            zorder=6,
        )


def contrast_text_color(hex_color: str) -> str:
    """Pick dark or light label color for readable text on ``hex_color``."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "#1a1a1a"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # Relative luminance (sRGB approx).
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#1a1a1a" if luma > 0.55 else "#ffffff"


def load_input(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_bands(raw: list[dict]) -> list[BandSpec]:
    bands = []
    for b in raw:
        w = b["wavelength_nm"]
        if len(w) != 2:
            raise ValueError("each band needs wavelength_nm: [lo, hi]")
        bands.append(
            BandSpec(
                wl_lo=float(w[0]) * _NM,
                wl_hi=float(w[1]) * _NM,
                R_min=b.get("R_min"),
                R_max=b.get("R_max"),
                T_min=b.get("T_min"),
                T_max=b.get("T_max"),
                weight=float(b.get("weight", 1.0)),
                R_target=b.get("R_target"),
                T_target=b.get("T_target"),
            )
        )
    return bands


def parse_layers(raw: list[dict]) -> list[tuple[str, float]]:
    layers = []
    for layer in raw:
        mat = dsp.normalize_material_name(layer["material"])
        if "thickness_nm" in layer:
            d = float(layer["thickness_nm"]) * _NM
        elif "thickness_m" in layer:
            d = float(layer["thickness_m"])
        else:
            raise ValueError("layer needs thickness_nm or thickness_m")
        if mat not in dsp.MATERIALS:
            raise KeyError(
                f"material '{mat}' not in library; known: {sorted(dsp.MATERIALS)}"
            )
        layers.append((mat, d))
    return layers


def build_chirped_seed(seed: dict) -> list[tuple[str, float]]:
    centres = [float(c) * _NM for c in seed.get("centres_nm", [900, 1200, 1500])]
    raw_p = seed.get("periods_per_centre", 3)
    if isinstance(raw_p, list):
        if len(raw_p) != len(centres):
            raise ValueError("periods_per_centre list must match centres_nm")
        periods_list = [int(p) for p in raw_p]
    else:
        periods_list = [int(raw_p)] * len(centres)
    cell = [
        dsp.normalize_material_name(m) for m in seed.get("cell", ["tio2", "sio2"])
    ]
    layers: list[tuple[str, float]] = []
    for lam0, periods in zip(centres, periods_list):
        for _ in range(periods):
            for mat in cell:
                if mat not in dsp.MATERIALS:
                    raise KeyError(f"seed material '{mat}' not in library")
                n = dsp.material_n(mat, lam0).real
                layers.append((mat, 0.25 * lam0 / max(n, 1.01)))
    return layers


def plot_range_nm(cfg: dict, bands: list[BandSpec]) -> tuple[float, float]:
    if "plot_wavelength_nm" in cfg:
        lo, hi = cfg["plot_wavelength_nm"]
        return float(lo), float(hi)
    lo = min(b.wl_lo for b in bands) / _NM
    hi = max(b.wl_hi for b in bands) / _NM
    return lo, hi


def dense_grid_nm(lo_nm: float, hi_nm: float, step_nm: float) -> list[float]:
    n = max(1, int(round((hi_nm - lo_nm) / step_nm)))
    return [lo_nm + i * (hi_nm - lo_nm) / n for i in range(n + 1)]


# Distinct fills for common coating materials on stack-thickness bars.
# (Canonical map lives above as _LAYER_FACE_COLORS / _LAYER_FALLBACK.)


def layer_face_color(material: str, index: int = 0) -> str:
    """Return the fill color for a material.

    Color depends only on the material identity so stack bars and legends
    stay consistent. ``index`` is accepted for call-site compatibility but
    ignored.
    """
    del index  # color must not depend on layer position
    key = dsp.normalize_material_name(material)
    if key in _LAYER_FACE_COLORS:
        return _LAYER_FACE_COLORS[key]
    digest = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    return _LAYER_FALLBACK[digest % len(_LAYER_FALLBACK)]


def material_line_color(material: str, index: int = 0) -> str:
    """Stroke color for n/k curves; keyed like stack fills but darker."""
    del index
    key = dsp.normalize_material_name(material)
    if key in _LAYER_LINE_COLORS:
        return _LAYER_LINE_COLORS[key]
    digest = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    return _LAYER_LINE_FALLBACK[digest % len(_LAYER_LINE_FALLBACK)]


def _format_stack_thickness(width_nm: float) -> str:
    """Thickness label without units."""
    if width_nm >= 100.0:
        return f"{width_nm:.0f}"
    if width_nm >= 10.0:
        return f"{width_nm:.1f}"
    return f"{width_nm:.2f}"


def draw_stack_thickness(
    ax,
    layers: list[tuple[str, float]],
    *,
    label: str = "",
    y: float = 0.0,
    height: float = 0.78,
    show_values: bool = True,
) -> float:
    """Draw one horizontal stacked bar of physical thicknesses (nm).

    Thickness numbers use the same font size as R/T labels. Thin layers are
    labeled outside the bar with an arrow. Returns total thickness in nm.
    """
    if not layers:
        ax.text(
            0.5,
            y,
            "(no layers)",
            ha="center",
            va="center",
            fontsize=FONT_LABEL,
            transform=ax.get_yaxis_transform(),
        )
        return 0.0

    total_nm = sum(d for _, d in layers) / _NM
    x = 0.0
    outside_index = 0
    for mat, d in layers:
        w = d / _NM
        color = layer_face_color(mat)
        ax.barh(
            y,
            w,
            left=x,
            height=height,
            color=color,
            edgecolor="#5a5a5a",
            linewidth=0.7,
            align="center",
        )
        if show_values and total_nm > 0 and w > 0.0:
            text = _format_stack_thickness(w)
            frac = w / total_nm
            center_x = x + 0.5 * w
            if frac >= _STACK_OUTSIDE_FRAC:
                ax.text(
                    center_x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=FONT_TICK,
                    color=contrast_text_color(color),
                    clip_on=True,
                )
            else:
                # Point straight to the top/bottom edge of the bar so the
                # arrow never cuts through the layer interior.
                side = 1 if outside_index % 2 == 0 else -1
                lane = outside_index // 2
                y_edge = y + side * (0.5 * height)
                y_text = y_edge + side * (0.22 + 0.18 * lane)
                ax.annotate(
                    text,
                    xy=(center_x, y_edge),
                    xytext=(center_x, y_text),
                    textcoords="data",
                    ha="center",
                    va="bottom" if side > 0 else "top",
                    fontsize=FONT_TICK,
                    color="#1a1a1a",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#555555",
                        "lw": 0.85,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    annotation_clip=False,
                    zorder=5,
                )
                outside_index += 1
        x += w

    if label:
        ax.text(
            -0.012 * max(total_nm, 1.0),
            y,
            label,
            ha="right",
            va="center",
            fontsize=FONT_LABEL,
            fontweight="bold",
            color="#2a2a2a",
        )
    return total_nm


def materials_used_in_stack(
    incident: str,
    layers: Sequence[tuple[str, float]] | None,
    substrate: str,
) -> list[str]:
    """Unique materials in stack order (incident → coating → substrate)."""
    names: list[str] = []
    for name in (incident, *[m for m, _ in (layers or ())], substrate):
        key = dsp.normalize_material_name(name)
        if key not in names:
            names.append(key)
    return names


def _lookup_fixed_nk(fixed_nk: dict[str, complex], name: str) -> complex:
    key = dsp.normalize_material_name(name)
    low = str(name).strip().lower()
    for cand in (key, low, name):
        if cand in fixed_nk:
            return complex(fixed_nk[cand])
        cl = str(cand).lower()
        if cl in fixed_nk:
            return complex(fixed_nk[cl])
    raise KeyError(
        f"material {name!r} missing from fixed n,k table; "
        f"have: {sorted(fixed_nk)}"
    )


def sample_nk_curves(
    materials: Sequence[str],
    wavelengths_m: Sequence[float],
    *,
    fixed_nk: dict[str, complex] | None = None,
) -> dict[str, list[complex]]:
    """n(λ)+ik(λ) actually used: library dispersion or constant fixed_nk."""
    out: dict[str, list[complex]] = {}
    for name in materials:
        key = dsp.normalize_material_name(name)
        if fixed_nk is not None:
            N = _lookup_fixed_nk(fixed_nk, name)
            out[key] = [N] * len(wavelengths_m)
        else:
            out[key] = [dsp.material_n(key, wl) for wl in wavelengths_m]
    return out


def write_nk_csv(
    path: str,
    wavelengths_m: Sequence[float],
    nk_by_material: dict[str, list[complex]],
) -> None:
    """Write wavelength + per-material n,k columns."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mats = list(nk_by_material.keys())
    with open(path, "w", encoding="utf-8") as fh:
        header = ["wavelength_nm"]
        for m in mats:
            header += [f"{m}_n", f"{m}_k"]
        fh.write(",".join(header) + "\n")
        for i, wl in enumerate(wavelengths_m):
            row = [f"{wl / _NM:.2f}"]
            for m in mats:
                N = nk_by_material[m][i]
                row += [f"{N.real:.6f}", f"{N.imag:.6f}"]
            fh.write(",".join(row) + "\n")


def plot_used_nk(
    path: str,
    wavelengths_m: Sequence[float],
    nk_by_material: dict[str, list[complex]],
    *,
    source_label: str = "library",
    title: str | None = None,
) -> None:
    """Plot n(λ) and k(λ) for materials actually used in the TMM stack."""
    try:
        apply_plot_style()
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; pip install matplotlib"
        ) from exc

    if not nk_by_material:
        return

    wl_nm = [w / _NM for w in wavelengths_m]
    mats = list(nk_by_material.keys())

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
    for i, name in enumerate(mats):
        color = material_line_color(name, i)
        curve = nk_by_material[name]
        axes[0].plot(
            wl_nm,
            [z.real for z in curve],
            color=color,
            lw=LINEWIDTH,
            label=name,
        )
        axes[1].plot(
            wl_nm,
            [z.imag for z in curve],
            color=color,
            lw=LINEWIDTH,
            label=name,
        )

    hdr = title or f"Optical constants used in TMM  (nk_source={source_label})"
    style_axes(axes[0], title=hdr, ylabel="n")
    style_axes(axes[1], xlabel="Wavelength (nm)", ylabel="k")
    axes[0].legend(
        loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8"
    )
    axes[1].legend(
        loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8"
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    close_all_figures()


def plot_nk_panel(
    ax,
    wavelengths_m: Sequence[float],
    nk_by_material: dict[str, list[complex]],
    *,
    quantity: str = "n",
    source_label: str = "library",
) -> None:
    """Draw n or k vs wavelength for used materials on an existing axes."""
    wl_nm = [w / _NM for w in wavelengths_m]
    is_k = quantity.lower() == "k"
    for i, (name, curve) in enumerate(nk_by_material.items()):
        ys = [z.imag if is_k else z.real for z in curve]
        ax.plot(
            wl_nm,
            ys,
            color=material_line_color(name, i),
            lw=LINEWIDTH,
            label=name,
        )
    style_axes(
        ax,
        title=f"{'Extinction k' if is_k else 'Refractive index n'} "
        f"(used; nk_source={source_label})",
        xlabel="Wavelength (nm)",
        ylabel="k" if is_k else "n",
    )
    ax.legend(
        loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8"
    )


def format_layers_caption(
    layers: list[tuple[str, float]],
    *,
    max_lines: int | None = None,
) -> str:
    """Multi-line caption: ``i. material  d.dd nm`` plus total thickness."""
    if not layers:
        return ""
    total_nm = sum(d for _, d in layers) / _NM
    lines = [f"layers Σ={total_nm:.1f} nm"]
    show = layers if max_lines is None else layers[:max_lines]
    for i, (mat, d) in enumerate(show, 1):
        lines.append(f"{i:2d}. {mat:<8} {d / _NM:7.2f} nm")
    if max_lines is not None and len(layers) > max_lines:
        lines.append(f"  … +{len(layers) - max_lines} more")
    return "\n".join(lines)


def _layers_list_text(layers: list[tuple[str, float]]) -> str:
    """Compact readable layer list for under the thickness bar."""
    parts = [f"{i}.{mat} {d / _NM:.1f}" for i, (mat, d) in enumerate(layers, 1)]
    # Wrap every 4 entries so lines stay on-canvas.
    chunk = 4 if len(parts) > 4 else len(parts)
    lines = []
    for i in range(0, len(parts), chunk):
        lines.append("  ·  ".join(parts[i : i + chunk]))
    return "\n".join(lines)


def plot_stack_panel(
    ax,
    layers: list[tuple[str, float]] | None = None,
    layers_before: list[tuple[str, float]] | None = None,
    layers_after: list[tuple[str, float]] | None = None,
) -> None:
    """Bottom / side panel: stacked thickness bar(s) with total Σ."""
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    if layers_before is not None:
        rows.append(("before", layers_before))
    if layers_after is not None:
        rows.append(("after", layers_after))
    if not rows and layers is not None:
        rows.append(("stack", layers))

    if not rows:
        ax.set_axis_off()
        return

    totals: list[float] = []
    for i, (label, lyrs) in enumerate(rows):
        y = float(len(rows) - 1 - i)
        totals.append(
            draw_stack_thickness(
                ax, lyrs, label=label, y=y, height=0.7, show_values=True
            )
        )

    xmax = max(totals) if totals else 1.0
    ax.set_xlim(0.0, xmax * 1.02 if xmax > 0 else 1.0)
    # Extra vertical room for outside thickness arrows on thin layers.
    ax.set_ylim(-1.45, len(rows) + 0.15)
    ax.set_yticks([])

    parts = []
    for label, lyrs in rows:
        tot = sum(d for _, d in lyrs) / _NM
        parts.append(f"{label}: {len(lyrs)} lyrs, Σ={tot:.1f} nm")
    style_axes(
        ax,
        title="Layer thicknesses  (" + "; ".join(parts) + ")",
        xlabel="Layer thickness (nm)",
    )
    ax.grid(True, axis="x", alpha=0.28, color="#9a9a9a", linewidth=0.8)
    ax.grid(False, axis="y")

    # Legend outside the axes so it never covers thickness numbers.
    seen: dict[str, tuple[str, str]] = {}
    for _, lyrs in rows:
        for mat, _ in lyrs:
            key = dsp.normalize_material_name(mat)
            if key not in seen:
                seen[key] = (str(mat), layer_face_color(mat))
    if seen:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=color, edgecolor="#5a5a5a", label=label)
            for label, color in seen.values()
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=FONT_LEGEND,
            framealpha=0.95,
            edgecolor="#c8c8c8",
            ncol=1,
        )


def plot_results(
    path: str,
    wavelengths_m: list[float],
    R_before: list[float],
    T_before: list[float],
    R_after: list[float],
    T_after: list[float],
    bands: list[BandSpec],
    materials_to_show: list[str] | None = None,
    layers_before: list[tuple[str, float]] | None = None,
    layers_after: list[tuple[str, float]] | None = None,
    nk_by_material: dict[str, list[complex]] | None = None,
    nk_source_label: str = "library",
) -> None:
    try:
        apply_plot_style()
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; pip install matplotlib"
        ) from exc

    wl_nm = [w / _NM for w in wavelengths_m]
    materials_to_show = materials_to_show or [
        "sio2",
        "tio2",
        "tio2_a",
        "tio2_rutile",
        "glass",
        "air",
    ]
    has_stack = layers_before is not None or layers_after is not None
    has_nk = bool(nk_by_material)
    n_rows = 2 + (1 if has_stack else 0) + (2 if has_nk else (0 if has_stack else 1))
    # Layout:
    #   always: RT overlay, R
    #   + stack panel if layers given
    #   + n & k panels if nk_by_material given
    #   else if no stack: legacy library-n panel from materials_to_show
    fig_h = 4.0 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10.5, min(fig_h, 18.0)),
        sharex=False,
    )
    if n_rows == 1:
        axes = [axes]
    else:
        axes = list(axes)
    axes[1].sharex(axes[0])

    ax = axes[0]
    ax.plot(
        wl_nm,
        [100 * r for r in R_before],
        "--",
        color=COLOR_R,
        lw=LINEWIDTH,
        alpha=0.75,
        label="R before",
    )
    ax.plot(
        wl_nm,
        [100 * r for r in R_after],
        "-",
        color=COLOR_R,
        lw=LINEWIDTH,
        label="R after",
    )
    ax.plot(
        wl_nm,
        [100 * t for t in T_before],
        "--",
        color=COLOR_T,
        lw=LINEWIDTH,
        alpha=0.75,
        label="T before",
    )
    ax.plot(
        wl_nm,
        [100 * t for t in T_after],
        "-",
        color=COLOR_T,
        lw=LINEWIDTH,
        label="T after",
    )
    shade_bands(ax, bands)
    for b in bands:
        if b.R_min is not None:
            ax.hlines(
                100 * b.R_min,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors=COLOR_R,
                linestyles=":",
                lw=1.2,
            )
        if b.R_max is not None:
            ax.hlines(
                100 * b.R_max,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors=COLOR_R,
                linestyles=":",
                lw=1.2,
            )
        if b.T_min is not None:
            ax.hlines(
                100 * b.T_min,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors=COLOR_T,
                linestyles=":",
                lw=1.2,
            )
        if b.T_max is not None:
            ax.hlines(
                100 * b.T_max,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors=COLOR_T,
                linestyles=":",
                lw=1.2,
            )
    style_axes(
        ax,
        title="Reflectance & transmittance before / after optimisation",
        ylabel="R, T (%)",
    )
    style_percent_yaxis(ax)
    annotate_band_means(
        ax,
        wavelengths_m,
        R_after,
        T_after,
        bands,
        R_before=R_before,
        T_before=T_before,
    )
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8")

    ax = axes[1]
    ax.plot(
        wl_nm,
        [100 * r for r in R_before],
        "--",
        color=COLOR_R,
        lw=LINEWIDTH,
        alpha=0.75,
        label="R before",
    )
    ax.plot(
        wl_nm,
        [100 * r for r in R_after],
        "-",
        color=COLOR_R,
        lw=LINEWIDTH,
        label="R after",
    )
    shade_bands(ax, bands)
    if has_stack or has_nk:
        style_axes(ax, title="Reflectance", ylabel="R (%)")
    else:
        style_axes(ax, title="Reflectance", xlabel="Wavelength (nm)", ylabel="R (%)")
    style_percent_yaxis(ax)
    annotate_band_means(
        ax,
        wavelengths_m,
        R_after,
        None,
        bands,
        R_before=R_before,
    )
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8")

    row = 2
    if has_stack:
        plot_stack_panel(
            axes[row],
            layers_before=layers_before,
            layers_after=layers_after,
        )
        row += 1

    if has_nk:
        assert nk_by_material is not None
        axes[row].sharex(axes[0])
        plot_nk_panel(
            axes[row],
            wavelengths_m,
            nk_by_material,
            quantity="n",
            source_label=nk_source_label,
        )
        row += 1
        axes[row].sharex(axes[0])
        plot_nk_panel(
            axes[row],
            wavelengths_m,
            nk_by_material,
            quantity="k",
            source_label=nk_source_label,
        )
    elif not has_stack:
        ax = axes[row]
        ax.sharex(axes[0])
        mat_colors = (COLOR_R, COLOR_T, COLOR_A, "#7a6a9a", "#8a7a5a")
        for i, name in enumerate(materials_to_show):
            if dsp.normalize_material_name(name) not in dsp.MATERIALS:
                continue
            nk = material_index(name, wavelengths_m)
            ax.plot(
                wl_nm,
                [z.real for z in nk],
                color=mat_colors[i % len(mat_colors)],
                lw=LINEWIDTH,
                label=f"n({name})",
            )
        style_axes(
            ax,
            title="Material refractive index (library)",
            xlabel="Wavelength (nm)",
            ylabel="n",
        )
        ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    close_all_figures()


def band_stats(
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    bands: list[BandSpec],
) -> list[dict[str, Any]]:
    """Mean / min / max of R and T inside each band."""
    rows = []
    for b in bands:
        idx = [
            i
            for i, wl in enumerate(wavelengths_m)
            if b.wl_lo <= wl <= b.wl_hi
        ]
        if not idx:
            continue
        rr = [R[i] for i in idx]
        tt = [T[i] for i in idx]
        rows.append(
            {
                "wl_nm": (b.wl_lo / _NM, b.wl_hi / _NM),
                "R_mean": sum(rr) / len(rr),
                "R_min": min(rr),
                "R_max": max(rr),
                "T_mean": sum(tt) / len(tt),
                "T_min": min(tt),
                "T_max": max(tt),
            }
        )
    return rows


def write_band_stats_csv(
    path: str,
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    bands: list[BandSpec],
) -> None:
    """Write per-band R/T mean/min/max table as CSV."""
    rows = band_stats(wavelengths_m, R, T, bands)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "band,wl_lo_nm,wl_hi_nm,"
            "R_mean,R_min,R_max,T_mean,T_min,T_max\n"
        )
        for i, row in enumerate(rows, 1):
            lo, hi = row["wl_nm"]
            fh.write(
                f"{i},{lo:.2f},{hi:.2f},"
                f"{row['R_mean']:.6f},{row['R_min']:.6f},{row['R_max']:.6f},"
                f"{row['T_mean']:.6f},{row['T_min']:.6f},{row['T_max']:.6f}\n"
            )


def write_spectrum_csv(
    path: str,
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("wavelength_nm,R,T,A\n")
        for wl, r, t in zip(wavelengths_m, R, T):
            a = max(0.0, 1.0 - r - t)
            fh.write(f"{wl / _NM:.2f},{r:.6f},{t:.6f},{a:.6f}\n")


def plot_rt(
    path: str,
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    bands: list[BandSpec],
    title: str = "Reflectance & transmittance",
    layers: list[tuple[str, float]] | None = None,
) -> None:
    try:
        apply_plot_style()
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; pip install matplotlib"
        ) from exc

    wl_nm = [w / _NM for w in wavelengths_m]
    if layers:
        fig = plt.figure(figsize=(10.5, 11.4))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.62])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2])
        axes = [ax0, ax1, ax2]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
        axes = list(axes)

    ax = axes[0]
    ax.plot(wl_nm, [100 * r for r in R], color=COLOR_R, lw=LINEWIDTH, label="R")
    ax.plot(wl_nm, [100 * t for t in T], color=COLOR_T, lw=LINEWIDTH, label="T")
    ax.plot(
        wl_nm,
        [100 * max(0.0, 1.0 - r - t) for r, t in zip(R, T)],
        color=COLOR_A,
        lw=LINEWIDTH,
        label="A ≈ 1−R−T",
        alpha=0.9,
    )
    shade_bands(ax, bands)
    for b in bands:
        for val, color in (
            (b.R_min, COLOR_R),
            (b.R_max, COLOR_R),
            (b.T_min, COLOR_T),
            (b.T_max, COLOR_T),
        ):
            if val is not None:
                ax.hlines(
                    100 * val,
                    b.wl_lo / _NM,
                    b.wl_hi / _NM,
                    colors=color,
                    linestyles=":",
                    lw=1.2,
                )
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8")
    style_axes(ax, title=title, ylabel="R, T, A (%)")
    style_percent_yaxis(ax)
    annotate_band_means(ax, wavelengths_m, R, T, bands)

    ax = axes[1]
    ax.plot(wl_nm, [100 * r for r in R], color=COLOR_R, lw=LINEWIDTH, label="R")
    ax.plot(wl_nm, [100 * t for t in T], color=COLOR_T, lw=LINEWIDTH, label="T")
    shade_bands(ax, bands)
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.92, edgecolor="#c8c8c8")
    style_axes(ax, xlabel="Wavelength (nm)", ylabel="R, T (%)")
    style_percent_yaxis(ax)
    annotate_band_means(ax, wavelengths_m, R, T, bands)

    if layers:
        plot_stack_panel(axes[2], layers=layers)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    close_all_figures()


def resolve_layers(cfg: dict) -> list[tuple[str, float]]:
    if cfg.get("layers"):
        return parse_layers(cfg["layers"])
    if cfg.get("seed"):
        return build_chirped_seed(cfg["seed"])
    raise ValueError("input needs 'layers' (list) or 'seed'")


def run(cfg: dict, input_path: str, bands_cfg: dict | None = None) -> int:
    src = bands_cfg if bands_cfg is not None else cfg
    bands = parse_bands(src["bands"]) if src.get("bands") else []
    layers = resolve_layers(cfg)

    angle_deg = float(cfg.get("incident_angle_deg", 0.0))
    theta0 = math.radians(angle_deg)
    substrate_model = cfg.get("substrate_model", "semi_infinite")

    out_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "plot_rt"),
    )
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(input_path)), out_dir)
        )

    if bands:
        lo_nm, hi_nm = plot_range_nm(cfg if "plot_wavelength_nm" in cfg else src, bands)
    else:
        lo_nm, hi_nm = cfg.get("plot_wavelength_nm", [400, 1800])
        lo_nm, hi_nm = float(lo_nm), float(hi_nm)
    plot_step = float(cfg.get("plot_step_nm", src.get("plot_step_nm", 5)))
    plot_wls = [x * _NM for x in dense_grid_nm(lo_nm, hi_nm, plot_step)]

    calc = make_calculator(
        cfg.get("rt_engine", "tmm"),
        cfg.get("external_command"),
        use_cuda=bool(cfg.get("use_cuda", False)),
    )
    rt_kw = dict(
        incident=cfg.get("incident_medium", "air"),
        substrate=cfg.get("substrate", "glass"),
        substrate_thickness=float(cfg.get("substrate_thickness_m", 0.7e-3)),
        exit_medium=cfg.get("exit_medium", "air"),
        polarization=cfg.get("polarization", "unpolarized"),
        substrate_model=substrate_model,
    )
    R, T = calc.spectrum(layers, plot_wls, theta0, **rt_kw)

    total_nm = sum(d for _, d in layers) / _NM
    print("Film R/T spectrum")
    print(f"  input: {input_path}")
    print(f"  angle: {angle_deg} deg  engine: {cfg.get('rt_engine', 'tmm')}")
    print(f"  substrate_model: {substrate_model}")
    print(f"  {len(layers)} layers, total thin-film thickness {total_nm:.1f} nm")
    for i, (m, d) in enumerate(layers, 1):
        print(f"    {i:2d}. {m:<8} {d / _NM:8.2f} nm")

    for row in band_stats(plot_wls, R, T, bands):
        lo, hi = row["wl_nm"]
        print(
            f"  {lo:.0f}-{hi:.0f} nm: "
            f"R mean/min/max = {100*row['R_mean']:.1f}/"
            f"{100*row['R_min']:.1f}/{100*row['R_max']:.1f}%  "
            f"T mean/min/max = {100*row['T_mean']:.1f}/"
            f"{100*row['T_min']:.1f}/{100*row['T_max']:.1f}%"
        )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum.csv")
    plot_path = os.path.join(out_dir, "rt_spectrum.png")
    write_spectrum_csv(csv_path, plot_wls, R, T)
    plot_rt(
        plot_path,
        plot_wls,
        R,
        T,
        bands,
        title=os.path.basename(input_path),
        layers=layers,
    )
    print(f"\n  wrote {csv_path}")
    print(f"  wrote {plot_path}")
    close_all_figures()
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_matplotlib()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Read a layer stack (+ optional bands) from JSON, "
        "compute R/T with TMM, and plot with matplotlib."
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=os.path.join(here, "examples", "example_plot_rt.json"),
        help="JSON with layers (or seed) and optional bands",
    )
    ap.add_argument(
        "--bands-from",
        default=None,
        help="optional second JSON that supplies the bands list "
        "(useful when plotting stack_optimised.json)",
    )
    args = ap.parse_args(argv)
    cfg = load_input(args.input)
    bands_cfg = load_input(args.bands_from) if args.bands_from else None
    return run(cfg, args.input, bands_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
