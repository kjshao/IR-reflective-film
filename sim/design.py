"""Design and evaluate IR-reflective stacks: visible pass, 700-1300 nm stop.

Produces the numbers quoted in docs/oghmanano-ir-film-simulation.md, the layer
tables to type into OghmaNano's layer editor, and the reference spectra to
compare against OghmaNano's reflect.csv / transmit.csv.

    python3 sim/design.py            # evaluate the reference designs
    python3 sim/design.py --optimise # refine thicknesses by coordinate descent
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dispersion as dsp
import tmm

VIS = (400e-9, 700e-9)
NIR = (700e-9, 1300e-9)
# The 700 nm boundary is a spec edge, not a physical one: every real filter
# needs a finite transition width, so the deep band is reported separately to
# keep the pass/stop transition from masking the actual stop-band depth.
DEEP_NIR = (800e-9, 1300e-9)
GRID_STEP = 2e-9
SUBSTRATE_THICKNESS = 0.7e-3
ANGLES = (0.0, 30.0, 45.0, 60.0)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def grid(lo, hi, step=GRID_STEP):
    n = int(round((hi - lo) / step)) + 1
    return [lo + i * step for i in range(n)]


FULL_GRID = grid(380e-9, 1400e-9)
VIS_GRID = [w for w in FULL_GRID if VIS[0] <= w <= VIS[1]]
NIR_GRID = [w for w in FULL_GRID if NIR[0] <= w <= NIR[1]]
DEEP_GRID = [w for w in FULL_GRID if DEEP_NIR[0] <= w <= DEEP_NIR[1]]


# --------------------------------------------------------------------------
# Stack definitions.  A stack is a list of (material name, thickness) pairs
# ordered from the incident side; the substrate is appended separately.
# --------------------------------------------------------------------------

OMOMO = [
    ("ito", 35e-9),
    ("ag", 12e-9),
    ("ito", 85e-9),
    ("ag", 12e-9),
    ("ito", 35e-9),
]

OMO = [
    ("ito", 40e-9),
    ("ag", 14e-9),
    ("ito", 40e-9),
]


# A unit cell is a list of (material, f), where f is the layer's optical
# thickness as a fraction of the block centre wavelength. A Bragg period has
# total optical thickness lambda0/2.
QW_CELL = [("tio2", 0.25), ("sio2", 0.25)]

DEFAULT_CENTRES = (824e-9, 985e-9, 1170e-9)


def chirped_dielectric(centres=DEFAULT_CENTRES, periods=8, cell=None):
    """Chirped TiO2/SiO2 blocks forming a wide near-infrared stop band.

    One quarter-wave stack spans only ~30% fractional bandwidth at this index
    contrast while 700-1300 nm needs ~60%, so several blocks are chirped
    across the band.  The shortest centre wavelength goes on top so visible
    light meets the weakest reflector first.
    """
    cell = cell or QW_CELL
    layers = []
    for lam0 in centres:
        for _ in range(periods):
            for mat, frac in cell:
                n = dsp.MATERIALS[mat](lam0).real
                layers.append((mat, frac * lam0 / n))
    return layers


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def spectrum(layers, substrate="glass", angles=(0.0,), wavelengths=None):
    """R(lambda), T(lambda) for each angle, unpolarised, with substrate back face."""
    wavelengths = wavelengths or FULL_GRID
    sub_fn = dsp.MATERIALS[substrate]
    out = {}
    for deg in angles:
        theta = math.radians(deg)
        rs, ts = [], []
        for wl in wavelengths:
            n_sub = sub_fn(wl)
            front = [(dsp.air(wl), 0.0)]
            front += [(dsp.MATERIALS[m](wl), d) for m, d in layers]
            front.append((n_sub, 0.0))
            r, t = tmm.unpolarised(
                tmm.with_incoherent_substrate,
                front,
                wl,
                theta,
                n_sub=n_sub,
                d_sub=SUBSTRATE_THICKNESS,
                n_exit=dsp.air(wl),
            )
            rs.append(r)
            ts.append(t)
        out[deg] = (wavelengths, rs, ts)
    return out


def _point(layers, wl, theta, sub_fn):
    n_sub = sub_fn(wl)
    front = [(dsp.air(wl), 0.0)]
    front += [(dsp.MATERIALS[m](wl), d) for m, d in layers]
    front.append((n_sub, 0.0))
    return tmm.unpolarised(
        tmm.with_incoherent_substrate,
        front,
        wl,
        theta,
        n_sub=n_sub,
        d_sub=SUBSTRATE_THICKNESS,
        n_exit=dsp.air(wl),
    )


def cutoff_edge(wavelengths, transmittance, level=0.5):
    """Wavelength where T last falls through `level` on its way into the stop band."""
    for i in range(len(wavelengths) - 1):
        t0, t1 = transmittance[i], transmittance[i + 1]
        if t0 >= level > t1:
            f = (t0 - level) / (t0 - t1)
            return wavelengths[i] + f * (wavelengths[i + 1] - wavelengths[i])
    return None


def metrics(layers, substrate="glass", deg=0.0):
    theta = math.radians(deg)
    sub_fn = dsp.MATERIALS[substrate]

    def sweep(wls):
        rr, tt = [], []
        for wl in wls:
            r, t = _point(layers, wl, theta, sub_fn)
            rr.append(r)
            tt.append(t)
        return rr, tt

    r_vis, t_vis = sweep(VIS_GRID)
    r_deep, t_deep = sweep(DEEP_GRID)
    _, t_edge = sweep(FULL_GRID)

    return {
        "T_vis_mean": sum(t_vis) / len(t_vis),
        "T_vis_min": min(t_vis),
        "R_deep_mean": sum(r_deep) / len(r_deep),
        "R_deep_min": min(r_deep),
        "T_deep_max": max(t_deep),
        "A_vis_mean": max(0.0, sum(1.0 - r - t for r, t in zip(r_vis, t_vis)) / len(r_vis)),
        "edge_nm": (cutoff_edge(FULL_GRID, t_edge) or float("nan")) * 1e9,
    }


def merit(layers, substrate="glass"):
    """Lower is better: squared miss from T=1 in the visible, R=1 across 800-1300 nm.

    T_vis_min carries extra weight because the failure mode of a chirped
    quarter-wave stack is a narrow higher-order reflection band punched into
    the visible, which barely moves the band average.
    """
    m = metrics(layers, substrate)
    vis = (1.0 - m["T_vis_mean"]) ** 2 + 1.5 * (1.0 - m["T_vis_min"]) ** 2
    nir = (1.0 - m["R_deep_mean"]) ** 2 + 0.5 * (1.0 - m["R_deep_min"]) ** 2
    return vis + nir


BOUNDS = {
    "ag": (8e-9, 20e-9),
    "ito": (15e-9, 120e-9),
    "tio2": (20e-9, 300e-9),
    "sio2": (20e-9, 320e-9),
}


def optimise(layers, substrate="glass", rounds=6, verbose=True):
    """Bounded coordinate descent on layer thicknesses."""
    best = [list(l) for l in layers]
    best_merit = merit([tuple(l) for l in best], substrate)
    step = 8e-9
    for r in range(rounds):
        improved = False
        for i, (mat, _) in enumerate(best):
            lo, hi = BOUNDS[mat]
            for delta in (step, -step):
                trial = [list(l) for l in best]
                trial[i][1] = min(hi, max(lo, trial[i][1] + delta))
                if abs(trial[i][1] - best[i][1]) < 1e-12:
                    continue
                m = merit([tuple(l) for l in trial], substrate)
                if m < best_merit - 1e-9:
                    best, best_merit, improved = trial, m, True
        if verbose:
            print(f"    round {r + 1}: step {step * 1e9:5.2f} nm  merit {best_merit:.5f}")
        if not improved:
            step /= 2.0
            if step < 0.4e-9:
                break
    return [tuple(l) for l in best], best_merit


# --------------------------------------------------------------------------
# Self-tests: the engine has to reproduce results that are known analytically.
# --------------------------------------------------------------------------


def self_test():
    print("Engine self-tests")
    wl = 550e-9
    n_g = dsp.glass(wl)

    r, t = tmm.coherent_rt([(dsp.air(wl), 0.0), (n_g, 0.0)], wl)
    expect = abs((1.0 - n_g) / (1.0 + n_g)) ** 2
    print(f"  bare glass single surface   R = {r:.5f}  (Fresnel {expect.real:.5f})")
    assert abs(r - expect) < 1e-9 and abs(r + t - 1.0) < 1e-9

    r, t = tmm.with_incoherent_substrate(
        [(dsp.air(wl), 0.0), (n_g, 0.0)], wl, 0.0, "s",
        n_sub=n_g, d_sub=SUBSTRATE_THICKNESS, n_exit=dsp.air(wl),
    )
    print(f"  bare glass slab, two faces  R = {r:.5f}  T = {t:.5f}  (R+T = {r + t:.6f})")
    assert abs(r + t - 1.0) < 1e-9 and abs(t - 0.9200) < 0.002

    n_c = dsp.sio2(wl)
    d_qw = wl / (4.0 * n_c.real)
    r, t = tmm.coherent_rt(
        [(dsp.air(wl), 0.0), (n_c, d_qw), (n_g, 0.0)], wl
    )
    ideal = abs((n_g - n_c**2) / (n_g + n_c**2)) ** 2
    print(f"  quarter-wave SiO2 AR        R = {r:.5f}  (analytic {ideal.real:.5f})")
    assert abs(r - ideal) < 1e-9

    stack = [(dsp.air(wl), 0.0)]
    for _ in range(6):
        stack.append((dsp.tio2(wl), wl / (4 * dsp.tio2(wl).real)))
        stack.append((dsp.sio2(wl), wl / (4 * dsp.sio2(wl).real)))
    stack.append((n_g, 0.0))
    r, t = tmm.coherent_rt(stack, wl)
    print(f"  6-period Bragg at lambda0   R = {r:.5f}  T = {t:.5f}  (R+T = {r + t:.6f})")
    assert abs(r + t - 1.0) < 1e-9 and r > 0.99

    for deg in ANGLES:
        rs, ts = tmm.coherent_rt(stack, wl, math.radians(deg), "s")
        rp, tp = tmm.coherent_rt(stack, wl, math.radians(deg), "p")
        assert abs(rs + ts - 1.0) < 1e-9 and abs(rp + tp - 1.0) < 1e-9
    print("  energy conservation holds for s and p at 0/30/45/60 deg")
    print()


def print_dispersion():
    print("Dispersion models (n + ik)")
    header = f"  {'material':>8} " + " ".join(f"{int(w * 1e9):>16}nm" for w in
                                             (450e-9, 550e-9, 700e-9, 1000e-9, 1300e-9))
    print(header)
    for name in ("ag", "ito", "tio2", "sio2", "glass"):
        row = f"  {name:>8} "
        for w in (450e-9, 550e-9, 700e-9, 1000e-9, 1300e-9):
            n = dsp.MATERIALS[name](w)
            row += f" {n.real:7.3f}+{n.imag:6.3f}i"
        print(row)
    print()


def show(label, layers, substrate="glass"):
    print(f"{label}")
    total = sum(d for _, d in layers) * 1e9
    print(f"  {len(layers)} layers, {total:.0f} nm of thin film on "
          f"{substrate} ({SUBSTRATE_THICKNESS * 1e3:.2f} mm)")
    print(f"  {'angle':>5}  {'T_vis avg':>9} {'T_vis min':>9}  "
          f"{'R 800-1300 avg':>14} {'min':>6} {'T max':>6}  {'A_vis':>5}  {'50% edge':>8}")
    for deg in ANGLES:
        m = metrics(layers, substrate, deg)
        print(f"  {deg:3.0f} deg  {m['T_vis_mean'] * 100:8.1f}% {m['T_vis_min'] * 100:8.1f}%  "
              f"{m['R_deep_mean'] * 100:13.1f}% {m['R_deep_min'] * 100:5.1f}% "
              f"{m['T_deep_max'] * 100:5.1f}%  {m['A_vis_mean'] * 100:4.1f}%  "
              f"{m['edge_nm']:6.0f} nm")
    print()


def layer_table(label, layers):
    print(f"{label} -- OghmaNano layer editor, layer0 is the incident side")
    print(f"  {'#':>3}  {'material':<8} {'thickness':>10}  {'dy (m)':>10}  optical")
    for i, (mat, d) in enumerate(layers):
        print(f"  {i:>3}  {mat:<8} {d * 1e9:8.1f} nm  {d:10.3e}  Yes - n/k")
    print(f"  {len(layers):>3}  {'glass':<8} {SUBSTRATE_THICKNESS * 1e3:8.2f} mm  "
          f"{SUBSTRATE_THICKNESS:10.3e}  Yes - k (incoherent)")
    print()


def write_spectrum(path, layers, substrate="glass"):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = spectrum(layers, substrate, ANGLES)
    wls = data[ANGLES[0]][0]
    cols = []
    for deg in ANGLES:
        cols.append((f"R_{int(deg)}deg", data[deg][1]))
        cols.append((f"T_{int(deg)}deg", data[deg][2]))
    full = os.path.join(OUT_DIR, path)
    with open(full, "w") as fh:
        fh.write("wavelength_nm," + ",".join(name for name, _ in cols) + "\n")
        for i, wl in enumerate(wls):
            fh.write(f"{wl * 1e9:.1f}," +
                     ",".join(f"{col[i]:.6f}" for _, col in cols) + "\n")
    print(f"  wrote {os.path.relpath(full)}")


def write_oghma_materials():
    """Emit oghma_local/materials-style n.csv and alpha.csv for each material."""
    root = os.path.join(OUT_DIR, "oghma_materials")
    for name in ("ag", "ito", "tio2", "sio2", "glass", "pet"):
        folder = os.path.join(root, name)
        os.makedirs(folder, exist_ok=True)
        fn = dsp.MATERIALS[name]
        with open(os.path.join(folder, "n.csv"), "w") as f_n, \
                open(os.path.join(folder, "alpha.csv"), "w") as f_a:
            f_n.write("#oghma_data\n#x wavelength (m)\n#y n (au)\n")
            f_a.write("#oghma_data\n#x wavelength (m)\n#y alpha (m-1)\n")
            for wl in grid(300e-9, 1600e-9, 5e-9):
                n = fn(wl)
                alpha = 4.0 * math.pi * n.imag / wl
                f_n.write(f"{wl:.6e}\t{n.real:.6f}\n")
                f_a.write(f"{wl:.6e}\t{alpha:.6e}\n")
    print(f"  wrote {os.path.relpath(root)}/<material>/{{n,alpha}}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimise", action="store_true",
                    help="refine thicknesses by bounded coordinate descent")
    ap.add_argument("--periods", type=int, default=8,
                    help="periods per block in the chirped dielectric stack")
    args = ap.parse_args()

    self_test()
    print_dispersion()

    designs = [
        ("Route A1 - OMO   ITO/Ag/ITO", OMO),
        ("Route A2 - balanced OMOMO ITO/Ag/ITO/Ag/ITO (recommended start)", OMOMO),
        (f"Rejected baseline - quarter-wave chirped TiO2/SiO2, "
         f"{args.periods} periods x 3 blocks",
         chirped_dielectric(periods=args.periods)),
    ]

    for label, layers in designs:
        show(label, layers)

    if args.optimise:
        print("Coordinate-descent refinement")
        refined = []
        # Coordinate descent is useful for the few-layer metal stacks. It is
        # deliberately not run on the rejected 48-layer quarter-wave baseline:
        # independent per-layer refinement would be slow and would destroy its
        # repeatable-cell manufacturability.
        for label, layers in designs[:2]:
            print(f"  {label}")
            opt, _ = optimise(layers)
            refined.append((label + " [refined]", opt))
        print()
        for label, layers in refined:
            show(label, layers)
        designs = refined + [designs[2]]

    for label, layers in designs:
        layer_table(label, layers)

    print("Reference spectra for comparison against OghmaNano")
    write_spectrum("route_a1_omo.csv", designs[0][1])
    write_spectrum("route_a2_omomo.csv", designs[1][1])
    write_spectrum("route_b_dielectric.csv", designs[2][1])
    write_oghma_materials()


if __name__ == "__main__":
    main()
