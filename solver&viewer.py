#!/usr/bin/env python3
# ccf_all_in_one.py
# End-to-end post-processing for the CCF solver:
# - Load VTU (volume) and VTP (surface) files
# - Plot Mach/pressure/temperature contours and orthogonal slices
# - Extract wall distributions
# - Compare against reference CSV and compute L1/L2/L∞/RelL2 errors
#
# Usage examples:
#   python ccf_all_in_one.py --vtu output/ccf_50000.vtu --vtp output/ccf_surf_50000.vtp --ref paper_pressure.csv
#   python ccf_all_in_one.py --vtu output/ccf_50000.vtu --make-anim output/ccf_*.vtu --outdir plots

import argparse
from pathlib import Path
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def ensure_outdir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_vtu(path):
    # Reads VTU UnstructuredGrid with cell-centered arrays: pressure, Mach, temperature, density, velocity
    return pv.read(path)  # one-liner reader for VTK formats
def load_vtp(path):
    # Reads VTP PolyData with surface arrays: pressure_wall, temperature_wall, Mach_wall
    return pv.read(path)

def save_contours(vtu_path, outdir="plots"):
    out = ensure_outdir(outdir)
    m = load_vtu(vtu_path)
    items = [("Mach","jet"), ("pressure","plasma"), ("temperature","hot")]
    for name, cmap in items:
        if name not in m.cell_data and name not in m.point_data:
            print(f"[warn] scalar '{name}' not found in {vtu_path}, skipping")
            continue
        p = pv.Plotter(off_screen=True)
        p.add_mesh(m, scalars=name, cmap=cmap, show_edges=False)
        p.add_scalar_bar(name)
        p.camera_position = "xy"
        p.show(screenshot=str(out/f"{name}_contour.png"))
        p.close()
    print(f"[ok] Contour images saved to {out}")

def save_slices(vtu_path, outdir="plots", scalar="Mach"):
    out = ensure_outdir(outdir)
    m = load_vtu(vtu_path)
    # Three orthogonal slices through domain center
    bounds = m.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = 0.5*(bounds[0]+bounds[1])
    cy = 0.5*(bounds[2]+bounds[3])
    cz = 0.5*(bounds[4]+bounds[5])
    planes = [
        ((cx,cy,cz),(1,0,0),"yz"),
        ((cx,cy,cz),(0,1,0),"xz"),
        ((cx,cy,cz),(0,0,1),"xy"),
    ]
    for (o,n,label) in planes:
        sl = m.slice(origin=o, normal=n)
        p = pv.Plotter(off_screen=True)
        p.add_mesh(sl, scalars=scalar, cmap="jet", show_edges=False)
        p.add_scalar_bar(f"{scalar} ({label}-slice)")
        p.show(screenshot=str(out/f"{scalar}_slice_{label}.png"))
        p.close()
    print(f"[ok] Slice images saved to {out}")

def extract_wall(vtp_path):
    surf = load_vtp(vtp_path)
    pts = surf.points
    x = pts[:,0]
    order = np.argsort(x)
    data = {"x": x[order]}
    # Read common arrays if available
    for name in ["pressure_wall","temperature_wall","Mach_wall"]:
        if name in surf.array_names:
            data[name] = np.asarray(surf[name])[order]
    return data

def load_reference_csv(csv_path, x_col=0, y_col=1, skiprows=1):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=skiprows)
    return {"x": arr[:,x_col], "y": arr[:,y_col]}

def metrics(x_sim, y_sim, x_ref, y_ref):
    # Interpolate sim onto ref x-grid for fair comparison
    f = interp1d(x_sim, y_sim, bounds_error=False, fill_value="extrapolate")
    y_sim_on_ref = f(x_ref)
    diff = y_sim_on_ref - y_ref
    L1 = float(np.mean(np.abs(diff)))
    L2 = float(np.sqrt(np.mean(diff**2)))
    Linf = float(np.max(np.abs(diff)))
    relL2 = float(L2 / (np.sqrt(np.mean(y_ref**2)) + 1e-16))
    return {"L1":L1, "L2":L2, "Linf":Linf, "RelL2":relL2}

def plot_compare(x_sim, y_sim, x_ref, y_ref, ylabel, out_png):
    plt.figure(figsize=(9,6))
    plt.plot(x_sim, y_sim, 'b-', lw=2, label="Computed")
    plt.plot(x_ref, y_ref, 'ro', ms=4, label="Reference")
    plt.xlabel("x [m]")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def compare_wall(vtp_path, reference_csv=None, outdir="plots"):
    out = ensure_outdir(outdir)
    data = extract_wall(vtp_path)
    results = {}
    # If reference is provided, compare pressure by default
    if reference_csv and "pressure_wall" in data:
        ref = load_reference_csv(reference_csv)
        errs = metrics(data["x"], data["pressure_wall"], ref["x"], ref["y"])
        results["pressure"] = errs
        plot_compare(data["x"], data["pressure_wall"], ref["x"], ref["y"], "p_wall [Pa]", str(out/"pwall_compare.png"))
        with open(out/"pwall_errors.txt","w") as fp:
            for k,v in errs.items(): fp.write(f"{k}: {v:.6e}\n")
        print(f"[ok] Wrote {out/'pwall_compare.png'} and {out/'pwall_errors.txt'}")
    else:
        # Dump available arrays to CSV
        for k,v in data.items():
            if k!="x":
                np.savetxt(out/f"{k}.csv", np.c_[data["x"], v], delimiter=",", header="x,value", comments="")
        print(f"[ok] Exported wall CSVs to {out}")
    return results

def make_animation(vtu_glob, out_mp4="mach_anim.mp4", clim=(0,7)):
    files = sorted(glob.glob(vtu_glob))
    if not files:
        print(f"[warn] No files matched {vtu_glob}")
        return
    pl = pv.Plotter(off_screen=True)
    pl.open_movie(out_mp4, framerate=20)
    for f in files:
        m = pv.read(f)
        pl.clear()
        if "Mach" in m.cell_data:
            pl.add_mesh(m, scalars="Mach", cmap="jet", clim=clim)
        else:
            pl.add_mesh(m)
        pl.add_text(Path(f).name, font_size=10)
        pl.write_frame()
    pl.close()
    print(f"[ok] Animation saved to {out_mp4}")

def main():
    ap = argparse.ArgumentParser(description="All-in-one post for CCF VTU/VTP")
    ap.add_argument("--vtu", required=True, help="Path to volume VTU file")
    ap.add_argument("--vtp", help="Path to surface VTP file (optional)")
    ap.add_argument("--ref", help="CSV of reference curve (x,value) for pressure comparison (optional)")
    ap.add_argument("--outdir", default="plots", help="Output directory for images/reports")
    ap.add_argument("--make-anim", help="Glob for VTU time series to build a Mach animation (optional)")
    args = ap.parse_args()

    save_contours(args.vtu, args.outdir)
    save_slices(args.vtu, args.outdir, scalar="Mach")

    if args.vtp:
        compare_wall(args.vtp, args.ref, args.outdir)

    if args.make_anim:
        make_animation(args.make_anim, out_mp4=str(Path(args.outdir)/"mach_anim.mp4"))

if __name__ == "__main__":
    main()
