#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- inputs ----------
sim_path  = sys.argv[1] if len(sys.argv) > 1 else "sim_log.csv"
map_path  = "data/lane_centerlines.csv"
# Prefer enhanced file if present
if Path("data/lane_centerlines_enhanced.csv").exists():
    map_path = "data/lane_centerlines_enhanced.csv"

# ---------- load sim log ----------
import pandas as _pd

def _load_sim_csv(path):
    # First try pandas with the python engine (handles ragged rows by padding with NaN)
    try:
        df = _pd.read_csv(path, engine="python")
        return df
    except Exception as e:
        # Fallback to numpy if needed
        import numpy as _np
        data = _np.genfromtxt(path, delimiter=",", names=True)
        df = _pd.DataFrame({name: data[name] for name in data.dtype.names})
        return df

df = _load_sim_csv(sim_path)

# required columns
def _need(col): 
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {sim_path}")
    return df[col].values

t      = _need("t")
x      = _need("x")
y      = _need("y")
v      = _need("vx")
a      = _need("ax")
delta  = _need("delta")
R_cmd  = _need("R_cmd")
ddcmd  = _need("ddelta_cmd")
ey     = _need("ey")
epsi   = _need("epsi")
dv     = _need("dv")
v_ref  = _need("v_ref")
x_ref  = _need("x_ref")
y_ref  = _need("y_ref")
F_fl   = _need("Fy_f")
F_rl   = _need("Fy_r")

# Optional columns
psi = df["psi"].values if "psi" in df.columns else None
psi_ref = df["psi_ref"].values if "psi_ref" in df.columns else None
alpha = df["alpha"].values if "alpha" in df.columns else None
dmin = df["dmin"].values if "dmin" in df.columns else None
v_lead = df["v_lead"].values if "v_lead" in df.columns else None
d_gap = df["d_gap"].values if "d_gap" in df.columns else None

print(f"Loaded: {sim_path}")
print(f"Duration: {t[-1]:.2f}s, samples: {len(t)}")
import numpy as np
print(f"RMS ey: {np.sqrt(np.nanmean(ey**2)):.4f} m,  RMS epsi: {np.sqrt(np.nanmean(epsi**2)):.5f} rad")
print(f"RMS ey: {np.sqrt(np.mean(ey**2)):.4f} m,  RMS epsi: {np.sqrt(np.mean(epsi**2)):.5f} rad")

# ---------- try to load lane map ----------
have_map = Path(map_path).exists()
lane = None
if have_map:
    lane = np.genfromtxt(map_path, delimiter=",", names=True)
    cols = set(lane.dtype.names or [])

    # Base lane center curves
    def get(name_candidates):
        for nm in name_candidates:
            if nm in cols: return lane[nm]
        raise KeyError(name_candidates[0])

    # Support both legacy and enhanced CSVs
    xr = get(["x_right", "xr"]) if have_map else None
    yr = get(["y_right", "yr"]) if have_map else None
    xl = get(["x_left",  "xl"]) if have_map else None
    yl = get(["y_left",  "yl"]) if have_map else None

    # Optional enhanced columns
    xcenter = lane["x_centerline"] if "x_centerline" in cols else 0.5*(xl + xr)
    ycenter = lane["y_centerline"] if "y_centerline" in cols else 0.5*(yl + yr)

    xlb = lane["x_left_border"]  if "x_left_border"  in cols else None
    ylb = lane["y_left_border"]  if "y_left_border"  in cols else None
    xrb = lane["x_right_border"] if "x_right_border" in cols else None
    yrb = lane["y_right_border"] if "y_right_border" in cols else None
else:
    xr = yr = xl = yl = xcenter = ycenter = xlb = ylb = xrb = yrb = None

# ---------- obstacles (optional) ----------
obs_path = "data/obstacles.csv" if Path("data/obstacles.csv").exists() else ("data/obstacles_example.csv" if Path("data/obstacles_example.csv").exists() else None)
obstacles = None
if obs_path is not None:
    try:
        import csv as _csv
        obstacles = []
        with open(obs_path, "r") as f:
            reader = _csv.DictReader(f)
            for r in reader:
                obstacles.append({
                    "id": int(r["id"]),
                    "kind": (r["kind"].strip().lower()),
                    "x0": float(r["x0"]), "y0": float(r["y0"]),
                    "vx": float(r["vx"]), "vy": float(r["vy"]),
                    "radius": float(r["radius"]),
                    "t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
                })
    except Exception as e:
        print(f"[viz] obstacles load failed: {e}")

## ---------- Figure 1: Trajectory ----------
fig1, ax1 = plt.subplots(figsize=(8, 6))

# 1) XY trajectory + borders + divider
if have_map:
    if xlb is not None and xrb is not None:
        ax1.plot(xlb, ylb, label="Left lane outer border", linewidth=1)
        ax1.plot(xrb, yrb, label="Right lane outer border", linewidth=1)
        try:
            idxL = np.argsort(xl); idxR = np.argsort(xr)
            xL, yL = xl[idxL], yl[idxL]
            xR, yR = xr[idxR], yr[idxR]
            yR_on_L = np.interp(xL, xR, yR)
            ax1.fill_between(xL, yL, yR_on_L, alpha=0.06, zorder=0)
        except Exception:
            pass
    else:
        ax1.plot(xl, yl, label="Left lane center", linewidth=1)
        ax1.plot(xr, yr, label="Right lane center", linewidth=1)
        try:
            idxL = np.argsort(xl); idxR = np.argsort(xr)
            xL, yL = xl[idxL], yl[idxL]
            xR, yR = xr[idxR], yr[idxR]
            yR_on_L = np.interp(xL, xR, yR)
            ax1.fill_between(xL, yL, yR_on_L, alpha=0.06, zorder=0)
        except Exception:
            pass

    ax1.plot(xcenter, ycenter, linestyle='-', linewidth=1.5, label="Lane divider (centerline)")

# Vehicle reference & actual
ax1.plot(x_ref, y_ref, linewidth=2, label="Reference path")
ax1.plot(x, y, "--", label="Vehicle path")

# Obstacles
if obstacles:
    T0, T1 = float(t[0]), float(t[-1])
    T_show = float(t[-1])
    for ob in obstacles:
        ta, tb = max(T0, ob["t_start"]), min(T1, ob["t_end"])
        if ob["kind"] == "moving" and tb > ta:
            tseg = np.linspace(ta, tb, 40)
            xs = ob["x0"] + ob["vx"]*(tseg - ob["t_start"])
            ys = ob["y0"] + ob["vy"]*(tseg - ob["t_start"])
            ax1.plot(xs, ys, linewidth=1, alpha=0.8)

        tt = min(max(T_show, ob["t_start"]), ob["t_end"])
        cx = ob["x0"] + ob["vx"]*(tt - ob["t_start"])
        cy = ob["y0"] + ob["vy"]*(tt - ob["t_start"])
        r  = ob["radius"]
        circle = plt.Circle((cx, cy), r, fill=False, linewidth=1.5, zorder=6)
        ax1.add_patch(circle)
        ax1.plot([cx], [cy], "o", ms=3, zorder=7)

ax1.set_title("Trajectory (XY)")
ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]")
ax1.legend(); ax1.grid(True)
plt.tight_layout()


# ---------- Figure 2: Other signals ----------
fig2, axs = plt.subplots(3, 3, figsize=(12, 10))
axs = axs.ravel()

# 2) Speed tracking
axs[0].plot(t, v_ref, label="v_ref")
axs[0].plot(t, v, "--", label="v")
axs[0].plot(t, v_lead, "--", label="v_lead")
axs[0].set_title("Speed Tracking")
axs[0].set_xlabel("Time [s]"); axs[0].set_ylabel("Speed [m/s]")
axs[0].legend(); axs[0].grid(True)

# 3) Errors
axs[1].plot(t, ey, label="e_y [m]")
axs[1].plot(t, epsi, "--", label="e_psi [rad]")
axs[1].set_title("Tracking Errors")
axs[1].set_xlabel("Time [s]"); axs[1].set_ylabel("Error")
axs[1].legend(); axs[1].grid(True)

# 4) Acceleration
axs[2].plot(t, R_cmd)
axs[2].set_title("Propulsion Command")
axs[2].set_xlabel("Time [s]"); axs[2].set_ylabel("R [N]")
axs[2].grid(True)

# 5) Steering rate
axs[3].plot(t, ddcmd)
axs[3].set_title("Steering Rate Command")
axs[3].set_xlabel("Time [s]"); axs[3].set_ylabel("d(delta)/dt [rad/s]")
axs[3].grid(True)

# 6) Steering angle
axs[4].plot(t, delta)
axs[4].set_title("Steering Angle")
axs[4].set_xlabel("Time [s]"); axs[4].set_ylabel("delta [rad]")
axs[4].grid(True)

# 7) Minimum distance to obstacles
if 'dmin' in locals() and dmin is not None:
    dmin_arr = np.asarray(dmin, dtype=float)
    good = np.isfinite(dmin_arr)
    axs[5].set_title("Minimum distance to...")
    if good.any():
        axs[5].plot(t[good], dmin_arr[good], label="obstacles")
        axs[5].set_xlabel("Time [s]"); axs[5].set_ylabel("d_min [m]")
        axs[5].grid(True)
    if 'd_gap' in locals() and d_gap is not None:
        axs[5].plot(t, d_gap, label="lead vehicle")
        axs[5].legend()

# 8) Steering angle
axs[6].plot(t, F_fl, label="Lateral forces at front")
axs[6].plot(t, F_rl, label="Lateral forces at rear")
axs[6].set_title("Tire force")
axs[6].set_xlabel("Time [s]"); axs[6].set_ylabel("Tire force [N]")
axs[6].legend()
axs[6].grid(True)

# 9) Acceleration
axs[7].plot(t, a, label="Longitudinal acceleration")
axs[7].set_title("Acceleration")
axs[7].set_xlabel("Time [s]"); axs[7].set_ylabel("Acceleration [m/s^2]")
axs[7].legend()
axs[7].grid(True)


plt.tight_layout()
plt.show()
