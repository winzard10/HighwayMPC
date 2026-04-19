#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- inputs -----------
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
ax     = _need("ax")
ay     = _need("ay")
ddpsi  = _need("ddpsi")
jerk   = _need("jerk")
delta  = _need("delta")
R_cmd  = _need("R_cmd")
ddcmd  = _need("ddelta_cmd")
ey     = _need("ey")
epsi   = _need("epsi")
dv     = _need("dv")
v_ref  = _need("v_ref")
x_ref  = _need("x_ref")
y_ref  = _need("y_ref")
Fx_f   = _need("Fx_f")
Fx_r   = _need("Fx_r")
Fy_f   = _need("Fy_f")
Fy_r   = _need("Fy_r")
Fz_f   = _need("Fz_f")
Fz_r   = _need("Fz_r")
Mz     = _need("Mz")
alpha_f = _need("alpha_f")
alpha_r = _need("alpha_r")

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
        ax1.plot(xlb, ylb, label="Left lane outer border", linewidth=5)
        ax1.plot(xrb, yrb, label="Right lane outer border", linewidth=5)
        try:
            idxL = np.argsort(xl); idxR = np.argsort(xr)
            xL, yL = xl[idxL], yl[idxL]
            xR, yR = xr[idxR], yr[idxR]
            yR_on_L = np.interp(xL, xR, yR)
            ax1.fill_between(xL, yL, yR_on_L, alpha=0.06, zorder=0)
        except Exception:
            pass
    else:
        ax1.plot(xl, yl, label="Left lane center", linewidth=5)
        ax1.plot(xr, yr, label="Right lane center", linewidth=5)
        try:
            idxL = np.argsort(xl); idxR = np.argsort(xr)
            xL, yL = xl[idxL], yl[idxL]
            xR, yR = xr[idxR], yr[idxR]
            yR_on_L = np.interp(xL, xR, yR)
            ax1.fill_between(xL, yL, yR_on_L, alpha=0.06, zorder=0)
        except Exception:
            pass

    ax1.plot(xcenter, ycenter, linestyle='--', linewidth=3, label="Lane divider (centerline)")

# Vehicle reference & actual
ax1.plot(x_ref, y_ref, linestyle='--',linewidth=2, label="Reference path")
ax1.plot(x, y, ".", label="Vehicle path", color="C0", markersize=1)

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

# ======================================================================
# Figure 2: Time Histories (9 subplots)
#   0: e_y
#   1: e_psi
#   2: speed tracking
#   3: propulsion command
#   4: longitudinal & lateral acceleration
#   5: yaw acceleration
#   6: jerk
#   7: steering rate cmd
#   8: steering angle
# ======================================================================
fig2, axs2 = plt.subplots(3, 3, figsize=(12, 12))
axs2 = axs2.ravel()

# 0) e_y
axs2[0].plot(t, ey, label="e_y [m]")
axs2[0].set_title("Lateral Error e_y")
axs2[0].set_xlabel("Time [s]")
axs2[0].set_ylabel("e_y [m]")
axs2[0].legend()
axs2[0].grid(True)

# 1) e_psi
axs2[1].plot(t, epsi, label="e_psi [rad]")
axs2[1].set_title("Heading Error e_psi")
axs2[1].set_xlabel("Time [s]")
axs2[1].set_ylabel("e_psi [rad]")
axs2[1].legend()
axs2[1].grid(True)

# 2) Speed tracking
axs2[2].plot(t, v_ref, label="v_ref")
axs2[2].plot(t, v, "--", label="v")
if v_lead is not None:
    axs2[2].plot(t, v_lead, "--", label="v_lead")
axs2[2].set_title("Speed Tracking")
axs2[2].set_xlabel("Time [s]")
axs2[2].set_ylabel("Speed [m/s]")
axs2[2].legend()
axs2[2].grid(True)

# 3) Propulsion command
axs2[3].plot(t, R_cmd)
axs2[3].set_title("Propulsion Command")
axs2[3].set_xlabel("Time [s]")
axs2[3].set_ylabel("R [N]")
axs2[3].grid(True)

# 4) Longitudinal & lateral acceleration
axs2[4].plot(t, ax, label="Longitudinal [m/s²]")
axs2[4].plot(t, ay, label="Lateral [m/s²]")
axs2[4].set_title("Acceleration (Longitudinal & Lateral)")
axs2[4].set_xlabel("Time [s]")
axs2[4].set_ylabel("Acceleration")
axs2[4].legend()
axs2[4].grid(True)

# 5) Yaw acceleration
axs2[5].plot(t, ddpsi, label="Yaw accel [rad/s²]")
axs2[5].set_title("Yaw Acceleration")
axs2[5].set_xlabel("Time [s]")
axs2[5].set_ylabel("ddpsi [rad/s²]")
axs2[5].legend()
axs2[5].grid(True)

# 6) Jerk
axs2[6].plot(t[2:], jerk[2:], label="Jerk [m/s³]")
axs2[6].set_title("Jerk")
axs2[6].set_xlabel("Time [s]")
axs2[6].set_ylabel("Jerk [m/s³]")
axs2[6].legend()
axs2[6].grid(True)

# 7) Steering rate command
axs2[7].plot(t, ddcmd)
axs2[7].set_title("Steering Rate Command")
axs2[7].set_xlabel("Time [s]")
axs2[7].set_ylabel("d(delta)/dt [rad/s]")
axs2[7].grid(True)

# 8) Steering angle
axs2[8].plot(t, delta)
axs2[8].set_title("Steering Angle")
axs2[8].set_xlabel("Time [s]")
axs2[8].set_ylabel("delta [rad]")
axs2[8].grid(True)

plt.tight_layout()

# ======================================================================
# Figure 3: Tire Forces & ACC / Safety
#   0: Fy_f & Fy_r
#   1: Fz_f & Fz_r
#   2: Mz
#   3: dmin / gap
# ======================================================================
fig3, axs3 = plt.subplots(2, 2, figsize=(12, 8))
axs3 = axs3.ravel()

# 0) Lateral tire forces
axs3[0].plot(t, Fy_f, label="Fy_front")
axs3[0].plot(t, Fy_r, label="Fy_rear")
axs3[0].set_title("Lateral Tire Forces")
axs3[0].set_xlabel("Time [s]")
axs3[0].set_ylabel("Fy [N]")
axs3[0].legend()
axs3[0].grid(True)

# 1) Normal loads
axs3[1].plot(t, Fz_f, label="Fz_front")
axs3[1].plot(t, Fz_r, label="Fz_rear")
axs3[1].set_title("Normal Loads")
axs3[1].set_xlabel("Time [s]")
axs3[1].set_ylabel("Fz [N]")
axs3[1].legend()
axs3[1].grid(True)

# 2) Yaw moment
axs3[2].plot(t, Mz)
axs3[2].set_title("Yaw Moment Mz")
axs3[2].set_xlabel("Time [s]")
axs3[2].set_ylabel("Mz [Nm]")
axs3[2].grid(True)

# 3) Minimum distance / ACC gap
if dmin is not None or d_gap is not None:
    if dmin is not None:
        dmin_arr = np.asarray(dmin, dtype=float)
        good = np.isfinite(dmin_arr)
        if good.any():
            axs3[3].plot(t[good], dmin_arr[good], label="d_min to obstacles")
    if d_gap is not None:
        axs3[3].plot(t, d_gap, label="ACC gap to lead")
    axs3[3].set_title("Minimum Distance / ACC Gap")
    axs3[3].set_xlabel("Time [s]")
    axs3[3].set_ylabel("Distance [m]")
    axs3[3].legend()
    axs3[3].grid(True)
else:
    axs3[3].set_visible(False)  # nothing to show

plt.tight_layout()

# ======================================================================
# Figure 4: Tire Behavior / Handling Analysis (5 subplots)
#   0: Fy_f vs alpha_f & Fy_r vs alpha_r
#   1: Fy_norm vs slip angle (front/rear)
#   2: Mz vs lateral acceleration ay
#   3: (alpha_f - alpha_r) vs ay   <-- understeer gradient trend
#   4: Fy vs Fz (load sensitivity)
#   5: Friction circle Fx vs Fy (front+rear)
# ======================================================================

fig4, axs4 = plt.subplots(3, 2, figsize=(12, 14))
axs4 = axs4.ravel()

# ---------------------------------------------------
# 0) Fy vs slip angle
axs4[0].plot(alpha_f, Fy_f, ".", markersize=2, label="Front Fy vs α_f")
axs4[0].plot(alpha_r, Fy_r, ".", markersize=2, label="Rear Fy vs α_r")
axs4[0].set_title("Fy vs Slip Angle")
axs4[0].set_xlabel("Slip angle α [rad]")
axs4[0].set_ylabel("Fy [N]")
axs4[0].legend(); axs4[0].grid(True)

# ---------------------------------------------------
# 1) Normalized Fy vs slip angle
# --- You may define mu_f, mu_r here (assuming constant for visualization) ---
mu_f = 0.9
mu_r = 0.9

# Avoid zero-div
Fz_f_safe = np.maximum(Fz_f, 1e-3)
Fz_r_safe = np.maximum(Fz_r, 1e-3)

Fy_f_norm = Fy_f / (mu_f * Fz_f_safe)
Fy_r_norm = Fy_r / (mu_r * Fz_r_safe)

axs4[1].plot(alpha_f, Fy_f_norm, ".", markersize=2, label="Front Fy_norm")
axs4[1].plot(alpha_r, Fy_r_norm, ".", markersize=2, label="Rear  Fy_norm")
axs4[1].axhline(1.0, color="r", linestyle="--", alpha=0.7)   # Saturation limit
axs4[1].axhline(-1.0, color="r", linestyle="--", alpha=0.7)
axs4[1].set_title("Normalized Lateral Force Fy / (μFz)")
axs4[1].set_xlabel("Slip angle α [rad]")
axs4[1].set_ylabel("Normalized Fy")
axs4[1].legend(); axs4[1].grid(True)

# ---------------------------------------------------
# 2) Mz vs lateral acceleration (stability trend)
axs4[2].plot(ay, Mz, ".", markersize=2)
axs4[2].set_title("Yaw Moment vs Lateral Acceleration")
axs4[2].set_xlabel("ay [m/s²]")
axs4[2].set_ylabel("Mz [Nm]")
axs4[2].grid(True)

# ---------------------------------------------------
# 3) Understeer indicator: (alpha_f - alpha_r) vs ay
alpha_diff = alpha_f - alpha_r
axs4[3].plot(ay, alpha_diff, ".", markersize=2)
axs4[3].set_title("Understeer Indicator: (α_f - α_r) vs ay")
axs4[3].set_xlabel("ay [m/s²]")
axs4[3].set_ylabel("α_f - α_r [rad]")
axs4[3].grid(True)

# ---------------------------------------------------
# 4) Load sensitivity: Fy vs Fz
axs4[4].plot(Fz_f, Fy_f, ".", markersize=2, label="Front")
axs4[4].plot(Fz_r, Fy_r, ".", markersize=2, label="Rear")
axs4[4].set_title("Load Sensitivity: Fy vs Fz")
axs4[4].set_xlabel("Fz [N]")
axs4[4].set_ylabel("Fy [N]")
axs4[4].legend(); axs4[4].grid(True)

# ---------------------------------------------------
# 5) Friction circle (Fx vs Fy)
axs4[5].plot(Fx_f, Fy_f, ".", markersize=2, label="Front")
axs4[5].plot(Fx_r, Fy_r, ".", markersize=2, label="Rear")

Fz_f_mean = np.mean(Fz_f)     # Approximate mean normal load for visualization
Fz_r_mean = np.mean(Fz_r)

theta = np.linspace(0, 2*np.pi, 400)

R_f = mu_f * Fz_f_mean
R_r = mu_r * Fz_r_mean

axs4[5].plot(R_f * np.cos(theta),
             R_f * np.sin(theta),
             "--", linewidth=1.0, label="Front limit")

axs4[5].plot(R_r * np.cos(theta),
             R_r * np.sin(theta),
             "--", linewidth=1.0, label="Rear limit")

axs4[5].set_title("Friction Circle (Fx vs Fy)")
axs4[5].set_xlabel("Fx [N]")
axs4[5].set_ylabel("Fy [N]")
axs4[5].legend(); axs4[5].grid(True)

plt.tight_layout()
plt.show()
