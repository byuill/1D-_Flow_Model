"""
Boundary-Condition Generator for the 1-D Flow Model
=====================================================
Generates a CSV file (BC_#.csv) with three columns:

    Elapsed Seconds | Discharge (m3/s) | Downstream Stage (m)

The hydrograph is built from user-defined parameters (all prompted with
sensible defaults).  Downstream stage is automatically computed from the
discharge time series using Manning's equation and the channel geometry
(width, slope, roughness) that match the 1-D flow model.

Dependencies: numpy, pandas, matplotlib, scipy
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============================================================
#  USER INPUTS  —  all prompts are gathered here
# ============================================================

def _get_float(prompt, default):
    """Prompt for a float; return *default* on empty / invalid / non-interactive."""
    if not sys.stdin.isatty():
        print(f"  {prompt} -> (non-interactive) using default: {default}")
        return default
    try:
        val = input(f"  {prompt} [default {default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"  -> using default: {default}")
        return default
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"  Invalid input '{val}', using default: {default}")
        return default


def _get_int(prompt, default):
    """Prompt for an int; return *default* on empty / invalid / non-interactive."""
    return int(_get_float(prompt, default))


def _get_choice(prompt, options, default):
    """Prompt for one of *options* (case-insensitive); return *default* if empty."""
    opts_str = " / ".join(options)
    if not sys.stdin.isatty():
        print(f"  {prompt} ({opts_str}) -> (non-interactive) using default: '{default}'")
        return default
    try:
        val = input(f"  {prompt} ({opts_str}) [default '{default}']: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"  -> using default: '{default}'")
        return default
    if not val:
        return default
    if val in [o.lower() for o in options]:
        return val
    print(f"  Unknown option '{val}', using default: '{default}'")
    return default


def collect_inputs():
    """Gather all user inputs in one clear block and return a dict."""
    print("\n" + "=" * 60)
    print("  BOUNDARY-CONDITION HYDROGRAPH GENERATOR")
    print("=" * 60)
    print("  Answer each prompt or press Enter to accept the default.\n")

    cfg = {}

    # --- Time parameters ---
    print("  ── Time Series ──")
    cfg["dt"]            = _get_float("Time step (s)", 1.0)
    cfg["t_total"]       = _get_float("Total simulation length (s)", 6000.0)

    # --- Hydrograph shape ---
    print("\n  ── Hydrograph Shape ──")
    cfg["n_peaks"]       = _get_int("Number of discharge peaks", 1)
    cfg["Q_peak"]        = _get_float("Peak discharge (m³/s)", 100.0)
    cfg["Q_base"]        = _get_float("Base-flow discharge (m³/s)", 20.0)
    cfg["fall_coeff"]    = _get_float(
        "Falling-limb coefficient (1 = symmetric, 2 = twice as long, 0.5 = half)", 1.5)

    # --- Optional secondary peak magnitude list ---
    #     If the user enters nothing, every peak uses Q_peak.
    if cfg["n_peaks"] > 1 and sys.stdin.isatty():
        print(f"\n  You chose {cfg['n_peaks']} peaks.  By default every peak = {cfg['Q_peak']} m³/s.")
        custom = input("  Enter individual peak magnitudes separated by commas,\n"
                       "  or press Enter to keep them all equal: ").strip()
        if custom:
            try:
                vals = [float(v) for v in custom.split(",")]
                if len(vals) == cfg["n_peaks"]:
                    cfg["peak_magnitudes"] = vals
                else:
                    print(f"  Expected {cfg['n_peaks']} values, got {len(vals)}.  Using uniform peaks.")
                    cfg["peak_magnitudes"] = None
            except ValueError:
                print("  Could not parse values.  Using uniform peaks.")
                cfg["peak_magnitudes"] = None
        else:
            cfg["peak_magnitudes"] = None
    else:
        cfg["peak_magnitudes"] = None

    # --- Noise ---
    print("\n  ── Random Noise ──")
    print("    none  – no noise")
    print("    small – ±3 % variance at 10-s intervals")
    print("    large – ±10 % variance at 60-s intervals")
    print("    both  – small + large combined")
    cfg["noise"]         = _get_choice("Noise mode", ["none", "small", "large", "both"], "none")

    # --- Smoothing ---
    print("\n  ── Smoothing ──")
    cfg["smooth_sigma"]  = _get_float(
        "Gaussian smoothing sigma (higher = smoother; 0 = none)", 5.0)

    # --- Channel geometry (must match the flow model for consistent stage) ---
    print("\n  ── Channel Geometry (must match the 1-D flow model) ──")
    cfg["chan_width"]     = _get_float("Channel width B (m)", 10.0)
    cfg["chan_slope"]     = _get_float("Bed slope S0 (m/m)", 0.01)
    cfg["chan_manning"]   = _get_float("Manning's n (s/m^(1/3))", 0.035)
    cfg["chan_length"]    = _get_float("Channel length L (m)", 1000.0)
    cfg["z_bed_upstream"] = _get_float("Bed elevation at upstream end (m)", -10.0)

    # --- Random seed (reproducibility) ---
    print("\n  ── Extras ──")
    cfg["seed"]          = _get_int("Random seed (-1 for random)", -1)

    print("\n" + "=" * 60)
    print("  Inputs accepted — generating hydrograph …")
    print("=" * 60 + "\n")
    return cfg


# ============================================================
#  HYDROGRAPH CONSTRUCTION
# ============================================================

def build_hydrograph(cfg):
    """
    Build a multi-peak triangular hydrograph from the user configuration.

    Peak placement
    --------------
    N peaks are placed at equal fractions of the total time:
        1 peak  →  50 %
        2 peaks →  33 %, 67 %
        3 peaks →  25 %, 50 %, 75 %
        N peaks →  1/(N+1), 2/(N+1), …, N/(N+1)

    Rising limb  = half the base-flow gap before the peak.
    Falling limb = rising limb × fall_coeff.

    Returns (time, discharge) arrays at the requested time step.
    """
    rng = np.random.default_rng(cfg["seed"] if cfg["seed"] >= 0 else None)

    dt      = cfg["dt"]
    t_total = cfg["t_total"]
    n_peaks = max(cfg["n_peaks"], 1)
    Q_peak  = cfg["Q_peak"]
    Q_base  = cfg["Q_base"]
    fall_c  = cfg["fall_coeff"]

    time = np.arange(0, t_total + dt, dt)
    Q    = np.full_like(time, Q_base)

    # --- locate peaks evenly ------------------------------------------------
    peak_fractions = np.linspace(0, 1, n_peaks + 2)[1:-1]   # e.g. [0.5] for 1 peak
    peak_times = peak_fractions * t_total

    # If the user supplied individual magnitudes, use them; else uniform.
    if cfg["peak_magnitudes"] is not None:
        magnitudes = np.array(cfg["peak_magnitudes"])
    else:
        magnitudes = np.full(n_peaks, Q_peak)

    # --- determine gap before each peak (for rising-limb duration) -----------
    boundaries = np.concatenate(([0.0], peak_times, [t_total]))
    for ip in range(n_peaks):
        t_pk  = peak_times[ip]
        Q_pk  = magnitudes[ip]

        gap_before = t_pk - boundaries[ip]          # base-flow window before this peak
        t_rise     = gap_before / 2.0                # rising limb = half that gap
        t_fall     = t_rise * fall_c                  # falling limb = rise × coefficient

        # Clamp the falling limb so it doesn't overrun the next peak / end
        gap_after  = boundaries[ip + 2] - t_pk if ip + 2 < len(boundaries) else t_total - t_pk
        t_fall     = min(t_fall, gap_after * 0.9)     # keep a small buffer

        t_start = t_pk - t_rise
        t_end   = t_pk + t_fall

        # --- rising limb (linear ramp) ---
        rise_mask = (time >= t_start) & (time <= t_pk)
        if t_rise > 0:
            Q[rise_mask] = np.maximum(
                Q[rise_mask],
                Q_base + (Q_pk - Q_base) * (time[rise_mask] - t_start) / t_rise,
            )

        # --- falling limb (linear ramp) ---
        fall_mask = (time > t_pk) & (time <= t_end)
        if t_fall > 0:
            Q[fall_mask] = np.maximum(
                Q[fall_mask],
                Q_pk - (Q_pk - Q_base) * (time[fall_mask] - t_pk) / t_fall,
            )

    # --- Random noise -------------------------------------------------------
    if cfg["noise"] in ("small", "both"):
        # ±3 % variance at 10-s frequency
        n_pts_small  = int(t_total / 10.0) + 1
        noise_small  = rng.uniform(-0.03, 0.03, n_pts_small)
        t_noise_sm   = np.linspace(0, t_total, n_pts_small)
        noise_interp = np.interp(time, t_noise_sm, noise_small)
        Q *= (1.0 + noise_interp)

    if cfg["noise"] in ("large", "both"):
        # ±10 % variance at 60-s frequency
        n_pts_large  = int(t_total / 60.0) + 1
        noise_large  = rng.uniform(-0.10, 0.10, n_pts_large)
        t_noise_lg   = np.linspace(0, t_total, n_pts_large)
        noise_interp = np.interp(time, t_noise_lg, noise_large)
        Q *= (1.0 + noise_interp)

    # --- Smoothing ----------------------------------------------------------
    sigma = cfg["smooth_sigma"]
    if sigma > 0:
        Q = gaussian_filter1d(Q, sigma=sigma)

    # Floor at zero (noise / smoothing could create negatives near base flow)
    Q = np.maximum(Q, 0.0)

    return time, Q


# ============================================================
#  DOWNSTREAM STAGE via MANNING'S EQUATION
# ============================================================

def _normal_depth_newton(Q, B, n, S, y0=1.0):
    """
    Compute normal depth for a rectangular channel using Newton-Raphson.

    Manning's equation:  Q = (1/n) * A * R^(2/3) * S^(1/2)
    with  A = B*y,  P = B + 2y,  R = A/P.

    We solve  f(y) = (1/n)*A*R^(2/3)*sqrt(S) - Q = 0  iteratively.
    """
    if S <= 0 or Q <= 0:
        return max(y0, 0.01)
    y = max(y0, 0.01)
    sqrt_S = S ** 0.5
    inv_n = 1.0 / n
    for _ in range(60):
        A = B * y
        P = B + 2.0 * y
        R = A / P
        R23 = R ** (2.0 / 3.0)
        f  = inv_n * A * R23 * sqrt_S - Q
        df = inv_n * sqrt_S * ((5.0 / 3.0) * B * R23
                               - (4.0 / 3.0) * R ** (5.0 / 3.0))
        if abs(df) < 1e-14:
            break
        y_new = y - f / df
        if abs(y_new - y) < 1e-7:
            break
        y = max(y_new, 1e-4)
    return y


def compute_stage_from_Q(Q_array, cfg, smooth_sigma=3.0):
    """
    Convert a discharge time series to a downstream-stage time series.

    Steps
    -----
    1. For every Q value, compute normal depth *y_n* with Manning's equation
       using the channel geometry specified in *cfg*.
    2. Compute the bed elevation at the downstream end:
           z_bed_ds = z_bed_upstream - S0 * L
    3. Stage = z_bed_ds + y_n   (absolute water-surface elevation).
    4. Apply a weak Gaussian smoother so transitions are gradual.
    """
    B  = cfg["chan_width"]
    S0 = cfg["chan_slope"]
    n  = cfg["chan_manning"]
    L  = cfg["chan_length"]
    z0 = cfg["z_bed_upstream"]

    z_bed_ds = z0 - S0 * L        # lowest bed elevation

    # Vectorised normal-depth calculation (element-wise Newton-Raphson)
    depth = np.array([_normal_depth_newton(max(q, 0.001), B, n, S0)
                      for q in Q_array])

    stage = z_bed_ds + depth       # absolute WSE at the downstream node

    # Weak smoother to keep transitions gradual
    if smooth_sigma > 0:
        stage = gaussian_filter1d(stage, sigma=smooth_sigma)

    return stage


# ============================================================
#  FILE OUTPUT  (sequential BC_#.csv in the working directory)
# ============================================================

def save_bc_csv(time, Q, stage):
    """Write BC_#.csv to the working directory with auto-incrementing number."""
    output_dir = os.getcwd()
    pattern = re.compile(r"BC_(\d+)\.csv", re.IGNORECASE)
    max_num = 0
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            max_num = max(max_num, int(m.group(1)))

    next_num  = max_num + 1
    filename  = f"BC_{next_num}.csv"
    full_path = os.path.join(output_dir, filename)

    df = pd.DataFrame({
        "Elapsed Seconds":       time,
        "Discharge (m3/s)":      Q,
        "Downstream Stage (m)":  stage,
    })
    df.to_csv(full_path, index=False)
    print(f"Saved  →  {full_path}  ({len(df)} rows)")
    return full_path, filename


# ============================================================
#  VISUALISATION
# ============================================================

def plot_bc(time, Q, stage, filename, cfg):
    """Two-panel plot: discharge hydrograph + stage placeholder."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # --- Discharge ---
    ax1.fill_between(time, Q, alpha=0.25, color="tab:blue")
    ax1.plot(time, Q, color="tab:blue", linewidth=1.2, label="Discharge")
    ax1.axhline(cfg["Q_base"], color="gray", linestyle="--", linewidth=0.8,
                label=f'Base flow ({cfg["Q_base"]} m³/s)')

    # Mark peaks
    peak_fracs = np.linspace(0, 1, max(cfg["n_peaks"], 1) + 2)[1:-1]
    for frac in peak_fracs:
        ax1.axvline(frac * cfg["t_total"], color="red", alpha=0.4, linestyle=":")

    ax1.set_ylabel("Discharge (m³/s)")
    ax1.set_title(f"Generated Boundary Conditions  —  {filename}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Stage (computed from Manning's equation) ---
    ax2.plot(time, stage, color="tab:red", linewidth=1.2, label="Downstream Stage (Manning's)")
    ax2.set_ylabel("Stage (m)")
    ax2.set_xlabel("Elapsed Time (s)")
    s_range = stage.max() - stage.min()
    ax2.set_ylim(stage.min() - max(s_range * 0.15, 0.2),
                 stage.max() + max(s_range * 0.15, 0.2))
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if sys.stdin.isatty():
        plt.show()
    else:
        img_path = os.path.join(os.getcwd(), filename.replace(".csv", ".png"))
        fig.savefig(img_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot  →  {img_path}")


# ============================================================
#  SUMMARY TABLE  (printed to console)
# ============================================================

def print_summary(time, Q, cfg):
    """Print a quick statistical summary of the generated hydrograph."""
    peak_fracs = np.linspace(0, 1, max(cfg["n_peaks"], 1) + 2)[1:-1]
    peak_times = peak_fracs * cfg["t_total"]

    print("\n┌─────────────────────────────────────────────┐")
    print("│          HYDROGRAPH SUMMARY                 │")
    print("├─────────────────────────────────────────────┤")
    print(f"│  Duration        : {cfg['t_total']:.0f} s")
    print(f"│  Time step       : {cfg['dt']:.2f} s  ({len(time)} rows)")
    print(f"│  Base flow       : {cfg['Q_base']:.2f} m³/s")
    print(f"│  Peak(s)         : {cfg['n_peaks']}")
    for i, tp in enumerate(peak_times):
        # find actual peak near the intended time
        idx = np.argmin(np.abs(time - tp))
        window = max(0, idx - 5), min(len(Q), idx + 6)
        actual_peak = np.max(Q[window[0]:window[1]])
        print(f"│    Peak {i+1}  t ≈ {tp:.0f} s   Q_max ≈ {actual_peak:.2f} m³/s")
    print(f"│  Falling coeff   : {cfg['fall_coeff']:.2f}")
    print(f"│  Noise           : {cfg['noise']}")
    print(f"│  Smooth σ        : {cfg['smooth_sigma']:.1f}")
    print(f"│  Total volume    : {np.trapz(Q, time):.1f} m³")
    print(f"│  Max discharge   : {np.max(Q):.2f} m³/s")
    print(f"│  Mean discharge  : {np.mean(Q):.2f} m³/s")
    print("└─────────────────────────────────────────────┘\n")


# ============================================================
#  MAIN
# ============================================================

def main():
    cfg = collect_inputs()

    time, Q = build_hydrograph(cfg)

    # Compute downstream stage from Manning's equation using channel geometry
    stage = compute_stage_from_Q(Q, cfg, smooth_sigma=3.0)

    path, fname = save_bc_csv(time, Q, stage)
    print_summary(time, Q, cfg)
    plot_bc(time, Q, stage, fname, cfg)

    return path


if __name__ == "__main__":
    main()
