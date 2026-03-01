"""
Boundary-Condition & Channel Geometry Generator  —  1-D Flow Model
===================================================================
Generates two output files:

  Flow_Data_##.csv    : Time (s) | Discharge (m3/s) | Downstream Stage (m)
  Channel_Data_##.csv : Distance (m) | Bed Elevation (m) | Width (m) | Manning's n

Configure everything in the CONFIGURATION SWITCHES section below.
No terminal prompts are used — edit the switches and run the script.

Dependencies: numpy, pandas, matplotlib, scipy
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


# ================================================================
#  CONFIGURATION SWITCHES
#  Edit values below.  All other code can be left untouched.
# ================================================================

# ----------------------------------------------------------------
#  SIMULATION TIME
# ----------------------------------------------------------------
DT: float      = 1.0       # Time step (seconds)
T_TOTAL: float = 6000.0    # Total simulation duration (seconds)

# ================================================================
#  DEFINE DISCHARGE
# ================================================================
# Choose one of:
#   1 -> Steady       - constant discharge throughout simulation
#   2 -> Unsteady     - triangular multi-peak hydrograph
#   3 -> Sine curve   - sinusoidal oscillation about a base flow
#   4 -> Time series  - read discharge from a CSV file

DISCHARGE_TYPE: int = 2     # 1 | 2 | 3 | 4

# ----------------------------------------------------------------
#  Option 1 - Steady Discharge
# ----------------------------------------------------------------
STEADY_Q: float = 50.0      # Constant discharge (m3/s)

# ----------------------------------------------------------------
#  Option 2 - Unsteady (triangular multi-peak hydrograph)
# ----------------------------------------------------------------
UNSTEADY_N_PEAKS: int      = 1      # Number of discharge peaks
UNSTEADY_Q_PEAK: float     = 100.0  # Peak discharge (m3/s)
UNSTEADY_Q_BASE: float     = 20.0   # Base-flow discharge (m3/s)
UNSTEADY_FALL_COEFF: float = 1.5    # Falling-limb coefficient
                                    #   1.0 = symmetric rise/fall
                                    #   2.0 = falling limb twice as long as rising
                                    #   0.5 = falling limb half as long as rising

# Individual peak magnitudes (optional).
#   None  -> every peak equals UNSTEADY_Q_PEAK.
#   List  -> one value per peak, e.g. [80.0, 120.0] for two peaks.
UNSTEADY_PEAK_MAGNITUDES = None

UNSTEADY_NOISE: str     = "none"  # "none" | "small" (+/-3% @10s) | "large" (+/-10% @60s) | "both"
UNSTEADY_SMOOTH: float  = 5.0     # Gaussian smoothing sigma (0 = no smoothing)
UNSTEADY_SEED: int      = -1      # Random seed for noise (-1 = non-reproducible)

# ----------------------------------------------------------------
#  Option 3 - Sine Curve Discharge
# ----------------------------------------------------------------
SINE_Q_BASE: float  = 50.0    # Base (mean) discharge (m3/s)
SINE_Q_AMP: float   = 30.0    # Wave amplitude (m3/s)  -> range: base +/- amp
SINE_Q_FREQ: float  = 0.001   # Frequency (Hz, cycles per second)
SINE_Q_MIN: float   = 5.0     # Minimum allowable discharge (m3/s) - clamps the trough

# ----------------------------------------------------------------
#  Option 4 - User Time Series File
# ----------------------------------------------------------------
# CSV must have two columns (with headers): elapsed minutes, discharge (m3/s).
# If the time series is shorter than T_TOTAL, the last discharge value is
# held constant for the remainder of the simulation.
Q_FILE_NAME: str = "discharge_input.csv"   # filename in current working directory

# ================================================================
#  DEFINE DOWNSTREAM STAGE
# ================================================================
#   1 -> Constant stage  - fixed water-surface elevation
#   2 -> Rating curve    - Manning's equation applied to outlet geometry
#   3 -> Sine curve      - sinusoidal oscillation about a base stage
#   4 -> Storm surge     - trapezoidal surge superimposed on a base stage

STAGE_TYPE: int = 2     # 1 | 2 | 3 | 4

# ----------------------------------------------------------------
#  Option 1 - Constant Stage
# ----------------------------------------------------------------
CONSTANT_STAGE: float = -9.0    # Fixed downstream water-surface elevation (m)

# ----------------------------------------------------------------
#  Option 2 - Rating Curve (Manning's equation at the outlet)
# ----------------------------------------------------------------
# The code reads the channel geometry at the downstream end as defined
# in the DEFINE CHANNEL section and solves Manning's equation for normal
# depth at each discharge value.  No extra inputs are needed here.
RATING_SMOOTH_SIGMA: float = 3.0    # Gaussian smoothing applied to the resulting stage
                                    # time series (0 = none)

# ----------------------------------------------------------------
#  Option 3 - Sine Curve Stage
# ----------------------------------------------------------------
SINE_STAGE_BASE: float  = -9.0    # Base (mean) stage (m)
SINE_STAGE_AMP: float   = 1.0     # Wave amplitude (m)
SINE_STAGE_FREQ: float  = 0.001   # Frequency (Hz)
SINE_STAGE_MIN: float   = -11.0   # Minimum allowable stage (m) - clamps the trough

# ----------------------------------------------------------------
#  Option 4 - Storm Surge
# ----------------------------------------------------------------
SURGE_BASE: float     = -9.0    # Background stage before / after the surge (m)
SURGE_PEAK: float     = -5.0    # Peak stage during the surge (m)
SURGE_DURATION: float = 1800.0  # Total surge duration (s)
                                #   Rise  = first  1/3 of this duration
                                #   Peak  = middle 1/3 of this duration
                                #   Fall  = last   1/3 of this duration
SURGE_TIMING: float   = 0.5     # Fraction of T_TOTAL when the surge STARTS (0-1)
                                #   0.5 means the surge begins halfway through the sim

# ================================================================
#  DEFINE CHANNEL
# ================================================================
# --- Bed Elevation ---
# List of [distance_downstream, bed_elevation] pairs.  At least two pairs required.
# Bed elevation is linearly interpolated between defined points.
# First pair = upstream inlet; last pair = downstream outlet.
# Example: [[0, 0], [1000, -10]] -> 1000 m channel, drops 10 m from inlet to outlet.
BED_PAIRS: list = [
    [0,    0.0],
    [1000, -10.0],
]

SMOOTH_BED: bool        = False  # Apply Gaussian smoothing to bed elevation profile
SMOOTH_BED_SIGMA: float = 5.0    # Smoothing length scale (m)

# --- Channel Width ---
# List of [distance_downstream, width] pairs.
# If only one pair is given the width is constant throughout the domain.
# Example: [[0, 30], [500, 50], [1000, 30]] -> narrows from 50 m at mid to 30 m at ends.
WIDTH_PAIRS: list = [
    [0,    10.0],
    [1000, 10.0],
]

SMOOTH_WIDTH: bool        = False  # Apply Gaussian smoothing to width profile
SMOOTH_WIDTH_SIGMA: float = 5.0    # Smoothing length scale (m)

# --- Manning's Roughness ---
# List of [distance_downstream, Manning's n] pairs.
# If only one pair is given, that n applies to the full channel.
# Example: [[0, 0.030], [500, 0.050], [1000, 0.030]]
MANNING_PAIRS: list = [
    [0,    0.035],
    [1000, 0.035],
]

# --- Spatial Resolution ---
# The Channel_Data output file is written at this spatial interval (m).
# Bed elevation, width, and Manning's n are all interpolated to this grid.
# Smoothing (if enabled) is applied at this step.
CHANNEL_DX: float = 10.0   # Grid spacing (m)


# ================================================================
#  END OF CONFIGURATION SWITCHES
# ================================================================
 
# --- Optional schematic output ---
# When True, two JPEG schematics are written alongside the CSV outputs:
#  - Flow_Data_##.jpg    : time (minutes) vs discharge (left axis) and stage (right axis)
#  - Channel_Data_##.jpg : profile view (z vs x) and map view (plan with channel width)
OUTPUT_SCHEMATICS: bool = True
SCHEMATIC_DPI: int = 300
SCHEMATIC_FONT: str = "Times New Roman"


# ----------------------------------------------------------------
#  DISCHARGE GENERATORS
# ----------------------------------------------------------------

def _build_time_array() -> np.ndarray:
    return np.arange(0, T_TOTAL + DT, DT)


def gen_discharge_steady(time: np.ndarray) -> np.ndarray:
    """Option 1: uniform discharge at STEADY_Q."""
    return np.full_like(time, STEADY_Q)


def gen_discharge_unsteady(time: np.ndarray) -> np.ndarray:
    """Option 2: multi-peak triangular hydrograph."""
    rng = np.random.default_rng(UNSTEADY_SEED if UNSTEADY_SEED >= 0 else None)
    n_peaks = max(UNSTEADY_N_PEAKS, 1)
    Q_base  = UNSTEADY_Q_BASE
    fall_c  = UNSTEADY_FALL_COEFF

    Q = np.full_like(time, Q_base)

    peak_fracs = np.linspace(0, 1, n_peaks + 2)[1:-1]
    peak_times = peak_fracs * T_TOTAL

    if (UNSTEADY_PEAK_MAGNITUDES is not None
            and len(UNSTEADY_PEAK_MAGNITUDES) == n_peaks):
        magnitudes = np.array(UNSTEADY_PEAK_MAGNITUDES, dtype=float)
    else:
        magnitudes = np.full(n_peaks, UNSTEADY_Q_PEAK)

    boundaries = np.concatenate(([0.0], peak_times, [T_TOTAL]))

    for ip in range(n_peaks):
        t_pk = peak_times[ip]
        Q_pk = magnitudes[ip]

        gap_before = t_pk - boundaries[ip]
        t_rise     = gap_before / 2.0
        t_fall     = t_rise * fall_c
        gap_after  = (boundaries[ip + 2] - t_pk
                      if ip + 2 < len(boundaries) else T_TOTAL - t_pk)
        t_fall     = min(t_fall, gap_after * 0.9)

        t_start = t_pk - t_rise
        t_end   = t_pk + t_fall

        rise_mask = (time >= t_start) & (time <= t_pk)
        if t_rise > 0:
            Q[rise_mask] = np.maximum(
                Q[rise_mask],
                Q_base + (Q_pk - Q_base) * (time[rise_mask] - t_start) / t_rise,
            )

        fall_mask = (time > t_pk) & (time <= t_end)
        if t_fall > 0:
            Q[fall_mask] = np.maximum(
                Q[fall_mask],
                Q_pk - (Q_pk - Q_base) * (time[fall_mask] - t_pk) / t_fall,
            )

    # Noise
    if UNSTEADY_NOISE in ("small", "both"):
        n_pts = int(T_TOTAL / 10.0) + 1
        noise = rng.uniform(-0.03, 0.03, n_pts)
        t_pts = np.linspace(0, T_TOTAL, n_pts)
        Q *= (1.0 + np.interp(time, t_pts, noise))

    if UNSTEADY_NOISE in ("large", "both"):
        n_pts = int(T_TOTAL / 60.0) + 1
        noise = rng.uniform(-0.10, 0.10, n_pts)
        t_pts = np.linspace(0, T_TOTAL, n_pts)
        Q *= (1.0 + np.interp(time, t_pts, noise))

    # Smoothing
    if UNSTEADY_SMOOTH > 0:
        Q = gaussian_filter1d(Q, sigma=UNSTEADY_SMOOTH)

    return np.maximum(Q, 0.0)


def gen_discharge_sine(time: np.ndarray) -> np.ndarray:
    """Option 3: sinusoidal discharge."""
    Q = SINE_Q_BASE + SINE_Q_AMP * np.sin(2.0 * np.pi * SINE_Q_FREQ * time)
    return np.maximum(Q, SINE_Q_MIN)


def gen_discharge_file(time: np.ndarray) -> np.ndarray:
    """Option 4: discharge read from a two-column CSV (elapsed minutes, discharge)."""
    fpath = os.path.join(os.getcwd(), Q_FILE_NAME)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(
            f"Discharge file not found: {fpath}\n"
            f"  Expected two columns: elapsed minutes, discharge (m3/s)."
        )
    df = pd.read_csv(fpath)
    if df.shape[1] < 2:
        raise ValueError(f"Discharge file must have at least 2 columns: {fpath}")

    t_min = df.iloc[:, 0].values.astype(float)
    q_val = df.iloc[:, 1].values.astype(float)
    t_sec = t_min * 60.0

    # Extend to simulation duration by holding the last value constant
    if t_sec[-1] < T_TOTAL:
        t_sec = np.append(t_sec, T_TOTAL)
        q_val = np.append(q_val, q_val[-1])

    return np.interp(time, t_sec, q_val, left=q_val[0], right=q_val[-1])


def generate_discharge(time: np.ndarray) -> np.ndarray:
    """Dispatch to the selected discharge generator."""
    dispatch = {
        1: gen_discharge_steady,
        2: gen_discharge_unsteady,
        3: gen_discharge_sine,
        4: gen_discharge_file,
    }
    if DISCHARGE_TYPE not in dispatch:
        raise ValueError(f"DISCHARGE_TYPE must be 1-4, got {DISCHARGE_TYPE}")
    return dispatch[DISCHARGE_TYPE](time)


# ----------------------------------------------------------------
#  STAGE GENERATORS
# ----------------------------------------------------------------

def _normal_depth_newton(Q: float, B: float, n: float, S: float,
                          y0: float = 1.0) -> float:
    """
    Solve for normal depth in a rectangular channel via Newton-Raphson.
    Manning: Q = (1/n) * B*y * (B*y / (B + 2y))^(2/3) * sqrt(S)
    """
    if S <= 0 or Q <= 0:
        return max(y0, 0.01)
    y      = max(y0, 0.01)
    sqrt_S = S ** 0.5
    inv_n  = 1.0 / n
    for _ in range(80):
        A   = B * y
        P   = B + 2.0 * y
        R   = A / P
        R23 = R ** (2.0 / 3.0)
        f   = inv_n * A * R23 * sqrt_S - Q
        df  = inv_n * sqrt_S * (
            (5.0 / 3.0) * B * R23 - (4.0 / 3.0) * B * R ** (5.0 / 3.0) / P
        )
        if abs(df) < 1e-14:
            break
        y_new = y - f / df
        delta = abs(y_new - y)
        y = max(y_new, 1e-5)
        if delta < 1e-8:
            break
    return y


def gen_stage_constant(time: np.ndarray, **_) -> np.ndarray:
    """Option 1: constant stage."""
    return np.full_like(time, CONSTANT_STAGE)


def gen_stage_rating_curve(time: np.ndarray, Q: np.ndarray,
                            outlet_bed_z: float, outlet_width: float,
                            outlet_manning: float, outlet_slope: float,
                            **_) -> np.ndarray:
    """
    Option 2: Manning's rating curve at the channel outlet.
    For each Q, compute normal depth; stage = outlet_bed_z + depth.
    """
    depth = np.array([
        _normal_depth_newton(
            max(q, 1e-4), outlet_width, outlet_manning, outlet_slope
        )
        for q in Q
    ])
    stage = outlet_bed_z + depth
    if RATING_SMOOTH_SIGMA > 0:
        stage = gaussian_filter1d(stage, sigma=RATING_SMOOTH_SIGMA)
    return stage


def gen_stage_sine(time: np.ndarray, **_) -> np.ndarray:
    """Option 3: sinusoidal stage."""
    stage = SINE_STAGE_BASE + SINE_STAGE_AMP * np.sin(
        2.0 * np.pi * SINE_STAGE_FREQ * time
    )
    return np.maximum(stage, SINE_STAGE_MIN)


def gen_stage_surge(time: np.ndarray, **_) -> np.ndarray:
    """
    Option 4: trapezoidal storm surge.
    Surge phases (each = 1/3 of SURGE_DURATION):
        Rise  - linear ramp from SURGE_BASE to SURGE_PEAK
        Peak  - held at SURGE_PEAK
        Fall  - linear ramp from SURGE_PEAK back to SURGE_BASE
    Surge starts at SURGE_TIMING * T_TOTAL.
    """
    stage = np.full_like(time, SURGE_BASE)

    t_start = SURGE_TIMING * T_TOTAL
    d3      = SURGE_DURATION / 3.0
    t_r1    = t_start               # rise start
    t_r2    = t_start + d3          # rise end / peak start
    t_p2    = t_start + 2.0 * d3   # peak end / fall start
    t_f2    = t_start + SURGE_DURATION  # fall end

    mask_r = (time >= t_r1) & (time < t_r2)
    if d3 > 0:
        stage[mask_r] = (SURGE_BASE
                         + (SURGE_PEAK - SURGE_BASE)
                         * (time[mask_r] - t_r1) / d3)

    mask_p = (time >= t_r2) & (time < t_p2)
    stage[mask_p] = SURGE_PEAK

    mask_f = (time >= t_p2) & (time < t_f2)
    if d3 > 0:
        stage[mask_f] = (SURGE_PEAK
                         - (SURGE_PEAK - SURGE_BASE)
                         * (time[mask_f] - t_p2) / d3)

    return stage


def generate_stage(time: np.ndarray, Q: np.ndarray,
                   outlet_bed_z: float, outlet_width: float,
                   outlet_manning: float, outlet_slope: float) -> np.ndarray:
    """Dispatch to the selected stage generator."""
    kwargs = dict(
        Q=Q,
        outlet_bed_z=outlet_bed_z,
        outlet_width=outlet_width,
        outlet_manning=outlet_manning,
        outlet_slope=outlet_slope,
    )
    dispatch = {
        1: gen_stage_constant,
        2: gen_stage_rating_curve,
        3: gen_stage_sine,
        4: gen_stage_surge,
    }
    if STAGE_TYPE not in dispatch:
        raise ValueError(f"STAGE_TYPE must be 1-4, got {STAGE_TYPE}")
    return dispatch[STAGE_TYPE](time, **kwargs)


# ----------------------------------------------------------------
#  CHANNEL GEOMETRY BUILDER
# ----------------------------------------------------------------

def _parse_pairs(pairs: list, label: str) -> tuple:
    """Convert user pair list [[x,v],...] to sorted (x_array, v_array)."""
    arr = np.array(pairs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{label}: expected list of [x, value] pairs, got shape {arr.shape}"
        )
    idx = np.argsort(arr[:, 0])
    return arr[idx, 0], arr[idx, 1]


def build_channel_grid() -> pd.DataFrame:
    """
    Interpolate bed elevation, width, and Manning's n onto a uniform grid
    at spacing CHANNEL_DX.  Optionally smooth bed and/or width profiles.

    Returns a DataFrame with columns:
        Distance (m) | Bed Elevation (m) | Width (m) | Manning's n
    """
    x_bed, z_bed = _parse_pairs(BED_PAIRS,     "BED_PAIRS")
    x_wid, w_wid = _parse_pairs(WIDTH_PAIRS,   "WIDTH_PAIRS")
    x_man, n_man = _parse_pairs(MANNING_PAIRS, "MANNING_PAIRS")

    x_min, x_max = x_bed[0], x_bed[-1]
    if x_max <= x_min:
        raise ValueError(
            "BED_PAIRS: upstream distance must be less than downstream distance."
        )

    # Build uniform grid
    grid = np.arange(x_min, x_max + CHANNEL_DX, CHANNEL_DX)
    grid = np.clip(grid, x_min, x_max)
    if grid[-1] < x_max:
        grid = np.append(grid, x_max)   # ensure exact downstream endpoint

    # Bed elevation interpolation
    f_bed  = interp1d(x_bed, z_bed, kind="linear",
                      bounds_error=False, fill_value=(z_bed[0], z_bed[-1]))
    z_grid = f_bed(grid)

    # Width interpolation
    if len(x_wid) == 1:
        w_grid = np.full_like(grid, w_wid[0])
    else:
        f_wid  = interp1d(x_wid, w_wid, kind="linear",
                          bounds_error=False, fill_value=(w_wid[0], w_wid[-1]))
        w_grid = f_wid(grid)

    # Manning's n interpolation
    if len(x_man) == 1:
        n_grid = np.full_like(grid, n_man[0])
    else:
        f_man  = interp1d(x_man, n_man, kind="linear",
                          bounds_error=False, fill_value=(n_man[0], n_man[-1]))
        n_grid = f_man(grid)

    # Optional smoothing (expressed in metres, converted to grid cells)
    cells_bed = max(SMOOTH_BED_SIGMA   / CHANNEL_DX, 0.5)
    cells_wid = max(SMOOTH_WIDTH_SIGMA / CHANNEL_DX, 0.5)

    if SMOOTH_BED and SMOOTH_BED_SIGMA > 0:
        z_grid = gaussian_filter1d(z_grid, sigma=cells_bed)

    if SMOOTH_WIDTH and SMOOTH_WIDTH_SIGMA > 0:
        w_grid = gaussian_filter1d(w_grid, sigma=cells_wid)

    w_grid = np.maximum(w_grid, 0.0)   # floor width at zero

    return pd.DataFrame({
        "Distance (m)":      grid,
        "Bed Elevation (m)": z_grid,
        "Width (m)":         w_grid,
        "Manning's n":       n_grid,
    })


def get_outlet_properties(chan_df: pd.DataFrame) -> dict:
    """
    Extract hydraulic properties at the downstream outlet.
    Local bed slope is estimated from the last two grid nodes.
    """
    row_ds = chan_df.iloc[-1]
    row_us = chan_df.iloc[-2]

    dx    = row_ds["Distance (m)"]     - row_us["Distance (m)"]
    dz    = row_ds["Bed Elevation (m)"] - row_us["Bed Elevation (m)"]
    # Positive slope = bed descends in the downstream direction
    slope = -dz / dx if dx > 0 else 0.001

    return {
        "bed_z":   row_ds["Bed Elevation (m)"],
        "width":   row_ds["Width (m)"],
        "manning": row_ds["Manning's n"],
        "slope":   max(slope, 1e-6),   # guard against zero / adverse slope
    }


# ----------------------------------------------------------------
#  FILE OUTPUT
# ----------------------------------------------------------------

def _next_file_number(pattern_re, directory: str) -> int:
    """Return one more than the highest existing file number matching pattern."""
    max_num = 0
    for fname in os.listdir(directory):
        m = pattern_re.match(fname)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def save_flow_csv(time: np.ndarray, Q: np.ndarray,
                  stage: np.ndarray) -> tuple:
    """Write Flow_Data_##.csv to the working directory."""
    out_dir = os.getcwd()
    num   = _next_file_number(
        re.compile(r"Flow_Data_(\d+)\.csv", re.IGNORECASE), out_dir
    )
    fname = f"Flow_Data_{num:02d}.csv"
    fpath = os.path.join(out_dir, fname)

    df = pd.DataFrame({
        "Time (s)":             time,
        "Discharge (m3/s)":     Q,
        "Downstream Stage (m)": stage,
    })
    df.to_csv(fpath, index=False)
    return fpath, fname


def save_channel_csv(chan_df: pd.DataFrame) -> tuple:
    """Write Channel_Data_##.csv to the working directory."""
    out_dir = os.getcwd()
    num   = _next_file_number(
        re.compile(r"Channel_Data_(\d+)\.csv", re.IGNORECASE), out_dir
    )
    fname = f"Channel_Data_{num:02d}.csv"
    fpath = os.path.join(out_dir, fname)

    chan_df.to_csv(fpath, index=False)
    return fpath, fname


# ----------------------------------------------------------------
#  VISUALISATION
# ----------------------------------------------------------------

def plot_results(time: np.ndarray, Q: np.ndarray, stage: np.ndarray,
                 chan_df: pd.DataFrame,
                 flow_fname: str, chan_fname: str) -> None:
    """Create and save schematic JPEGs when `OUTPUT_SCHEMATICS` is True.

    - Flow schematic: time in minutes (x-axis), discharge (left y-axis), stage (right y-axis)
    - Channel schematic: profile (z vs x) and plan/map view (channel edges plotted vs x)
    """
    if not OUTPUT_SCHEMATICS:
        return

    # Matplotlib style for engineering schematic (serif font)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [SCHEMATIC_FONT],
        "font.size": 10,
    })

    # Derive JPG filenames from CSV filenames
    flow_jpg = os.path.splitext(flow_fname)[0] + ".jpg"
    chan_jpg = os.path.splitext(chan_fname)[0] + ".jpg"

    # --- Flow schematic ---
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    time_min = time / 60.0

    ax1.plot(time_min, Q, color="tab:blue", lw=1.4, label="Discharge (m^3/s)")
    ax1.fill_between(time_min, Q, color="tab:blue", alpha=0.15)
    ax1.set_xlabel("Elapsed Time (min)")
    ax1.set_ylabel("Discharge (m^3/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(time_min, stage, color="tab:red", lw=1.2, label="Stage (m)")
    ax2.set_ylabel("Stage (m)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Annotations: labels and grid
    ax1.grid(True, linestyle="--", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(os.getcwd(), flow_jpg), dpi=SCHEMATIC_DPI, format="jpg")
    plt.close(fig1)

    # --- Channel schematic (profile + plan) ---
    dist = chan_df["Distance (m)"].values
    z = chan_df["Bed Elevation (m)"].values
    w = chan_df["Width (m)"].values
    n = chan_df["Manning's n"].values

    fig2, (axp, axm) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1]})

    # Profile view
    axp.plot(dist, z, color="sienna", lw=1.6)
    axp.fill_between(dist, z, z.min() - abs(np.ptp(z)) * 0.05, color="sienna", alpha=0.12)
    axp.set_xlabel("Distance from Inlet (m)")
    axp.set_ylabel("Bed Elevation (m)")
    axp.set_title("Channel Profile (bed elevation)")
    axp.grid(True, linestyle="--", alpha=0.3)

    # Annotate end-to-end slope: compute overall bed slope (m/m)
    try:
        total_length = float(dist[-1] - dist[0])
        total_drop = float(z[0] - z[-1])
        bed_slope = total_drop / total_length if total_length > 0 else 0.0
        slope_text = f"Bed slope = {bed_slope:.5f} m/m ({bed_slope*100:.3f}%)"
        # dashed line between endpoints to visualise average slope
        axp.plot([dist[0], dist[-1]], [z[0], z[-1]], color="black", linestyle="--", linewidth=1.0,
                 label="End-to-end slope")
        axp.legend(loc="upper right", fontsize=8)
        # place slope text in axis coordinates
        axp.text(0.98, 0.02, slope_text, transform=axp.transAxes,
                 ha="right", va="bottom", fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    except Exception:
        pass

    # Annotate slope and representative Manning's n at a few positions
    try:
        isp = np.linspace(0, len(dist) - 1, min(6, len(dist))).astype(int)
        for i in isp:
            axp.text(dist[i], z[i], f"n={n[i]:.3f}", fontsize=8, ha="center", va="bottom")
    except Exception:
        pass

    # Plan/map view: channel edges as +/- width/2 around centerline y=0
    left_edge = w / 2.0
    right_edge = -w / 2.0
    axm.plot(dist, left_edge, color="teal", lw=1.2)
    axm.plot(dist, right_edge, color="teal", lw=1.2)
    axm.fill_between(dist, right_edge, left_edge, color="teal", alpha=0.12)
    axm.set_xlabel("Distance from Inlet (m)")
    axm.set_ylabel("Cross-stream position (m)")
    axm.set_title("Channel Plan View (edges, width)")
    axm.grid(True, linestyle="--", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(os.path.join(os.getcwd(), chan_jpg), dpi=SCHEMATIC_DPI, format="jpg")
    plt.close(fig2)


# ----------------------------------------------------------------
#  SUMMARY
# ----------------------------------------------------------------

# Summary printing removed per specification: suppress non-file outputs.


# ----------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------

def main():
    # Build channel grid (needed by rating-curve stage)
    chan_df = build_channel_grid()
    outlet  = get_outlet_properties(chan_df)

    # Generate discharge time series
    time = _build_time_array()
    Q    = generate_discharge(time)

    # Generate downstream stage time series
    stage = generate_stage(
        time, Q,
        outlet_bed_z   = outlet["bed_z"],
        outlet_width   = outlet["width"],
        outlet_manning = outlet["manning"],
        outlet_slope   = outlet["slope"],
    )

    # Write output files (CSV only)
    flow_path, flow_fname = save_flow_csv(time, Q, stage)
    chan_path, chan_fname  = save_channel_csv(chan_df)

    # Optionally create schematic JPEGs (no console output)
    if OUTPUT_SCHEMATICS:
        plot_results(time, Q, stage, chan_df, flow_fname, chan_fname)


if __name__ == "__main__":
    main()