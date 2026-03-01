"""
1D Open-Channel Flow Model — V2
================================

A teaching and engineering-analysis tool for studying how different
simplifications of the Saint-Venant (1-D shallow-water) equations affect
flood-wave propagation in a prismatic rectangular river channel.  The model
also includes a fully coupled multi-fraction sediment transport module.

Physical System
---------------
The model represents a straight, prismatic channel of rectangular cross-
section.  All spatial variation is along the streamwise coordinate x.
Transverse (y-direction) and vertical (z-direction) variations are
eliminated by depth-averaging and width-averaging the Navier-Stokes
equations — the central operation that produces the Saint-Venant system.

Assumptions of the Saint-Venant Equations
------------------------------------------
  1. Hydrostatic pressure distribution — vertical accelerations are small
     compared with gravity (valid when depth ≪ flood-wave wavelength).
  2. Uniform velocity over the cross-section — the momentum correction
     (Boussinesq) coefficient β ≈ 1.
  3. Slowly varying channel geometry — width and slope change gradually
     so that 1-D averaging is self-consistent.
  4. Incompressible flow — water density ρ_w = 1000 kg/m³ is constant.
  5. Friction is quasi-steady — the friction slope Sƒ is computed from
     the instantaneous velocity field via Manning's equation.

Governing Equations (full 1-D Saint-Venant)
--------------------------------------------
After depth-averaging, conservation of mass and momentum become:

  ∂A/∂t + ∂Q/∂x = 0                          … Continuity (Mass)

  ∂Q/∂t + ∂(Q²/A)/∂x + gA ∂y/∂x             … Momentum
       = gA (S₀ − Sƒ)

Variable definitions
  A  = flow cross-sectional area  [m²]   (A = B·y for a rectangle)
  Q  = volumetric discharge       [m³/s]
  y  = flow depth                 [m]
  S₀ = channel bed slope          [–]    (energy of position)
  Sƒ = friction slope (Manning)   [–]    (energy lost to friction)
  g  = gravitational acceleration [m/s²] = 9.81 m/s²
  B  = channel width              [m]

Momentum equation term-by-term
  ∂Q/∂t          – local (temporal) acceleration; dominant during rapid
                   rises and falls of the hydrograph.
  ∂(Q²/A)/∂x     – convective (spatial) acceleration; associated with
                   velocity changes along the channel.
  gA ∂y/∂x       – pressure-gradient force (drives backwater effects when
                   the water surface is not parallel to the bed).
  gA S₀          – gravitational driving force (bed slope × submerged weight).
  −gA Sƒ         – friction resistance (opposes flow; energy dissipation).

Solver Hierarchy (from simplest to most complete)
--------------------------------------------------
  KINEMATIC   – DROPS all momentum terms except gravity–friction balance:
                   S₀ = Sƒ   →   Q = f(A)   (Manning's rating curve).
                Q is a single-valued function of A; the wave travels at
                the kinematic celerity cₖ = (5/3)V.  Cannot model
                backwater or wave attenuation.

  DIFFUSIVE   – Drops the two INERTIA terms (∂Q/∂t and ∂(Q²/A)/∂x) but
                retains the pressure-gradient gA ∂y/∂x.  The resulting
                zero-inertia (parabolic) PDE in Q is:
                   ∂Q/∂t + cₖ ∂Q/∂x = D ∂²Q/∂x²
                where D = Q/(2BS₀) is the Hayami hydraulic diffusivity.
                Captures backwater and wave attenuation.  Implemented as
                the Muskingum-Cunge channel-routing scheme.

  DYNAMIC     – Retains every term → full Saint-Venant (hyperbolic PDE).
                Three numerical discretisations illustrate trade-offs:
                  • MacCormack     (2nd-order predictor-corrector;
                                    dispersive near shocks but accurate)
                  • Lax-Friedrichs  (1st-order central; strongly
                                    diffusive but always stable)
                  • HLL Riemann     (shock-capturing finite volume;
                                    industry-standard approach)

  COMPARE     – Runs all five solvers and produces a side-by-side
                comparison of WSE, velocity, and Froude-number profiles,
                plus an optional animation.

Coupled Sediment Transport
--------------------------
When SEDIMENT_TRANSPORT = True, each flow solver calls a fractional
Exner-equation module at every hydraulic time step.  Three bedload
transport formulas are available:
  • Meyer-Peter & Müller (1948)       – classic excess-Shields power law
  • Wilcock & Crowe (2003)            – surface-based; accounts for sand
                                        content effect on gravel mobility
  • Parker, Klingeman & McLean (1982) – hiding-function gravel transport

Bed-surface composition evolves through the Hirano (1971) active-layer
mixing model.  The Exner equation enforces sediment mass conservation:
   (1 − λ) ∂η/∂t + ∂q_b/∂x = 0
where η is bed elevation and λ is bed porosity.

File / Section Map
------------------
  Section 1  (line ~60)   – User configuration (all editable parameters)
  Section 2  (line ~235)  – Helper functions (grid, BCs, backwater profile)
  Section 2-B(line ~390)  – Sediment transport module
  Section 3  (line ~960)  – Solver engines (MacCormack, Lax, HLL, Kin., Diff.)
  Section 4  (line ~1700) – Post-processing and visualisation
  Section 5  (line ~2620) – Execution (COMPARE or single-solver)

Dependencies
------------
  numpy, pandas, matplotlib
  Optional: ffmpeg (for MP4 animation export)
"""

import os
import sys
import time as _timer
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ╔══════════════════════════════════════════════════════════════╗
# ║  1.  USER CONFIGURATION  — edit this section before running ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Solver Selection ────────────────────────────────────────────
# Choose which solver(s) to run.  In COMPARE mode all five run and
# results are plotted side-by-side, which is the best way to understand
# how each simplification affects the flood-wave shape and timing.
SOLVER_TYPE = 'COMPARE'
#   'DYNAMIC'      – MacCormack 2nd-order (full Saint-Venant)
#   'DYNAMIC_LAX'  – Lax-Friedrichs 1st-order (full Saint-Venant)
#   'DYNAMIC_HLL'  – HLL Riemann finite-volume (full Saint-Venant)
#   'KINEMATIC'    – Kinematic-wave approximation (S₀ = Sƒ)
#   'DIFFUSIVE'    – Diffusive wave / Muskingum-Cunge (zero inertia)
#   'COMPARE'      – all five solvers run together (recommended)

# ── Downstream Boundary-Condition Mode ─────────────────────────
# Subcritical flow (Fr < 1) requires one downstream BC.
# The BC controls whether the model imposes a fixed rating curve or
# a time-varying stage hydrograph at the outlet.
#   'NORMAL_DEPTH' – y_ds = Manning normal depth (rating-curve type).
#                    Appropriate when the downstream end is uncontrolled
#                    and the channel is hydraulically steep relative to
#                    the incoming flood wave.
#   'STAGE_TS'     – y_ds = stage column from the BC CSV file.
#                    Use when a downstream control (dam, tidal signal,
#                    confluence) sets the stage independently of Q.
BC_MODE      = 'NORMAL_DEPTH'
NORMAL_SLOPE = 0.01          # channel slope to use in Manning's equation
                             # for the NORMAL_DEPTH BC (usually = BED_SLOPE)

# ── Channel Geometry & Roughness ────────────────────────────────
# All solvers use a single prismatic rectangular channel.
# Resistance to flow is parameterised through Manning's n:
#   n ≈ 0.010–0.015  smooth concrete/glass flume
#   n ≈ 0.025–0.035  natural gravel-bed river (clean)
#   n ≈ 0.040–0.080  vegetation-choked or boulder-bed channel
LENGTH    = 1000.0           # channel length  L  [m]
WIDTH     = 10.0             # channel width   B  [m]
BED_SLOPE = 0.01             # longitudinal bed slope  S₀ [m/m]
                             # (steep for a natural channel; drives high Froude)
MANNING_N = 0.035            # Manning's roughness coefficient n [s/m^(1/3)]

# ── Numerical Parameters ─────────────────────────────────────────
# DX sets the spatial resolution.  Finer DX = more accuracy but
# proportionally more time steps (since dt ∝ dx via the CFL condition).
# Rule of thumb: ensure at least 20–50 computational nodes for smooth
# profile resolution (here 1000/20 = 51 nodes).
DX      = 20.0               # spatial grid step Δx  [m]
T_FINAL = 6000               # total simulation time  [s]

# ── Animation Switch ────────────────────────────────────────────
#   True  → render and save a longitudinal WSE animation as MP4
#            (requires ffmpeg on PATH; can be slow for long simulations)
#   False → skip animation entirely (recommended for quick testing)
SAVE_ANIMATION = False
ANIMATION_FPS  = 15          # frames per second in the saved MP4 file

# ── MacCormack / Lax / HLL Upstream Boundary Treatment ─────────
#
# Background — the subcritical inflow problem
#   For subcritical (Fr < 1) inflow, the theory of characteristics states
#   that only ONE boundary condition may be specified at the upstream end.
#   We prescribe Q (the hydrograph).  The depth A[0] must be *computed*
#   from the interior solution using the outgoing (C⁻) characteristic.
#
#   MacCormack, Lax-Friedrichs, and HLL all update only interior nodes
#   [1 … nx-2] in their flux steps, leaving A[0] frozen or set by a
#   simple neighbour copy.  Without correction this causes A[0] to drift
#   from the physical value, producing unrealistically high velocities
#   and an energy-grade-line spike at the inlet.
#
# Fix 1 — Riemann invariant (characteristic-compatible depth)
#   The C⁻ Riemann invariant  R⁻ = V − 2c  is conserved along the
#   characteristic dx/dt = V − c  (which travels UPSTREAM in subcritical
#   flow, i.e. it carries information from the interior to the boundary).
#   At node 0 we solve:
#       Q₀/(B·y₀) − 2√(g·y₀) = V₁ − 2·c₁   (= R⁻ evaluated at node 1)
#   via Newton-Raphson iteration for y₀.  This is the physically correct
#   inflow boundary treatment for subcritical shallow-water problems.
#
# Fix 2 — Sponge layer (gentle relaxation toward normal depth)
#   Even after the Riemann fix, small numerical artefacts can persist
#   in the first few cells due to the predictor's one-sided stencil.
#   The sponge layer blends A toward the local Manning normal depth with
#   an exponentially decaying weight:
#       w(i) = strength · exp(−3 i / N_sponge)
#   so that the boundary cell is most strongly corrected and the interior
#   is unaffected.  This does NOT alter the physics — it only smooths
#   the numerical transition at the inlet.
#
#   Set MC_UPSTREAM_FIX = False to disable both fixes (raw behaviour).
MC_UPSTREAM_FIX    = True
MC_SPONGE_CELLS    = 6          # number of inlet cells in the sponge layer
MC_SPONGE_STRENGTH = 0.25       # blending weight at the boundary (0–1; 0 = off)

# ── Output Plot Type ────────────────────────────────────────────
#   'PROFILE_PLOTS'   – longitudinal WSE, velocity, and Froude-number
#                       envelopes across the full channel length.
#   'MID_OBSERVATION' – stage and velocity time series + scatter
#                       comparison at the mid-channel observation point;
#                       useful for reproducing a virtual gauge record.
#   'SEDIMENT'        – comprehensive sediment analysis dashboard:
#                       bed change, rating curve, sedigraph, box plots.
OUTPUT_TYPE = 'PROFILE_PLOTS'

# ── Sediment Transport Module ───────────────────────────────────
#   When True, the model solves the Exner equation alongside each flow
#   solver and tracks multi-fraction bed composition.
#   When False, the model is hydraulics-only (faster).
SEDIMENT_TRANSPORT = False

# ── Bedload Transport Formula ───────────────────────────────────
#   Choose the empirical relationship between hydraulic driving force
#   and bed material transport rate:
#
#   'MPM'              – Meyer-Peter & Müller (1948)
#                        Classic excess-Shields power law:
#                          Φ = A · (τ* − τ*_c)^1.5
#                        Originally calibrated for uniform gravel.
#                        Appropriate for well-sorted, gravel-bed channels.
#                        Works for single or multiple grain-size fractions.
#
#   'WILCOCK_CROWE'    – Wilcock & Crowe (2003)
#                        Surface-based formula calibrated for sand/gravel
#                        mixtures (Oak Creek, Sandy River data sets).
#                        Accounts for the STRONG effect of sand content
#                        on gravel mobility: finer sand on the bed surface
#                        reduces hiding/protrusion of gravel, lowering the
#                        effective reference Shields stress for the mixture.
#                        Recommended for poorly-sorted gravel-bed rivers.
#
#   'PARKER_KLINGMAN'  – Parker, Klingeman & McLean (1982)
#                        Surface-based formula with a hiding function
#                        for gravel-bed rivers.  Uses a three-part Parker
#                        (1979) transport function and a power-law hiding
#                        exponent.  Good for gravel-dominated channels
#                        without significant sand content.
SED_FORMULA        = 'MPM'

# ── Sediment Physical Properties ────────────────────────────────
# Specific gravity (SG = ρ_s / ρ_w): quartz ≈ 2.65, feldspar ≈ 2.56,
# heavy minerals (magnetite, zircon) > 3.0.  This value appears in
# the Shields parameter (buoyant weight) and all bedload formulas.
SED_SPECIFIC_GRAVITY = 2.65      # specific gravity of sediment (quartz = 2.65)

# Erodible bed thickness: the depth of mobile sediment available for
# entrainment.  Once a cell is fully eroded to bedrock (bed_thick → 0),
# further erosion is suppressed.  A typical range is 0.5–5 m.
SED_BED_THICKNESS  = 2.0         # erodible bed-sediment thickness [m]

# Bed porosity λ: fraction of bed volume occupied by void space.
# Appears in the Exner equation as (1 − λ) to convert bulk bed volume
# change to solid-particle volume.
# Typical values: sand ≈ 0.35–0.40, gravel ≈ 0.30–0.38.
SED_POROSITY       = 0.40        # bed porosity λ [–]

# ── Grain-Size Distribution ─────────────────────────────────────
# Supply grain diameters in METRES, sorted ascending.
# Supply corresponding bed surface fractions that sum to 1.0.
#
# Grain-size classes are defined in phi (φ) or mm; convert to metres:
#   φ = −log₂(D/1mm) → φ=-6 = 64 mm (cobble), φ=0 = 1 mm (very coarse sand)
#
# Example — single fraction (classic MPM, e.g. uniform gravel):
#   SED_GRAIN_SIZES   = [0.002]           # D50 = 2 mm (coarse sand)
#   SED_BED_FRACTIONS = [1.0]             # 100 % of bed
#
# Example — four-fraction gravel/sand mix (typical gravel-bed river):
#   SED_GRAIN_SIZES   = [0.001, 0.004, 0.016, 0.064]   # 1, 4, 16, 64 mm
#   SED_BED_FRACTIONS = [0.20, 0.30, 0.30, 0.20]        # must sum to 1.0

SED_GRAIN_SIZES    = [0.001, 0.004, 0.016, 0.064]    # grain diameters [m]
SED_BED_FRACTIONS  = [0.20, 0.30, 0.30, 0.20]         # initial bed surface fractions (sum = 1)

# D50 is derived automatically from the log-interpolated grain-size
# distribution and used by the single-fraction MPM bedload path.
SED_D50            = None        # (set automatically below — do not edit)

# ── Sediment Inflow Boundary Condition ──────────────────────────
# Controls how much sediment is introduced at the upstream boundary.
# This is the single most important control on net erosion/deposition
# within the model domain:
#
#   'EQUILIBRIUM'   – inflow transport capacity = local hydraulic capacity
#                     at node 0, computed from the current flow.
#                     The upstream end supplies exactly what the flow could
#                     carry, so no net erosion/deposition occurs near the
#                     inlet — the channel evolves only where the flow
#                     diverges from equilibrium downstream.
#                     Use for sediment-supply-unlimited conditions.
#
#   'FEED_RATE'     – user-specified fractional feed per grain-size class
#                     (SED_FEED_FRACTIONS × total capacity at node 0).
#                     Allows selective coarsening or fining of the feed:
#                     e.g., supplying only coarse gravel forces fining
#                     in the reach.
#
#   'CONCENTRATION' – bulk inflow concentration  C_in [kg/m³].
#                     Total bedload = C_in × Q / ρ_s.
#                     Use when field measurements are in concentration units.
SED_INFLOW_BC      = 'EQUILIBRIUM'
SED_INFLOW_CONC    = 0.5         # [kg/m³] — used only when SED_INFLOW_BC='CONCENTRATION'

# Per-fraction upstream feed fractions (used when SED_INFLOW_BC='FEED_RATE').
# Length must match SED_GRAIN_SIZES and values must sum to 1.0.
# If None, the current bed surface fractions Fi[0,:] are used.
SED_FEED_FRACTIONS = None        # e.g. [0.20, 0.30, 0.30, 0.20]

# ── Sediment Numerical Parameters ───────────────────────────────
# Bed-change smoothing: applies a mild discrete Laplacian to the raw
# Exner-equation bed-change field to suppress checkerboard oscillations
# that can arise from coarse spatial discretisation of the flux divergence.
# Range 0–0.1; 0 = off, 0.05 = moderate.  Does NOT change total volume.
SED_SMOOTHING      = 0.05        # bed-change smoothing coefficient (0–0.1)

# ── Sediment Spin-Up Delay ───────────────────────────────────────
# Bed change is suppressed until the simulation clock reaches this time.
# Purpose: allow the hydraulic solver to reach a quasi-steady state
# before sediment transport begins, avoiding spurious bed changes driven
# by the initial transient (backwater profile adjustment).
# Set to 0 to allow bed change from t = 0.
SED_START_TIME     = 1000.0      # bed change start time [s]  (0 = no delay)

# ── Validate & derive sediment helpers ──
_n_fractions = len(SED_GRAIN_SIZES)
assert len(SED_BED_FRACTIONS) == _n_fractions, \
    "SED_BED_FRACTIONS length must match SED_GRAIN_SIZES"
assert abs(sum(SED_BED_FRACTIONS) - 1.0) < 1e-6, \
    "SED_BED_FRACTIONS must sum to 1.0"
SED_GRAIN_SIZES   = np.array(SED_GRAIN_SIZES,   dtype=float)
SED_BED_FRACTIONS = np.array(SED_BED_FRACTIONS, dtype=float)

if SED_FEED_FRACTIONS is not None:
    SED_FEED_FRACTIONS = np.array(SED_FEED_FRACTIONS, dtype=float)
    assert len(SED_FEED_FRACTIONS) == _n_fractions, \
        "SED_FEED_FRACTIONS length must match SED_GRAIN_SIZES"
    assert abs(SED_FEED_FRACTIONS.sum() - 1.0) < 1e-6, \
        "SED_FEED_FRACTIONS must sum to 1.0"

# Compute D50 from the user-defined grain-size distribution (log-interpolation
# of the cumulative fraction curve).
if SED_D50 is None:
    _cum = np.cumsum(SED_BED_FRACTIONS)
    if _n_fractions == 1:
        SED_D50 = SED_GRAIN_SIZES[0]
    else:
        SED_D50 = float(np.interp(0.5, _cum - SED_BED_FRACTIONS / 2.0,
                                   np.log(SED_GRAIN_SIZES)))
        SED_D50 = np.exp(SED_D50)

# --- BC File ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _select_bc_file():
    """Prompt for a BC file number; auto-select when non-interactive."""
    import re
    pattern = re.compile(r'BC_(\d+)\.csv', re.IGNORECASE)
    available = sorted(
        int(m.group(1))
        for f in os.listdir(SCRIPT_DIR)
        if (m := pattern.match(f))
    )
    if not sys.stdin.isatty():
        if available:
            num = available[0]
            print(f"(non-interactive) Auto-selecting BC_{num}.csv")
            return os.path.join(SCRIPT_DIR, f'BC_{num}.csv')
        print("No BC_*.csv files found in", SCRIPT_DIR)
        sys.exit(1)
    if available:
        print(f"Available BC files: {', '.join(f'BC_{n}.csv' for n in available)}")
    print("Enter the integer number for the BC file (e.g., 1 for BC_1.csv).")
    bc_num = input('Which BC file? ').strip()
    return os.path.join(SCRIPT_DIR, f'BC_{bc_num}.csv')


CSV_PATH = _select_bc_file()
print(f"Loading BC file: {CSV_PATH}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  2.  HELPER FUNCTIONS                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def get_normal_depth(Q, B, n, S):
    """
    Compute normal depth for a rectangular channel using Newton-Raphson.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    Normal depth y_n is the flow depth at which the gravitational
    driving force exactly balances friction resistance, i.e. uniform
    flow.  It satisfies Manning's equation implicitly:

        Q = (1/n) · A · R^(2/3) · √S₀

    where for a rectangle:
        A = B · y          (flow area)
        P = B + 2y         (wetted perimeter)
        R = A / P          (hydraulic radius)

    Newton-Raphson iteration
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Define  f(y) = (1/n) · A(y) · R(y)^(2/3) · √S − Q = 0.
    The analytical derivative f'(y) is used for fast quadratic convergence.
    For a rectangular channel:
        df/dy = (1/n) · √S · [ (5/3) B R^(2/3) − (4/3) R^(5/3) ]

    Convergence is reached when |Δy| < 1e-7 m (within 60 iterations).

    Parameters
    ----------
    Q : float  – discharge [m³/s]
    B : float  – channel width [m]
    n : float  – Manning's n [s/m^(1/3)]
    S : float  – energy slope (= bed slope for uniform flow) [m/m]

    Returns
    -------
    y_n : float – normal depth [m]
    """
    if S <= 0:
        return 1.0
    y = 1.0
    sqrt_S = S ** 0.5
    inv_n  = 1.0 / n
    for _ in range(60):
        A   = B * y
        P   = B + 2.0 * y
        R   = A / P
        R23 = R ** (2.0 / 3.0)
        f   = inv_n * A * R23 * sqrt_S - Q
        df  = inv_n * sqrt_S * (
            (5.0 / 3.0) * B * R23 - (4.0 / 3.0) * R ** (5.0 / 3.0)
        )
        if abs(df) < 1e-14:
            break
        y_new = y - f / df
        if abs(y_new - y) < 1e-7:
            break
        y = max(y_new, 1e-4)
    return y


def _load_bc_data(csv_file, bc_type):
    """
    Load upstream (and optionally downstream) boundary-condition data
    from a CSV file produced by GenBC_1DFlowModel_V2.py.

    Expected CSV columns (by position):
      col 0 – elapsed time  [s]
      col 1 – upstream discharge Q  [m³/s]
      col 2 – downstream stage WSE  [m absolute]  (used with STAGE_TS only)

    The Q time series is the upstream inflow hydrograph — it drives the
    entire simulation.  The stage column is only read when bc_type =
    'STAGE_TS'; all other modes use normal depth as the downstream BC.

    Fallback behaviour
    ~~~~~~~~~~~~~~~~~~
    If the CSV cannot be read (missing, wrong format, permission error),
    the function falls back to a synthetic sinusoidal hydrograph:
        Q(t) = 50 + 20 · sin(2π t / 3600)  [m³/s]
    This allows the model to run even without a valid BC file, which is
    useful for classroom demonstrations.

    Returns
    -------
    t_data     : (N,) time array [s]
    Q_data     : (N,) discharge array [m³/s]
    stage_data : (N,) downstream stage array [m] (zeros if NORMAL_DEPTH)
    """
    bc_type = bc_type.upper()
    try:
        df = pd.read_csv(csv_file)
        t_data = df.iloc[:, 0].values.astype(float)
        Q_data = df.iloc[:, 1].values.astype(float)
        if bc_type == 'STAGE_TS' and df.shape[1] >= 3:
            stage_data = df.iloc[:, 2].values.astype(float)
            # If stage column contains -999 placeholders, fall back to normal depth
            if np.all(stage_data < -900):
                print("Warning: stage column contains placeholders (-999).")
                print("  → computing normal-depth stage from Q and channel geometry.")
                stage_data = _compute_fallback_stage(Q_data)
            else:
                print("Hydrograph and Stage data loaded successfully.")
        else:
            stage_data = np.zeros_like(t_data)
            if bc_type == 'STAGE_TS':
                print("Warning: STAGE_TS selected but column 3 missing; computing from Manning's.")
                stage_data = _compute_fallback_stage(Q_data)
    except Exception as e:
        print(f"Error loading '{csv_file}': {e}.  Using synthetic fallback.")
        t_data     = np.linspace(0, 10000, 100)
        Q_data     = 50.0 + 20.0 * np.sin(2.0 * np.pi * t_data / 3600.0)
        stage_data = _compute_fallback_stage(Q_data)
    return t_data, Q_data, stage_data


def _compute_fallback_stage(Q_array):
    """
    Compute downstream stage from Q using Manning's equation.

    Used when the BC file does not contain a stage column, or when
    the stage column contains placeholder values (−999).

    The downstream bed elevation is fixed at:
        z_bed_ds = −10.0 − S₀ · L

    (The upstream datum is −10 m absolute, so the bed drops linearly
    from −10 m at x = 0 to −10 − S₀·L at x = L.)

    Stage  =  z_bed_ds  +  y_n(Q)

    where y_n is the Manning normal depth for each Q in Q_array.
    """
    z_bed_ds = -10.0 - BED_SLOPE * LENGTH
    depths   = np.array([get_normal_depth(max(q, 0.001), WIDTH, MANNING_N, BED_SLOPE)
                         for q in Q_array])
    return z_bed_ds + depths


def _setup_grid(L, dx, S0):
    """
    Create the 1-D computational grid.

    Grid layout
    ~~~~~~~~~~~
    Nodes are uniformly spaced at interval Δx from x = 0 (upstream)
    to x = L (downstream):
        nx = int(L / dx) + 1      (includes both end nodes)
        x  = [0, Δx, 2Δx, … L]

    Bed elevation
    ~~~~~~~~~~~~~
    The bed drops linearly at slope S₀:
        z_bed(x) = −10.0 − S₀ · x

    The upstream datum is −10 m (absolute).  This arbitrary choice
    keeps all elevations clearly negative (and therefore distinguishable
    from zero depths or zero velocities in debugging outputs).

    Returns
    -------
    nx    : int   – number of computational nodes
    x     : (nx,) – streamwise coordinate array [m]
    z_bed : (nx,) – bed elevation array [m absolute]
    """
    nx    = int(L / dx) + 1
    x     = np.linspace(0, L, nx)
    z_bed = -10.0 - S0 * x          # bed drops linearly at slope S₀
    return nx, x, z_bed


def _downstream_depth(bc_type, Q_last, B, n_manning, slope_bc,
                      t, t_data, stage_data, z_bed_ds):
    """
    Return the flow depth at the downstream boundary node.

    For subcritical flow, the downstream boundary requires ONE
    prescribed condition.  Two modes are supported:

    NORMAL_DEPTH (rating-curve BC)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The depth is the Manning normal depth for the current discharge:
        y_ds = y_n(Q)
    This is equivalent to assuming the reach exits into a long, uniform
    channel at the same slope and roughness.  It is the standard open
    BC for uncontrolled river reaches.

    STAGE_TS (prescribed stage)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The downstream stage is read from the BC CSV and interpolated to
    the current time.  The depth is obtained by subtracting the
    (fixed) bed elevation at the downstream node:
        y_ds = stage(t) − z_bed_ds
    Use when a downstream control structure, tidal boundary, or
    known stage-discharge rating is available.

    Returns
    -------
    y_ds : float – flow depth at downstream node [m]
    """
    if bc_type == 'NORMAL_DEPTH':
           return get_normal_depth(abs(Q_last), B, n_manning, slope_bc)
    # STAGE_TS — stage is an absolute WSE; convert to depth
    stage_val = np.interp(t, t_data, stage_data)
    depth = stage_val - z_bed_ds
    return max(depth, 0.1)


def _backwater_profile(nx, B, S0, n_manning, Q0, y_downstream, dx, g=9.81):
    """
    Compute an initial steady-state backwater (gradually-varied flow)
    profile using the Standard-Step method.

    Purpose
    ~~~~~~~
    Before the unsteady simulation begins the channel must be initialised
    to a physically consistent (non-zero, smooth) flow profile.  Starting
    from uniform flow everywhere would be fine for mild slopes, but for
    channels with a downstream control that imposes a depth different from
    normal depth, the backwater profile is a better initial condition
    because it avoids the initial transient caused by the mismatch between
    the prescribed downstream BC and a uniform-flow IC.

    GVF (Gradually Varied Flow) equation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The Saint-Venant momentum equation in steady state (∂Q/∂t = 0,
    ∂(Q²/A)/∂x ≈ 0 for slowly varying geometry) reduces to:

        dy/dx = (S₀ − Sƒ) / (1 − Fr²)

    This is integrated numerically by marching UPSTREAM from the known
    downstream depth y_ds, using a simple Euler step:
        y[i] = y[i+1] − dy/dx · Δx

    The denominator (1 − Fr²) becomes singular at critical flow (Fr = 1);
    a guard value of 1e-8 is applied to prevent division by zero.

    Parameters
    ----------
    nx            : int   – number of computational nodes
    B             : float – channel width [m]
    S0            : float – bed slope [m/m]
    n_manning     : float – Manning's n [s/m^(1/3)]
    Q0            : float – steady discharge [m³/s] (uniform along reach)
    y_downstream  : float – known depth at the downstream node [m]
    dx            : float – spatial step [m]
    g             : float – gravitational acceleration [m/s²]

    Returns
    -------
    y : (nx,) depth array [m]
    A : (nx,) area array [m²]
    Q : (nx,) discharge array [m³/s]  (constant = Q0)
    """
    y = np.empty(nx)
    y[-1] = y_downstream
    for i in range(nx - 2, -1, -1):
        yi  = y[i + 1]
        Ai  = B * yi
        Vi  = Q0 / Ai
        Ri  = Ai / (B + 2.0 * yi)
        Sfi = (n_manning ** 2 * Vi ** 2) / (Ri ** (4.0 / 3.0))
        Fr2 = Vi ** 2 / (g * yi)
        denom = 1.0 - Fr2
        if abs(denom) < 1e-8:
            denom = 1e-8       # guard against critical flow singularity
        dydx = (S0 - Sfi) / denom
        y[i] = max(y[i + 1] - dydx * dx, 0.1)
    A = y * B
    Q = np.full(nx, Q0)
    return y, A, Q


# ╔══════════════════════════════════════════════════════════════╗
# ║  2-B.  SEDIMENT TRANSPORT MODULE  (Multi-Fraction)           ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Conceptual framework for geomorphology students
# ------------------------------------------------
# Bedload transport occurs when boundary shear stress τ_bed exceeds a
# critical (threshold) value τ_c.  The dimensionless ratio of driving
# shear to resisting gravitational force is the Shields parameter τ*:
#
#   τ* = τ_bed / [(ρ_s − ρ_w) g D]
#
# where (ρ_s − ρ_w)gD is the buoyant weight per unit bed area of a
# grain of diameter D.  The Shields (1936) diagram shows that for
# well-sorted sand and gravel, the critical Shields parameter τ*_c ≈ 0.047
# (though it varies from ~0.03 to ~0.06 depending on grain Reynolds number).
#
# All three bedload formulas in this module:
#   1. Compute τ_bed from the flow field via Manning's equation.
#   2. Translate τ_bed into a dimensionless transport rate (Einstein
#      number Φ or Wilcock-Crowe/Parker W*).
#   3. Back-convert to a volumetric flux q_b [m²/s per unit width].
#
# Multi-fraction extension
# -------------------------
# Real rivers carry mixtures of grain sizes.  The transport rate of each
# fraction k depends on:
#   (a) the fraction's own Shields parameter (depends on D_k), and
#   (b) the fraction's availability on the bed surface (F_k).
# Hiding functions account for the fact that coarse grains shelter fine
# grains (reducing their mobility) while fine grains in a mixture are
# more exposed than they would be alone.
#
# Bed composition evolution — the Hirano (1971) active-layer model
# -----------------------------------------------------------------
# As grains are eroded or deposited, the composition of the topmost
# layer of the bed (the "active layer" or "exchange layer") changes.
# Hirano's model tracks only the surface fractions F_k, assuming:
#   • Deposition: deposited material has the composition of the
#     incoming transport flux.
#   • Erosion: the underlying substrate is exposed; its composition
#     is assumed equal to the current surface (substrate not tracked
#     separately in this implementation — a common simplification).
# Fractions are renormalised to sum to 1.0 after each mixing step.
#
# Exner equation (sediment mass conservation)
# -------------------------------------------
# The Exner (1920) equation ensures that sediment is conserved:
#
#   (1 − λ) ∂η/∂t = −∂q_b/∂x
#
# where η is bed elevation [m], λ is bed porosity (void fraction),
# and q_b is volumetric bedload transport per unit width [m²/s].
# A gradient of q_b (more sediment leaving a cell than entering it)
# causes the bed to lower (erosion), and vice versa.

# ── Hydraulic helpers (shared by all formulas) ──────────────────

def _sed_bed_shear(y, Q, B, n_manning, rho_w=1000.0, g=9.81):
    """
    Compute bed shear stress τ_bed [Pa] and friction slope Sƒ at every node.

    Derivation
    ~~~~~~~~~~
    Manning's equation gives the friction slope:
        Sƒ = n² V |V| / R^(4/3)

    where V = Q/A is the depth-averaged velocity and R = A/P is the
    hydraulic radius.  The absolute value of V is used so that Sƒ
    always opposes the flow direction.

    Bed shear stress is then:
        τ_bed = ρ_w g R |Sƒ|   [Pa]

    This is derived from the force balance on a control volume:
    the shear force per unit bed area equals the streamwise component
    of the pressure gradient times hydraulic radius.

    Engineering note
    ~~~~~~~~~~~~~~~~
    A minimum hydraulic radius of 1e-4 m is enforced to prevent
    division by zero in nearly-dry cells.

    Parameters
    ----------
    y        : (nx,) flow depth [m]
    Q        : (nx,) discharge [m³/s]
    B        : float channel width [m]
    n_manning: float Manning's n [s/m^(1/3)]

    Returns
    -------
    tau_bed : (nx,) bed shear stress [Pa]
    Sf      : (nx,) friction slope [–]  (signed: positive downstream)
    """
    A = B * np.maximum(y, 0.01)
    V = Q / A
    R = A / (B + 2.0 * np.maximum(y, 0.01))
    R = np.maximum(R, 1e-4)
    Sf = (n_manning ** 2 * V * np.abs(V)) / R ** (4.0 / 3.0)
    tau_bed = rho_w * g * R * np.abs(Sf)
    return tau_bed, Sf


def _sed_shields_parameter(y, Q, B, n_manning, D50, rho_w=1000.0, g=9.81):
    """
    Compute the dimensionless Shields parameter τ* at every node.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    The Shields parameter is the ratio of the hydrodynamic drag
    (bed shear stress) to the gravitational resistance of a grain:

        τ* = τ_bed / [(ρ_s − ρ_w) · g · D₅₀]

    Numerator   τ_bed = ρ_w g R Sƒ   — the drag that tends to move grains.
    Denominator (ρ_s − ρ_w) g D₅₀   — the buoyant weight of the grain per
                                        unit projected area (a measure of
                                        resistance to entrainment).

    The Shields (1936) threshold for grain motion:
        τ*_c ≈ 0.047   for fully turbulent, gravel-sized grains
    Grains move when τ* > τ*_c.

    Manning-based friction slope:
        Sƒ = n² V² / R^(4/3)

    Parameters
    ----------
    y        : (nx,) flow depth [m]
    Q        : (nx,) discharge [m³/s]
    B        : float channel width [m]
    n_manning: float Manning's n [s/m^(1/3)]
    D50      : float median grain diameter [m]

    Returns
    -------
    tau_star : (nx,) dimensionless Shields parameter [–]
    tau_bed  : (nx,) bed shear stress [Pa]
    Sf       : (nx,) friction slope [–]
    """
    rho_s = SED_SPECIFIC_GRAVITY * rho_w
    tau_bed, Sf = _sed_bed_shear(y, Q, B, n_manning, rho_w, g)
    tau_star = tau_bed / ((rho_s - rho_w) * g * D50)
    return tau_star, tau_bed, Sf


# ── 1. Meyer-Peter & Müller (1948) ──────────────────────────────

def _mpm_bedload(tau_star, D50, B, rho_w=1000.0, g=9.81,
                 tau_star_c=0.047, A_mpm=8.0):
    """
    Meyer-Peter & Müller (1948) bedload transport formula (single fraction).

    Historical context
    ~~~~~~~~~~~~~~~~~~
    MPM is the most widely used bedload formula in engineering practice.
    It was developed from flume experiments with uniform gravel and sand
    (0.4–29 mm) at ETH Zürich.  The formula expresses transport in terms
    of the EXCESS Shields parameter (τ* minus a critical threshold τ*_c),
    reflecting the idea that transport begins only above a threshold shear
    and increases steeply thereafter.

    Dimensionless transport rate (Einstein number Φ):
        Φ = A_mpm · (τ* − τ*_c)^1.5     for τ* > τ*_c
        Φ = 0                             otherwise

    Dimensional transport per unit width:
        q_b = Φ · √[(s−1) · g · D₅₀³]   [m²/s]

    where s = ρ_s / ρ_w is the specific gravity of sediment.

    Total cross-section bedload:
        Q_b = q_b · B                    [m³/s]  (of solid particles)

    Calibration constants
    ~~~~~~~~~~~~~~~~~~~~~
    A_mpm  = 8.0  (original MPM value, but ranges from 5–12 in literature)
    τ*_c   = 0.047  (Shields threshold for fully turbulent gravel)

    Applicability
    ~~~~~~~~~~~~~
    • Uniform or well-sorted gravel and sand (1–30 mm)
    • Does NOT account for hiding effects in poorly sorted mixtures
    • Works best when the bed surface is representative of the whole bed

    Returns
    -------
    Qb : (nx,) total volumetric bedload [m³/s]
    qb : (nx,) unit-width bedload [m²/s]
    """
    s = SED_SPECIFIC_GRAVITY
    excess = np.maximum(tau_star - tau_star_c, 0.0)
    Phi = A_mpm * excess ** 1.5
    qb = Phi * np.sqrt((s - 1.0) * g * D50 ** 3)   # m²/s per unit width
    Qb = qb * B                                     # m³/s total
    return Qb, qb


def _mpm_bedload_multifraction(tau_bed, grain_sizes, Fi, B,
                                rho_w=1000.0, g=9.81,
                                tau_star_c=0.047, A_mpm=8.0):
    """
    Multi-fraction Meyer-Peter & Müller bedload transport.

    Extension of MPM to size mixtures
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Each grain-size fraction k is treated independently using its own
    Shields parameter τ*_k based on its own diameter D_k:

        τ*_k = τ_bed / [(ρ_s − ρ_w) g D_k]

    The transport rate per fraction is weighted by the bed surface
    availability fraction Fᵢ_k (fraction of the bed area covered by
    grains of size D_k):

        Φ_k = A_mpm · (τ*_k − τ*_c)^1.5
        q_bk = Fᵢ_k · Φ_k · √[(s−1) g D_k³]

    Note: this "fractional MPM" does NOT include hiding corrections.
    Fine grains are treated as if they were equally exposed as coarse
    grains — which tends to OVERESTIMATE fine-fraction transport.
    For mixtures where hiding effects matter (poorly sorted gravel/sand),
    use WILCOCK_CROWE or PARKER_KLINGMAN instead.

    Parameters
    ----------
    tau_bed    : (nx,) bed shear stress [Pa]
    grain_sizes: (K,) grain diameters [m]
    Fi         : (nx, K) bed surface fractions at each node
    B          : channel width [m]

    Returns
    -------
    qb_k   : (nx, K) unit-width bedload per fraction [m²/s]
    qb_tot : (nx,) total unit-width bedload [m²/s]
    Qb_tot : (nx,) total volumetric bedload [m³/s]
    """
    s = SED_SPECIFIC_GRAVITY
    rho_s = s * rho_w
    K = len(grain_sizes)
    nx = len(tau_bed)
    qb_k = np.zeros((nx, K))

    for k in range(K):
        Dk = grain_sizes[k]
        tau_star_k = tau_bed / ((rho_s - rho_w) * g * Dk)
        excess = np.maximum(tau_star_k - tau_star_c, 0.0)
        Phi_k = A_mpm * excess ** 1.5
        qb_k[:, k] = Fi[:, k] * Phi_k * np.sqrt((s - 1.0) * g * Dk ** 3)

    qb_tot = qb_k.sum(axis=1)
    Qb_tot = qb_tot * B
    return qb_k, qb_tot, Qb_tot


# ── 2. Wilcock & Crowe (2003) ──────────────────────────────────

def _wilcock_crowe_bedload(tau_bed, grain_sizes, Fi, B,
                           rho_w=1000.0, g=9.81):
    """
    Wilcock & Crowe (2003) surface-based bedload transport.

    This formula was developed for poorly-sorted gravel/sand-bed rivers.
    It accounts for the strong influence of sand content on gravel
    mobility through a hiding/exposure correction.

    Key equations
    ~~~~~~~~~~~~~
    Dimensionless reference shear stress for the mean grain size:
        τ*_rm = 0.021 + 0.015 exp(−20 F_s)

    where F_s is the fraction of sand (D < 2 mm) on the bed surface.

    Per-fraction reference shear stress (hiding function):
        τ*_ri / τ*_rm = (D_i / D_sm)^b
        b = 0.67 / (1 + exp(1.5 − D_i / D_sm))

    Transport function:
        W*_i = 0.002 · φ^7.5               if φ < 1.35
        W*_i = 14 · (1 − 0.894/φ^0.5)^4.5  if φ ≥ 1.35
    where φ = τ_bed / τ_ri

    Volumetric transport per unit width:
        q_bi = F_i · W*_i · u*³ / [(s−1)g]

    Parameters
    ----------
    tau_bed    : (nx,) bed shear stress [Pa]
    grain_sizes: (K,) grain diameters [m]
    Fi         : (nx, K) bed surface fractions at each node
    B          : channel width [m]

    Returns
    -------
    qb_k   : (nx, K) unit-width bedload per fraction [m²/s]
    qb_tot  : (nx,) total unit-width bedload [m²/s]
    Qb_tot  : (nx,) total volumetric bedload [m³/s]
    """
    s = SED_SPECIFIC_GRAVITY
    rho_s = s * rho_w
    K = len(grain_sizes)
    nx = len(tau_bed)
    qb_k = np.zeros((nx, K))

    # Shear velocity  u* = sqrt(τ / ρ_w)
    u_star = np.sqrt(np.maximum(tau_bed, 0.0) / rho_w)   # (nx,)

    # Sand fraction on the bed surface  (D < 2 mm = 0.002 m)
    sand_mask = grain_sizes < 0.002
    Fs = Fi[:, sand_mask].sum(axis=1) if sand_mask.any() else np.zeros(nx)
    Fs = np.clip(Fs, 0.0, 1.0)

    # Surface geometric mean diameter  D_sm = exp(Σ Fᵢ ln Dᵢ)
    log_D = np.log(np.maximum(grain_sizes, 1e-8))
    Dsm = np.exp((Fi * log_D[np.newaxis, :]).sum(axis=1))   # (nx,)

    # Reference Shields number for the mean size
    tau_star_rm = 0.021 + 0.015 * np.exp(-20.0 * Fs)        # (nx,)

    # Dimensional reference shear stress for the mean size
    tau_rm = tau_star_rm * (rho_s - rho_w) * g * Dsm         # (nx,)

    for k in range(K):
        Dk = grain_sizes[k]
        ratio = Dk / np.maximum(Dsm, 1e-8)                  # D_i / D_sm

        # Hiding/exposure exponent
        b = 0.67 / (1.0 + np.exp(1.5 - ratio))              # (nx,)

        # Per-fraction reference shear stress
        tau_ri = tau_rm * ratio ** b                          # (nx,)

        # Dimensionless transport parameter φ = τ_bed / τ_ri
        phi = tau_bed / np.maximum(tau_ri, 1e-10)            # (nx,)

        # Wilcock-Crowe transport function W*_i
        W_star = np.where(
            phi < 1.35,
            0.002 * phi ** 7.5,
            14.0 * np.maximum(1.0 - 0.894 / np.sqrt(np.maximum(phi, 1e-10)), 0.0) ** 4.5
        )

        # Volumetric unit-width transport per fraction
        #   q_bi = F_i · W*_i · u*³ / [(s-1)g]
        qb_k[:, k] = Fi[:, k] * W_star * u_star ** 3 / ((s - 1.0) * g)

    qb_tot = qb_k.sum(axis=1)
    Qb_tot = qb_tot * B
    return qb_k, qb_tot, Qb_tot


# ── 3. Parker, Klingeman & McLean (1982) ───────────────────────

def _parker_klingman_bedload(tau_bed, grain_sizes, Fi, B,
                              rho_w=1000.0, g=9.81):
    """
    Parker, Klingeman & McLean (1982) bedload transport for gravel-bed
    rivers with multiple grain-size fractions.

    This surface-based approach uses a hiding function so that coarser
    grains are somewhat sheltered by finer ones, and finer grains are
    more exposed.

    Key equations
    ~~~~~~~~~~~~~
    Surface geometric mean diameter:
        D_sg = exp(Σ Fᵢ ln Dᵢ)

    Dimensionless bed shear stress scaled to the geometric mean:
        φ_sg0 = τ_bed / [(ρ_s − ρ_w) g D_sg]

    The bedload rating curve uses the Parker (1979) three-part function:

        φ₅₀ = φ_sg0 / φ*_r50              (φ*_r50 ≈ 0.0876)

    Transport intensity  W*(φ):
        W* = 0.0025 · φ^14                          φ < 0.95
        W* = exp[14.2 (φ − 1) − 9.28 (φ − 1)²]     0.95 ≤ φ ≤ 1.65
        W* = 11.2 · (1 − 0.822/φ)^4.5              φ > 1.65

    Per-fraction hiding function (straining):
        φ_i / φ_50 = (D_i / D_sg)^γ

    with the hiding exponent γ defined piecewise:
        γ = 0.982                                    D_i / D_sg ≤ 1
        γ = 0.982 (D_i / D_sg)^(−0.68)             D_i / D_sg > 1

    Per-fraction transport:
        q_bi = F_i · W*_i · u*³ / [(s−1)g]

    Parameters
    ----------
    tau_bed    : (nx,) bed shear stress [Pa]
    grain_sizes: (K,) grain diameters [m]
    Fi         : (nx, K) bed surface fractions at each node
    B          : channel width [m]

    Returns
    -------
    qb_k   : (nx, K) unit-width bedload per fraction [m²/s]
    qb_tot  : (nx,) total unit-width bedload [m²/s]
    Qb_tot  : (nx,) total volumetric bedload [m³/s]
    """
    s = SED_SPECIFIC_GRAVITY
    rho_s = s * rho_w
    K = len(grain_sizes)
    nx = len(tau_bed)
    qb_k = np.zeros((nx, K))

    u_star = np.sqrt(np.maximum(tau_bed, 0.0) / rho_w)

    # Surface geometric mean diameter
    log_D = np.log(np.maximum(grain_sizes, 1e-8))
    Dsg = np.exp((Fi * log_D[np.newaxis, :]).sum(axis=1))

    # Dimensionless shear stress based on D_sg
    tau_star_sg = tau_bed / ((rho_s - rho_w) * g * np.maximum(Dsg, 1e-8))

    # Reference dimensionless shear stress for D_sg  (Parker 1979)
    phi_r50 = 0.0876

    for k in range(K):
        Dk = grain_sizes[k]
        ratio = Dk / np.maximum(Dsg, 1e-8)          # D_i / D_sg

        # Hiding exponent γ (piecewise)
        gamma = np.where(ratio <= 1.0,
                         0.982,
                         0.982 * ratio ** (-0.68))

        # Per-fraction reference Shields number
        phi_ri = phi_r50 * ratio ** gamma

        # Normalised shear stress  φ_i = τ*_sg / φ_ri
        phi_i = tau_star_sg / np.maximum(phi_ri, 1e-10)

        # Parker three-part transport function W*(φ)
        W_star = np.piecewise(
            phi_i,
            [phi_i < 0.95,
             (phi_i >= 0.95) & (phi_i <= 1.65),
             phi_i > 1.65],
            [lambda p: 0.0025 * p ** 14,
             lambda p: np.exp(14.2 * (p - 1.0) - 9.28 * (p - 1.0) ** 2),
             lambda p: 11.2 * np.maximum(1.0 - 0.822 / np.maximum(p, 1e-10), 0.0) ** 4.5]
        )

        qb_k[:, k] = Fi[:, k] * W_star * u_star ** 3 / ((s - 1.0) * g)

    qb_tot = qb_k.sum(axis=1)
    Qb_tot = qb_tot * B
    return qb_k, qb_tot, Qb_tot


# ── Dispatch helper ─────────────────────────────────────────────

def _compute_bedload(formula, tau_bed, grain_sizes, Fi, B, y, Q, n_manning,
                     rho_w=1000.0, g=9.81):
    """
    Dispatch to the selected bedload formula and return consistent
    outputs regardless of which formula is chosen.

    Returns
    -------
    qb_k   : (nx, K) unit-width bedload per fraction [m²/s]
    qb_tot  : (nx,) total unit-width bedload [m²/s]
    Qb_tot  : (nx,) total volumetric bedload [m³/s]
    """
    K = len(grain_sizes)
    nx = len(tau_bed)

    if formula == 'WILCOCK_CROWE':
        return _wilcock_crowe_bedload(tau_bed, grain_sizes, Fi, B, rho_w, g)

    elif formula == 'PARKER_KLINGMAN':
        return _parker_klingman_bedload(tau_bed, grain_sizes, Fi, B, rho_w, g)

    else:  # 'MPM' (default)
        if K == 1:
            # Classic single-fraction path
            D50 = grain_sizes[0]
            rho_s = SED_SPECIFIC_GRAVITY * rho_w
            tau_star = tau_bed / ((rho_s - rho_w) * g * D50)
            Qb_tot, qb_tot = _mpm_bedload(tau_star, D50, B, rho_w, g)
            qb_k = qb_tot[:, np.newaxis]
            return qb_k, qb_tot, Qb_tot
        else:
            return _mpm_bedload_multifraction(tau_bed, grain_sizes, Fi, B,
                                               rho_w, g)


# ── Equilibrium / concentration helpers ─────────────────────────

def _sed_equilibrium_concentration(Qb, Q, B, y, rho_s=None, rho_w=1000.0):
    """
    Compute equilibrium (capacity) sediment concentration.

    C_eq = ρ_s · Q_b / Q   [kg/m³]
    """
    if rho_s is None:
        rho_s = SED_SPECIFIC_GRAVITY * rho_w
    Q_safe = np.maximum(np.abs(Q), 1e-6)
    C_eq = rho_s * Qb / Q_safe
    return C_eq


# ── State initialisation ───────────────────────────────────────

def _init_sediment_state(nx, z_bed, B, grain_sizes, bed_fractions):
    """
    Initialise multi-fraction sediment-transport state arrays.

    All sediment state is stored in a single dict that is passed
    between the hydraulic time-step loop and the sediment module.
    Storing state in a dict (rather than global variables) keeps the
    solvers self-contained and allows the COMPARE mode to run multiple
    independent instances simultaneously.

    State variables
    ~~~~~~~~~~~~~~~
    'eta'        : (nx,) current bed elevation [m absolute].
                   Mutable copy of the initial z_bed; changes as
                   erosion/deposition accumulates.

    'bed_thick'  : (nx,) remaining erodible thickness at each node [m].
                   Decremented on erosion; clamped at zero (bedrock floor).

    'Qb'         : (nx,) total volumetric bedload rate [m³/s].
                   Stored for plotting, even during spin-up.

    'Qb_k'       : (nx, K) per-fraction volumetric bedload [m³/s].

    'delta_eta'  : (nx,) cumulative bed change since t = 0 [m].
                   Positive = aggradation, negative = degradation.

    'Fi'         : (nx, K) bed surface grain-size fractions.
                   Initialised uniformly to the user-defined distribution;
                   evolves via the Hirano active-layer model.

    Returns
    -------
    sed : dict  – sediment state dictionary (modified in place each step)
    """
    K = len(grain_sizes)
    # Initialise bed surface fractions to the user-specified distribution
    Fi = np.tile(bed_fractions, (nx, 1)).astype(float)   # (nx, K)
    return {
        'eta':        z_bed.copy(),
        'bed_thick':  np.full(nx, SED_BED_THICKNESS),
        'Qb':         np.zeros(nx),
        'Qb_k':       np.zeros((nx, K)),
        'delta_eta':  np.zeros(nx),
        'Fi':         Fi,
    }


# ── Main sediment time step ────────────────────────────────────

def _sediment_step(sed, y, Q, B, n_manning, grain_sizes, dx, dt, nx,
                   sed_inflow_bc, sed_inflow_conc,
                   formula=None,
                   feed_fractions=None,
                   porosity=None, smoothing=None,
                   current_time=0.0,
                   g=9.81, rho_w=1000.0):
    """
    Advance the multi-fraction sediment module by one time step.

    Algorithm overview
    ~~~~~~~~~~~~~~~~~~
    The following steps are executed in sequence each hydraulic time step:

    Step 1 — Bed shear stress
        τ_bed is computed from the flow depth and velocity field using
        Manning's equation.  A depth floor prevents division by zero in
        nearly-dry cells (velocity ceiling V_MAX_SED = 10 m/s).

    Step 2 — Bedload capacity per fraction
        The selected formula (_compute_bedload) returns q_bk(nx, K):
        the unit-width bedload rate for each grain-size fraction at each
        node, computed from τ_bed and the current bed surface fractions Fi.

    Step 3 — Upstream boundary condition
        Controls how much sediment enters the domain at the inlet.
        See SED_INFLOW_BC options in Section 1 configuration.

    Step 4 — Downstream boundary condition
        Zero-gradient: q_bk[-1, :] = q_bk[-2, :].
        This assumes the outlet is free-draining and transport capacity
        is uniform near the exit.

    Step 5 — Exner equation (per fraction)
        The fractional Exner equation updates bed elevation:

            d_eta_k[i] = −Δt / (1 − λ) · ∂q_bk/∂x

        where the flux divergence ∂q_bk/∂x is computed with centred
        differences (interior nodes) and one-sided differences (ends).
        The total bed change is the sum over all fractions:
            d_eta = Σ_k d_eta_k

    Step 6 — Active-layer limiting (erosion cap)
        Erosion cannot exceed the available erodible thickness bed_thick.
        If the computed erosion exceeds bed_thick, all per-fraction changes
        are scaled down uniformly so that exactly bed_thick is eroded.

    Step 7 — Optional bed-change smoothing
        A mild discrete Laplacian (coefficient SED_SMOOTHING) is applied
        to d_eta to suppress numerical checkerboard instabilities.
        This does not change the total volume; it only redistributes
        the spatial pattern of bed change slightly.

    Step 8 — Apply bed change
        sed['eta']       += d_eta   (bed elevation)
        sed['delta_eta'] += d_eta   (cumulative change for plotting)
        sed['bed_thick'] -= d_eta   (remaining erodible thickness)

    Step 9 — Hirano active-layer mixing
        During deposition, the deposited material's grain-size composition
        is blended into the bed surface fractions Fi with a mixing rate
        limiter (mix = min(0.1·Δt, 0.5)).  During erosion, the surface
        composition is unchanged (the substrate is assumed to have the
        same composition as the current surface — no separate substrate
        tracking in this implementation).
        Fractions are renormalised after mixing.

    Parameters
    ----------
    sed           : dict  – sediment state dict from _init_sediment_state
                            (modified in-place)
    y, Q          : (nx,) flow depth [m] and discharge [m³/s]
    grain_sizes   : (K,)  grain diameters [m]
    formula       : str   – bedload formula ('MPM', 'WILCOCK_CROWE', etc.)
    feed_fractions: (K,)  upstream feed fractions (for FEED_RATE BC)
    current_time  : float – current simulation time [s] (for spin-up check)

    Returns
    -------
    z_bed : (nx,) updated bed elevation array [m]
    """
    if formula is None:
        formula = SED_FORMULA
    if porosity is None:
        porosity = SED_POROSITY
    if smoothing is None:
        smoothing = SED_SMOOTHING

    K = len(grain_sizes)
    Fi = sed['Fi']   # (nx, K)

    # 1. Sanitise depth (velocity ceiling)
    V_MAX_SED = 10.0
    y_sed = np.maximum(y, np.abs(Q) / (B * V_MAX_SED))
    y_sed = np.maximum(y_sed, 1e-3)

    # 2. Bed shear stress
    tau_bed, _ = _sed_bed_shear(y_sed, Q, B, n_manning, rho_w, g)

    # 3. Bedload capacity per fraction
    qb_k, qb_tot, Qb_tot = _compute_bedload(
        formula, tau_bed, grain_sizes, Fi, B, y_sed, Q, n_manning, rho_w, g
    )
    # qb_k: (nx, K) unit-width per fraction
    # qb_tot: (nx,) total unit-width
    # Qb_tot: (nx,) total volumetric

    # 4. Upstream boundary condition (per-fraction)
    rho_s = SED_SPECIFIC_GRAVITY * rho_w

    if sed_inflow_bc == 'EQUILIBRIUM':
        # Capacity at node 0 is already in qb_k[0, :] — no change needed.
        pass

    elif sed_inflow_bc == 'FEED_RATE':
        # Distribute total upstream capacity across user-specified feed fractions
        if feed_fractions is None:
            feed_fractions = Fi[0, :]   # fall back to bed fractions
        total_qb_in = qb_tot[0]
        qb_k[0, :] = feed_fractions * total_qb_in
        qb_tot[0] = qb_k[0, :].sum()
        Qb_tot[0] = qb_tot[0] * B

    else:   # 'CONCENTRATION'
        # Convert bulk concentration to total bedload, distribute by bed fractions
        Q_in = max(abs(Q[0]), 1e-6)
        Qb_in = sed_inflow_conc * Q_in / rho_s
        qb_in = Qb_in / B
        # Distribute across fractions using bed surface fractions at node 0
        qb_k[0, :] = Fi[0, :] * qb_in
        qb_tot[0] = qb_in
        Qb_tot[0] = Qb_in

    # 5. Downstream boundary: zero-gradient
    qb_k[-1, :] = qb_k[-2, :]
    qb_tot[-1]  = qb_k[-1, :].sum()
    Qb_tot[-1]  = qb_tot[-1] * B

    # 6. Record transport (even during spin-up) for plotting
    sed['Qb']   = Qb_tot.copy()
    sed['Qb_k'] = (qb_k * B).copy()   # store as volumetric per fraction

    if current_time < SED_START_TIME:
        return sed['eta']

    # 7. Exner equation per fraction → total bed change
    dq_dx_k = np.zeros((nx, K))
    for k in range(K):
        dq_dx_k[1:-1, k] = (qb_k[2:, k] - qb_k[:-2, k]) / (2.0 * dx)
        dq_dx_k[0, k]    = (qb_k[1, k]  - qb_k[0, k])   / dx
        dq_dx_k[-1, k]   = (qb_k[-1, k] - qb_k[-2, k])  / dx

    d_eta_k = -dt / (1.0 - porosity) * dq_dx_k       # (nx, K)
    d_eta   = d_eta_k.sum(axis=1)                      # total bed change

    # 8. Active-layer limiting
    for i in range(nx):
        if d_eta[i] < 0:
            max_erosion = sed['bed_thick'][i]
            if -d_eta[i] > max_erosion:
                scale = max_erosion / max(-d_eta[i], 1e-15)
                d_eta_k[i, :] *= scale
                d_eta[i] = -max_erosion

    # 9. Smoothing
    if smoothing > 0 and nx > 2:
        d_eta_smooth = d_eta.copy()
        d_eta_smooth[1:-1] += smoothing * (
            d_eta[2:] - 2.0 * d_eta[1:-1] + d_eta[:-2])
        d_eta = d_eta_smooth

    # 10. Apply bed change
    sed['eta']       += d_eta
    sed['delta_eta'] += d_eta
    sed['bed_thick'] -= d_eta
    sed['bed_thick']  = np.maximum(sed['bed_thick'], 0.0)

    # 11. Update bed surface fractions  (Hirano active-layer mixing)
    #     During deposition: new material has the composition of the
    #     incoming transport.  During erosion: the substrate is exposed
    #     (assumed to have the current surface composition — a common
    #     simplification when the substrate is not tracked separately).
    if K > 1:
        for i in range(nx):
            total_abs = np.sum(np.abs(d_eta_k[i, :]))
            if total_abs > 1e-12:
                if d_eta[i] > 0:
                    # Deposition: surface gains the deposited material's composition
                    dep_frac = np.maximum(d_eta_k[i, :], 0.0)
                    dep_sum = dep_frac.sum()
                    if dep_sum > 1e-15:
                        dep_frac /= dep_sum
                        mix = 0.1 * dt   # mixing rate limiter
                        mix = min(mix, 0.5)
                        Fi[i, :] = (1.0 - mix) * Fi[i, :] + mix * dep_frac
                # During erosion, surface composition stays (substrate ≈ surface)
        # Renormalise fractions
        row_sums = Fi.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-15)
        Fi[:] = Fi / row_sums

    sed['Fi'] = Fi
    return sed['eta']


# ╔══════════════════════════════════════════════════════════════╗
# ║  3.  SOLVER ENGINES                                         ║
# ╚══════════════════════════════════════════════════════════════╝

# ────────────────────────────────────────────────────────────────
#  3-A.  Dynamic Wave — MacCormack  (2nd-order)
# ────────────────────────────────────────────────────────────────

def solve_dynamic_wave(L, B, S0, n_manning, dx, t_end, csv_file,
                       bc_type, slope_bc, store_interval=30.0):
    """
    Full Saint-Venant solver using the MacCormack predictor-corrector scheme.

    Physics retained
    ~~~~~~~~~~~~~~~~
    ALL terms of the 1-D momentum equation are solved:
      ∂Q/∂t         – local acceleration (unsteady term)
      ∂(Q²/A)/∂x    – convective acceleration
      gA ∂y/∂x      – pressure gradient (backwater, M1/M2 profiles)
      gA S₀         – gravitational driving force
      −gA Sƒ        – friction (Manning's equation)

    Numerical method — MacCormack (1969)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MacCormack is an explicit predictor-corrector finite-difference scheme
    that achieves 2nd-order accuracy in both space and time:

      Predictor  (forward differences):
          U_p[i] = U[i] − Δt/Δx · (F[i+1] − F[i]) + Δt · S[i]

      Corrector  (backward differences):
          U_new[i] = ½·(U[i] + U_p[i]) − Δt/(2Δx) · (F_p[i] − F_p[i-1])
                   + ½·Δt · S_p[i]

    Here U = [A, Q]ᵀ is the state vector and F is the flux vector.
    Averaging the two half-steps cancels the leading-order truncation
    error, giving O(Δx², Δt²) accuracy.

    Artificial viscosity
    ~~~~~~~~~~~~~~~~~~~~
    MacCormack is dispersive at steep wave fronts — it produces
    Gibbs-type oscillations (wiggles) upstream and downstream of a
    hydraulic bore.  A small Laplacian dissipation term is added:
        Q_new[i] += ν·(Q[i+1] − 2Q[i] + Q[i-1])
    with ν = 0.10.  This damps the wiggles at the cost of a slight
    broadening of the shock (similar to how physical viscosity acts).
    Values ν < 0.05 were too weak; ν > 0.20 caused excessive smearing.

    CFL stability condition
    ~~~~~~~~~~~~~~~~~~~~~~~
    Explicit time-marching of hyperbolic PDEs requires the numerical
    domain of dependence to contain the physical domain of dependence
    (Courant-Friedrichs-Lewy criterion):

        Δt ≤ CFL · Δx / (|V| + c)

    where c = √(g·y) is the gravity-wave celerity.  CFL = 0.4 is used
    here (conservative to account for the non-linear terms and steep
    wave fronts common at S₀ = 0.01).

    When to use this solver
    ~~~~~~~~~~~~~~~~~~~~~~~
    MacCormack is appropriate for:
      • Research / teaching applications requiring high spatial accuracy
      • Channels where shock formation (hydraulic jumps) is expected
        but exact shock structure is less important than global accuracy
    The HLL scheme is generally preferred in practice for shock-capturing.
    """
    g = 9.81
    bc_type = bc_type.upper()
    nx, x, z_bed = _setup_grid(L, dx, S0)
    z_bed_ds = z_bed[-1]
    t_data, Q_data, stage_data = _load_bc_data(csv_file, bc_type)

    Q0   = np.interp(0.0, t_data, Q_data)
    y_ds = _downstream_depth(bc_type, Q0, B, n_manning, slope_bc,
                             0.0, t_data, stage_data, z_bed_ds)
    y, A, Q = _backwater_profile(nx, B, S0, n_manning, Q0, y_ds, dx, g)

    # Sediment transport state
    sed = _init_sediment_state(nx, z_bed, B, SED_GRAIN_SIZES, SED_BED_FRACTIONS) if SEDIMENT_TRANSPORT else None

    results    = []
    next_store = 0.0
    t = 0.0

    print(f"Starting Dynamic Wave (MacCormack, {bc_type}) …")
    while t < t_end:
        vel       = np.abs(Q / A)
        cel       = np.sqrt(g * A / B)
        max_speed = np.max(vel + cel)
        dt = min(0.4 * dx / max_speed, 2.0) if max_speed > 0 else 0.5
        if t + dt > t_end:
            dt = t_end - t

        # ── Predictor (forward differences) ──
        y_arr = A / B
        R     = A / (B + 2.0 * y_arr)
        Sf    = (n_manning ** 2 * Q * np.abs(Q)) / (
                 A ** 2 * np.maximum(R, 0.001) ** (4.0 / 3.0))

        dQ_dx   = np.diff(Q)            / dx
        dy_dx   = np.diff(y_arr)        / dx
        dQ2A_dx = np.diff(Q ** 2 / A)   / dx

        A_p = np.empty_like(A);  Q_p = np.empty_like(Q)
        A_p[:-1] = A[:-1] - dt * dQ_dx
        Q_p[:-1] = Q[:-1] - dt * (
            dQ2A_dx + g * A[:-1] * dy_dx - g * A[:-1] * (S0 - Sf[:-1]))

        y_down_p = _downstream_depth(bc_type, Q[-1], B, n_manning, slope_bc,
                                     t, t_data, stage_data, z_bed_ds)
        A_p[-1] = B * y_down_p
        Q_p[-1] = Q_p[-2]
        Q_p[0]  = np.interp(t, t_data, Q_data)
        A_p[0]  = A[0]

        # ── Corrector (backward differences) ──
        A_p  = np.maximum(A_p, 0.1)
        y_p  = A_p / B
        R_p  = A_p / (B + 2.0 * y_p)
        Sf_p = (n_manning ** 2 * Q_p * np.abs(Q_p)) / (
                A_p ** 2 * np.maximum(R_p, 0.001) ** (4.0 / 3.0))

        dQ_dx_p   = np.diff(Q_p)            / dx
        dy_dx_p   = np.diff(y_p)            / dx
        dQ2A_dx_p = np.diff(Q_p ** 2 / A_p) / dx

        A[1:] = 0.5 * (A[1:] + A_p[1:] - dt * dQ_dx_p)
        Q[1:] = 0.5 * (Q[1:] + Q_p[1:] - dt * (
            dQ2A_dx_p + g * A_p[1:] * dy_dx_p - g * A_p[1:] * (S0 - Sf_p[1:])))

        Q[0]  = np.interp(t + dt, t_data, Q_data)
        y_dwn = _downstream_depth(bc_type, Q[-1], B, n_manning, slope_bc,
                                  t + dt, t_data, stage_data, z_bed_ds)
        A[-1] = B * y_dwn

        # ── Upstream boundary depth fix (characteristic-compatible) ──
        #
        # For subcritical flow only Q is prescribed at the inlet; the
        # depth must be *computed*.  The standard MacCormack corrector
        # skips node 0 (A[1:] = ...), so without this fix A[0] would
        # stay frozen at its initial value while Q changes with the
        # hydrograph, producing unrealistically high velocities.
        #
        # Method – outgoing C⁻ Riemann invariant:
        #   The C⁻ characteristic travels from the interior toward the
        #   boundary, carrying the invariant  R⁻ = V − 2c.
        #   At node 0 we solve:  Q₀/(B·y₀) − 2√(g·y₀) = V₁ − 2·c₁
        #   via Newton-Raphson for y₀.
        if MC_UPSTREAM_FIX:
            _y1 = A[1] / B
            _V1 = Q[1] / A[1]
            _c1 = np.sqrt(g * _y1)
            _Rminus = _V1 - 2.0 * _c1   # outgoing Riemann invariant

            _y0 = _y1                    # initial guess = interior depth
            _Q0 = Q[0]
            for _it in range(25):
                _V0 = _Q0 / (B * _y0)
                _c0 = np.sqrt(g * _y0)
                _f  = _V0 - 2.0 * _c0 - _Rminus
                _df = -_Q0 / (B * _y0 ** 2) - np.sqrt(g / _y0)
                if abs(_df) < 1e-14:
                    break
                _y0_new = _y0 - _f / _df
                if abs(_y0_new - _y0) < 1e-8:
                    break
                _y0 = max(_y0_new, 0.01)

            # Safety: clamp to within ±50 % of normal depth to reject
            # rare non-physical Newton solutions (e.g. supercritical root).
            _yn0 = get_normal_depth(abs(_Q0), B, n_manning, S0)
            _y0  = np.clip(_y0, 0.5 * _yn0, 1.5 * _yn0)
            A[0] = B * _y0

        # Artificial viscosity (damps dispersive oscillations)
        # nu=0.10 gives adequate damping of Gibbs wiggles without over-diffusing;
        # 0.02 was too weak and let oscillations grow at steep wave fronts.
        nu = 0.10
        Q[1:-1] += nu * (Q[2:] - 2.0 * Q[1:-1] + Q[:-2])
        A[1:-1] += nu * (A[2:] - 2.0 * A[1:-1] + A[:-2])
        # Clamp: depth floor (0.1 m²) AND velocity ceiling (20 m/s);
        # the velocity ceiling prevents near-zero-area nodes from carrying
        # unrealistic velocities when Q is still finite.
        A = np.maximum(A, np.abs(Q) / 20.0)   # enforce V ≤ 20 m/s
        A = np.maximum(A, 0.1)                 # enforce minimum depth

        # ── Optional sponge layer ──
        #
        # Even after fixing A[0], the abrupt transition from a prescribed-
        # Q boundary to the interior can leave small residual oscillations
        # in the first few cells.  The sponge layer blends A toward the
        # local normal depth with an exponentially decaying weight:
        #   w(i) = strength · exp(−3 i / N_sponge)
        # This is gentle enough not to distort the interior but effectively
        # removes boundary artefacts from plots.
        if MC_UPSTREAM_FIX and MC_SPONGE_STRENGTH > 0:
            _n_sp = min(MC_SPONGE_CELLS, nx - 1)
            for _i in range(_n_sp):
                _w = MC_SPONGE_STRENGTH * np.exp(-3.0 * _i / _n_sp)
                _yn_i = get_normal_depth(abs(Q[_i]), B, n_manning, S0)
                A[_i] = (1.0 - _w) * A[_i] + _w * (B * _yn_i)

        # Coupled sediment transport step
        if sed is not None:
            z_bed = _sediment_step(sed, A / B, Q, B, n_manning, SED_GRAIN_SIZES,
                                   dx, dt, nx, SED_INFLOW_BC, SED_INFLOW_CONC,
                                   formula=SED_FORMULA,
                                   feed_fractions=SED_FEED_FRACTIONS,
                                   current_time=t)

        t += dt
        if t >= next_store:
            sed_snap = {k: v.copy() for k, v in sed.items()} if sed else None
            results.append((x.copy(), z_bed.copy(), (A / B).copy(), Q.copy(), t, sed_snap))
            next_store += store_interval

    print("  ✓ MacCormack complete.")
    return results


# ────────────────────────────────────────────────────────────────
#  3-B.  Dynamic Wave — Lax-Friedrichs  (1st-order)
# ────────────────────────────────────────────────────────────────

def solve_dynamic_wave_lax(L, B, S0, n_manning, dx, t_end, csv_file,
                           bc_type, slope_bc, store_interval=30.0):
    """
    Full Saint-Venant solver using the Lax-Friedrichs scheme.

    Physics retained
    ~~~~~~~~~~~~~~~~
    All Saint-Venant terms are retained — same as MacCormack.  The
    difference is entirely in the numerical discretisation.

    Numerical method — Lax-Friedrichs (1954)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The central value Uᵢ is replaced by the spatial average
    ½(Uᵢ₋₁ + Uᵢ₊₁) before applying central flux differences:

        U_new[i] = ½·(U[i-1] + U[i+1]) − Δt/(2Δx)·(F[i+1] − F[i-1])
                 + Δt·S[i]

    This introduces strong *numerical diffusion* (proportional to Δx²/Δt)
    that unconditionally stabilises the scheme but smears sharp wave fronts.

    Numerical diffusion vs. physical attenuation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Lax-Friedrichs artificially broadens and attenuates flood peaks even
    in a purely kinematic wave (which should have no attenuation).  This
    numerical diffusion can mimic flood routing attenuation — a critical
    distinction: the LF result looks like a physically diffused wave but
    the attenuation is an artefact of the discretisation, not physics.
    In COMPARE mode this is clearly visible: the LF peak is lower and
    later than the MacCormack or HLL peaks.

    Friction treatment
    ~~~~~~~~~~~~~~~~~~
    Friction is treated SEMI-IMPLICITLY to avoid the explicit-friction
    instability in shallow, fast-moving flows:
        Q_new = Q* / (1 + Δt · Kƒ)
    where Kƒ = g·n²·|Q| / (A·R^(4/3)) is the friction linearisation
    coefficient.  This avoids an algebraically negative Q from over-
    correction of friction in thin, fast flows.

    CFL = 0.6 (less restrictive than MacCormack because the averaging
    step introduces extra dissipation that helps stability).

    When to use this solver
    ~~~~~~~~~~~~~~~~~~~~~~~
    Lax-Friedrichs is useful as a pedagogical lower bound:
      • Its strong numerical diffusion shows how much a 1st-order scheme
        can distort a flood wave
      • Comparing it against MacCormack or HLL isolates the impact of
        numerical diffusion vs. physical dispersion
      • In production work it is generally superseded by HLL
    """
    g = 9.81
    bc_type = bc_type.upper()
    nx, x, z_bed = _setup_grid(L, dx, S0)
    z_bed_ds = z_bed[-1]
    t_data, Q_data, stage_data = _load_bc_data(csv_file, bc_type)

    Q0   = np.interp(0.0, t_data, Q_data)
    y_ds = _downstream_depth(bc_type, Q0, B, n_manning, slope_bc,
                             0.0, t_data, stage_data, z_bed_ds)
    _, A, Q = _backwater_profile(nx, B, S0, n_manning, Q0, y_ds, dx, g)

    # Sediment transport state
    sed = _init_sediment_state(nx, z_bed, B, SED_GRAIN_SIZES, SED_BED_FRACTIONS) if SEDIMENT_TRANSPORT else None

    results    = []
    next_store = 0.0
    t = 0.0

    print(f"Starting Dynamic Wave (Lax-Friedrichs, {bc_type}) …")
    while t < t_end:
        A = np.maximum(A, 0.1)
        vel       = np.abs(Q / A)
        cel       = np.sqrt(g * A / B)
        max_speed = np.max(vel + cel)
        dt = min(0.6 * dx / (max_speed + 1e-6), 1.0)
        if t + dt > t_end:
            dt = t_end - t

        F_mass = Q.copy()
        F_mom  = Q ** 2 / A + g * A ** 2 / (2.0 * B)

        A_avg   = 0.5 * (A[2:] + A[:-2])
        Q_avg   = 0.5 * (Q[2:] + Q[:-2])
        dF_mass = F_mass[2:] - F_mass[:-2]
        dF_mom  = F_mom[2:]  - F_mom[:-2]

        A_new = np.empty_like(A)
        Q_new = np.empty_like(Q)

        A_new[1:-1] = A_avg - 0.5 * (dt / dx) * dF_mass

        S_grav = g * A_avg * S0
        Q_star = Q_avg - 0.5 * (dt / dx) * dF_mom + dt * S_grav

        R_avg = np.maximum(A_avg / (B + 2.0 * A_avg / B), 0.01)
        Kf    = (g * n_manning ** 2 * np.abs(Q_avg)) / (A_avg * R_avg ** (4.0 / 3.0))
        Q_new[1:-1] = Q_star / (1.0 + dt * Kf)

        Q_new[0]  = np.interp(t + dt, t_data, Q_data)
        A_new[0]  = A_new[1]      # temporary — overwritten by Riemann fix below
        y_down    = _downstream_depth(bc_type, Q_new[-2], B, n_manning, slope_bc,
                                      t + dt, t_data, stage_data, z_bed_ds)
        A_new[-1] = B * y_down
        Q_new[-1] = Q_new[-2]

        A = np.maximum(A_new, 0.1)
        Q = Q_new

        # ── Upstream boundary depth fix (characteristic-compatible) ──
        # Lax-Friedrichs updates only interior nodes [1…nx-2]; A[0] is
        # set by neighbour-copy (A_new[1]) which drifts from physical
        # values when Q changes rapidly.  Replace with the C⁻ Riemann
        # invariant  R⁻ = V₁ − 2c₁  and solve for y₀ via Newton-Raphson.
        if MC_UPSTREAM_FIX:
            _y1 = A[1] / B
            _V1 = Q[1] / A[1]
            _c1 = np.sqrt(g * _y1)
            _Rminus = _V1 - 2.0 * _c1
            _y0 = _y1
            _Q0 = Q[0]
            for _it in range(25):
                _V0 = _Q0 / (B * _y0)
                _c0 = np.sqrt(g * _y0)
                _f  = _V0 - 2.0 * _c0 - _Rminus
                _df = -_Q0 / (B * _y0 ** 2) - np.sqrt(g / _y0)
                if abs(_df) < 1e-14:
                    break
                _y0_new = _y0 - _f / _df
                if abs(_y0_new - _y0) < 1e-8:
                    break
                _y0 = max(_y0_new, 0.01)
            _yn0 = get_normal_depth(abs(_Q0), B, n_manning, S0)
            _y0  = np.clip(_y0, 0.5 * _yn0, 1.5 * _yn0)
            A[0] = B * _y0

        # ── Optional sponge layer ──
        if MC_UPSTREAM_FIX and MC_SPONGE_STRENGTH > 0:
            _n_sp = min(MC_SPONGE_CELLS, nx - 1)
            for _i in range(_n_sp):
                _w = MC_SPONGE_STRENGTH * np.exp(-3.0 * _i / _n_sp)
                _yn_i = get_normal_depth(abs(Q[_i]), B, n_manning, S0)
                A[_i] = (1.0 - _w) * A[_i] + _w * (B * _yn_i)

        # Coupled sediment transport step
        if sed is not None:
            z_bed = _sediment_step(sed, A / B, Q, B, n_manning, SED_GRAIN_SIZES,
                                   dx, dt, nx, SED_INFLOW_BC, SED_INFLOW_CONC,
                                   formula=SED_FORMULA,
                                   feed_fractions=SED_FEED_FRACTIONS,
                                   current_time=t)

        t += dt
        if t >= next_store:
            sed_snap = {k: v.copy() for k, v in sed.items()} if sed else None
            results.append((x.copy(), z_bed.copy(), (A / B).copy(), Q.copy(), t, sed_snap))
            next_store += store_interval

    print("  ✓ Lax-Friedrichs complete.")
    return results


# ────────────────────────────────────────────────────────────────
#  3-C.  Dynamic Wave — HLL Riemann Solver  (Finite Volume)
# ────────────────────────────────────────────────────────────────

def solve_dynamic_wave_hll(L, B, S0, n_manning, dx, t_end, csv_file,
                           bc_type, slope_bc, store_interval=30.0):
    """
    Full Saint-Venant solver using the HLL approximate Riemann solver.

    Physics retained
    ~~~~~~~~~~~~~~~~
    All Saint-Venant terms — same as MacCormack and Lax-Friedrichs.

    Numerical method — Harten, Lax & van Leer (1983)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    HLL is a *finite-volume* Godunov-type method.  The domain is divided
    into cells; at each cell interface (i+½) the neighbouring cell values
    define a *local Riemann problem* (two constant states separated by a
    discontinuity).  HLL approximates the exact Riemann solution with a
    two-wave model (fastest left-going wave S_L, fastest right-going wave S_R):

        F_HLL = (S_R·F_L − S_L·F_R + S_L·S_R·(U_R − U_L)) / (S_R − S_L)

    The flux automatically reduces to the upwind (purely supersonic) flux
    when the entire wave fan lies on one side of the interface:
        if S_L ≥ 0 : F_HLL = F_L   (supersonic right-going flow)
        if S_R ≤ 0 : F_HLL = F_R   (supersonic left-going flow)

    Wave speed estimates (Davis bounds):
        S_L = min(V_L − c_L,  V_R − c_R)
        S_R = max(V_L + c_L,  V_R + c_R)

    These are guaranteed to enclose the true wave speeds, making HLL
    *entropy-stable* — it cannot generate spurious oscillations.

    Shock-capturing
    ~~~~~~~~~~~~~~~
    Unlike MacCormack (which disperses energy near shocks as wiggles),
    HLL resolves hydraulic bores and flood fronts as sharp, monotone
    transitions without artificial viscosity.  This is the key advantage
    of Godunov-type methods: the Riemann structure of the flux naturally
    handles wave interactions.

    Friction treatment
    ~~~~~~~~~~~~~~~~~~
    Same semi-implicit approach as Lax-Friedrichs: Q_new = Q*/(1 + Δt·Kƒ).

    Engineering significance
    ~~~~~~~~~~~~~~~~~~~~~~~~
    HLL (and its extensions HLLC, Roe) is the standard approach in modern
    production hydraulic codes (HEC-RAS 2-D, MIKE FLOOD, TUFLOW-FV,
    Iber) because of its robustness across sub-/super-critical transitions,
    wet-dry fronts, and dam-break problems — all without ad-hoc fixes.

    CFL = 0.5 (slightly more restrictive than Lax-Friedrichs but less
    diffusive, because HLL does not average neighbours).

    When to use this solver
    ~~~~~~~~~~~~~~~~~~~~~~~
    HLL is the best of the three dynamic-wave schemes for:
      • Near-critical or supercritical flow (Fr ≈ 1)
      • Problems with strong wave fronts or hydraulic bores
      • Any production/engineering analysis where accuracy matters
    """
    g = 9.81
    bc_type = bc_type.upper()
    nx, x, z_bed = _setup_grid(L, dx, S0)
    z_bed_ds = z_bed[-1]
    t_data, Q_data, stage_data = _load_bc_data(csv_file, bc_type)

    Q0   = np.interp(0.0, t_data, Q_data)
    y_ds = _downstream_depth(bc_type, Q0, B, n_manning, slope_bc,
                             0.0, t_data, stage_data, z_bed_ds)
    _, A, Q = _backwater_profile(nx, B, S0, n_manning, Q0, y_ds, dx, g)

    # Sediment transport state
    sed = _init_sediment_state(nx, z_bed, B, SED_GRAIN_SIZES, SED_BED_FRACTIONS) if SEDIMENT_TRANSPORT else None

    results    = []
    next_store = 0.0
    t = 0.0

    print(f"Starting Dynamic Wave (HLL, {bc_type}) …")
    while t < t_end:
        A = np.maximum(A, 0.1)
        y_depth = A / B
        V = Q / A
        c = np.sqrt(g * y_depth)

        max_speed = np.max(np.abs(V) + c)
        dt = min(0.5 * dx / (max_speed + 1e-6), 1.0)
        if t + dt > t_end:
            dt = t_end - t

        A_L, A_R = A[:-1], A[1:]
        Q_L, Q_R = Q[:-1], Q[1:]
        V_L, V_R = Q_L / A_L, Q_R / A_R
        c_L = np.sqrt(g * A_L / B)
        c_R = np.sqrt(g * A_R / B)

        S_L = np.minimum(V_L - c_L, V_R - c_R)
        S_R = np.maximum(V_L + c_L, V_R + c_R)

        F_mass_L = Q_L
        F_mom_L  = Q_L ** 2 / A_L + g * A_L ** 2 / (2.0 * B)
        F_mass_R = Q_R
        F_mom_R  = Q_R ** 2 / A_R + g * A_R ** 2 / (2.0 * B)

        den = np.maximum(S_R - S_L, 1e-6)
        Flux_mass = (S_R * F_mass_L - S_L * F_mass_R
                     + S_L * S_R * (A_R - A_L)) / den
        Flux_mom  = (S_R * F_mom_L  - S_L * F_mom_R
                     + S_L * S_R * (Q_R - Q_L)) / den

        sup_fwd = S_L >= 0
        Flux_mass = np.where(sup_fwd, F_mass_L, Flux_mass)
        Flux_mom  = np.where(sup_fwd, F_mom_L,  Flux_mom)

        sup_bck = S_R <= 0
        Flux_mass = np.where(sup_bck, F_mass_R, Flux_mass)
        Flux_mom  = np.where(sup_bck, F_mom_R,  Flux_mom)

        A_new = A.copy()
        A_new[1:-1] = A[1:-1] - (dt / dx) * (Flux_mass[1:] - Flux_mass[:-1])

        S_grav = g * A[1:-1] * S0
        Q_star = Q[1:-1] - (dt / dx) * (Flux_mom[1:] - Flux_mom[:-1]) + dt * S_grav

        R  = np.maximum(A[1:-1] / (B + 2.0 * A[1:-1] / B), 0.01)
        Kf = (g * n_manning ** 2 * np.abs(Q[1:-1])) / (A[1:-1] * R ** (4.0 / 3.0))

        Q_new = Q.copy()
        Q_new[1:-1] = Q_star / (1.0 + dt * Kf)

        Q_new[0]  = np.interp(t + dt, t_data, Q_data)
        A_new[0]  = A_new[1]      # temporary — overwritten by Riemann fix below
        y_down    = _downstream_depth(bc_type, Q[-2], B, n_manning, slope_bc,
                                      t + dt, t_data, stage_data, z_bed_ds)
        A_new[-1] = B * y_down
        Q_new[-1] = Q_new[-2]

        A = np.maximum(A_new, 0.1)
        Q = Q_new

        # ── Upstream boundary depth fix (characteristic-compatible) ──
        # HLL only updates interior cells [1…nx-2]; A[0] is set by
        # neighbour-copy (A_new[1]) which drifts from physical values
        # when Q changes rapidly.  Replace with the C⁻ Riemann invariant
        # R⁻ = V₁ − 2c₁ and solve for y₀ via Newton-Raphson.
        if MC_UPSTREAM_FIX:
            _y1 = A[1] / B
            _V1 = Q[1] / A[1]
            _c1 = np.sqrt(g * _y1)
            _Rminus = _V1 - 2.0 * _c1
            _y0 = _y1
            _Q0 = Q[0]
            for _it in range(25):
                _V0 = _Q0 / (B * _y0)
                _c0 = np.sqrt(g * _y0)
                _f  = _V0 - 2.0 * _c0 - _Rminus
                _df = -_Q0 / (B * _y0 ** 2) - np.sqrt(g / _y0)
                if abs(_df) < 1e-14:
                    break
                _y0_new = _y0 - _f / _df
                if abs(_y0_new - _y0) < 1e-8:
                    break
                _y0 = max(_y0_new, 0.01)
            _yn0 = get_normal_depth(abs(_Q0), B, n_manning, S0)
            _y0  = np.clip(_y0, 0.5 * _yn0, 1.5 * _yn0)
            A[0] = B * _y0

        # ── Optional sponge layer ──
        if MC_UPSTREAM_FIX and MC_SPONGE_STRENGTH > 0:
            _n_sp = min(MC_SPONGE_CELLS, nx - 1)
            for _i in range(_n_sp):
                _w = MC_SPONGE_STRENGTH * np.exp(-3.0 * _i / _n_sp)
                _yn_i = get_normal_depth(abs(Q[_i]), B, n_manning, S0)
                A[_i] = (1.0 - _w) * A[_i] + _w * (B * _yn_i)

        # Coupled sediment transport step
        if sed is not None:
            z_bed = _sediment_step(sed, A / B, Q, B, n_manning, SED_GRAIN_SIZES,
                                   dx, dt, nx, SED_INFLOW_BC, SED_INFLOW_CONC,
                                   formula=SED_FORMULA,
                                   feed_fractions=SED_FEED_FRACTIONS,
                                   current_time=t)

        t += dt
        if t >= next_store:
            sed_snap = {k: v.copy() for k, v in sed.items()} if sed else None
            results.append((x.copy(), z_bed.copy(), (A / B).copy(), Q.copy(), t, sed_snap))
            next_store += store_interval

    print("  ✓ HLL complete.")
    return results


# ────────────────────────────────────────────────────────────────
#  3-D.  Kinematic Wave
# ────────────────────────────────────────────────────────────────

def solve_kinematic_wave(L, B, S0, n_manning, dx, t_end, csv_file,
                         store_interval=30.0):
    """
    Kinematic-wave approximation of the Saint-Venant equations.

    Physics retained
    ~~~~~~~~~~~~~~~~
    Only the *continuity* equation is solved in its original form:
        ∂A/∂t + ∂Q/∂x = 0

    Physics DROPPED (entire momentum equation replaced)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The momentum equation is replaced by the kinematic assumption:
        S₀ = Sƒ   (gravity = friction, always in balance)

    This collapses Q to a single-valued function of A via Manning's:
        Q = (1/n) · A · R^(2/3) · √S₀

    Consequently:
      • No local or convective acceleration → no inertia effects
      • No pressure gradient term → no backwater, no rating-curve shift
      • The flow cannot "feel" downstream conditions at all

    Kinematic wave speed (Seddon celerity)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For a wide rectangular channel, dQ/dA = (5/3)·V, so the kinematic
    wave travels at celerity:
        cₖ = dQ/dA = (5/3) · V

    This is the speed at which a disturbance in A propagates downstream.
    Note: cₖ > V — the wave travels faster than the water itself.

    Numerical method — explicit upwind (BTCS-like)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The continuity equation ∂A/∂t + ∂Q/∂x = 0 is discretised with
    a backward (upwind) spatial difference for the flux gradient and
    a forward Euler time step:

        A[i]^{n+1} = A[i]^n − Δt/Δx · (Q[i]^n − Q[i-1]^n)

    This gives a 1st-order accurate, numerically stable scheme under:
        CFL = cₖ · Δt/Δx ≤ 0.9

    The numerical truncation error has the form ½·cₖ·Δx·(1 − Cr)·∂²A/∂x²,
    which acts as a small numerical diffusion term.  This is what gives the
    kinematic wave solver its characteristic slight broadening of the flood
    peak in numerical tests — it is NOT physical attenuation.

    Q consistency fix
    ~~~~~~~~~~~~~~~~~
    After updating A from continuity, Q is immediately recomputed from
    the updated A via Manning's equation.  This ensures Q and A are
    always at the same time level, preventing temporal inconsistency
    that would arise from using the old Q to update A and then storing
    the old Q alongside the new A.

    Limitations
    ~~~~~~~~~~~
    • No backwater effects → cannot model bridges, constrictions,
      tidal influence, or structures that set a downstream stage
    • Only valid for steep, friction-dominated channels where S₀ ≈ Sƒ
      (typically S₀ > ~0.001 and Fr not far from 1)
    • Hydrograph travels downstream without true physical attenuation;
      any peak reduction seen is numerical (scheme diffusion)

    When to use this solver
    ~~~~~~~~~~~~~~~~~~~~~~~
    • Rapid flood routing through steep tributary channels
    • Headwater catchments far from any downstream control
    • As a fast baseline to assess whether inertia terms matter
    """
    nx, x, z_bed = _setup_grid(L, dx, S0)
    t_data, Q_data, _ = _load_bc_data(csv_file, 'NORMAL_DEPTH')

    Q0      = np.interp(0.0, t_data, Q_data)
    y_start = get_normal_depth(Q0, B, n_manning, S0)
    A = np.full(nx, B * y_start)
    Q = np.full(nx, Q0)

    # Sediment transport state
    sed = _init_sediment_state(nx, z_bed, B, SED_GRAIN_SIZES, SED_BED_FRACTIONS) if SEDIMENT_TRANSPORT else None

    results    = []
    next_store = 0.0
    t = 0.0

    print("Starting Kinematic Wave …")
    while t < t_end:
        A = np.maximum(A, 0.01)
        R = A / (B + 2.0 * A / B)
        Q = (1.0 / n_manning) * A * R ** (2.0 / 3.0) * S0 ** 0.5

        ck     = (5.0 / 3.0) * Q / A
        max_ck = np.max(ck)
        dt = 0.9 * dx / max_ck if max_ck > 1e-6 else 1.0
        if t + dt > t_end:
            dt = t_end - t

        Q_in = np.interp(t + dt, t_data, Q_data)
        y_in = get_normal_depth(Q_in, B, n_manning, S0)

        # Continuity: backward (upwind) finite difference for positive flow
        #   ∂A/∂t + ∂Q/∂x = 0  →  A[i]ⁿ⁺¹ = A[i]ⁿ − Δt/Δx (Q[i]ⁿ − Q[i-1]ⁿ)
        dQ_dx  = np.diff(Q) / dx        # dQ_dx[i] = (Q[i+1]−Q[i])/Δx → A[1:] backward
        A[1:] -= dt * dQ_dx
        A[0]   = B * y_in

        # Q is a diagnostic (Manning's) — always recompute from the updated
        # A so that stored Q and A are at the same time level.
        A = np.maximum(A, 0.01)
        R = A / (B + 2.0 * A / B)
        Q = (1.0 / n_manning) * A * R ** (2.0 / 3.0) * S0 ** 0.5

        # Coupled sediment transport step
        if sed is not None:
            z_bed = _sediment_step(sed, A / B, Q, B, n_manning, SED_GRAIN_SIZES,
                                   dx, dt, nx, SED_INFLOW_BC, SED_INFLOW_CONC,
                                   formula=SED_FORMULA,
                                   feed_fractions=SED_FEED_FRACTIONS,
                                   current_time=t)

        t += dt
        if t >= next_store:
            sed_snap = {k: v.copy() for k, v in sed.items()} if sed else None
            results.append((x.copy(), z_bed.copy(), (A / B).copy(), Q.copy(), t, sed_snap))
            next_store += store_interval

    print("  ✓ Kinematic wave complete.")
    return results


# ────────────────────────────────────────────────────────────────
#  3-E.  Diffusive Wave  (Zero-Inertia)
# ────────────────────────────────────────────────────────────────

def solve_diffusive_wave(L, B, S0, n_manning, dx, t_end, csv_file,
                         bc_type, store_interval=30.0):
    """
    Diffusive-wave (zero-inertia) approximation — Muskingum-Cunge scheme.

    Physics retained
    ~~~~~~~~~~~~~~~~
    The TWO inertia terms are dropped from the 1-D momentum equation:
        ∂Q/∂t         → 0   (local acceleration)
        ∂(Q²/A)/∂x    → 0   (convective acceleration)

    What remains is a force balance:  Sƒ = S₀ − ∂y/∂x.
    Substituting Manning's equation and the continuity equation gives
    the *parabolic* (advection-diffusion) form in Q:

        ∂Q/∂t + cₖ ∂Q/∂x = D ∂²Q/∂x²

    where:
        cₖ = (5/3) V         kinematic (Seddon) wave celerity
        D  = Q/(2 B S₀)      Hayami hydraulic diffusivity

    This captures flood-wave *attenuation* (peak reduction and lag),
    which the kinematic wave cannot reproduce.

    Numerical method — Muskingum-Cunge
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The parabolic PDE is discretised with an explicit upwind scheme
    for the advective term and centred differences for the diffusion
    term.  The Courant number Cr = cₖ Δt/Δx and diffusion number
    d = D Δt/Δx² are computed each step and the timestep is set so
    that both stability conditions are satisfied:
        Cr ≤ 1   (advective CFL)
        2 d ≤ 1  (parabolic / von Neumann)

    This formulation works directly with Q as the primary variable,
    which avoids the spurious negative-Q instability that arises in
    the A-based staggered-grid approach when the water-surface
    gradient temporarily reverses during a flood-front arrival.
    Depth is recovered each timestep from Q via Manning's rating curve.

    Engineering use
    ~~~~~~~~~~~~~~~
    The Muskingum-Cunge method is the standard channel-routing
    technique used in operational hydrology (HEC-HMS, SWMM, TOPMODEL)
    and is equivalent to the linearised diffusive wave for a prismatic
    channel.  It captures flood-wave attenuation without the complexity
    of the full Saint-Venant equations.

    Mass-balance note
    ~~~~~~~~~~~~~~~~~
    Because Q is the primary variable and A is recovered from the
    Manning rating curve (A = f(Q), not evolved by the continuity PDE),
    the volume-based mass-balance check will show a non-zero error.
    This is a known artefact of the linearised routing formulation —
    the actual Q field conserves the routed volume; the discrepancy
    arises because the algebraic A(Q) relationship does not exactly
    satisfy the finite-difference continuity equation.
    """
    bc_type = bc_type.upper()
    nx, x, z_bed = _setup_grid(L, dx, S0)
    z_bed_ds = z_bed[-1]
    t_data, Q_data, stage_data = _load_bc_data(csv_file, bc_type)

    Q0      = np.interp(0.0, t_data, Q_data)
    Q       = np.full(nx, Q0)
    # Initial depth via Manning's rating curve (uniform normal depth)
    y_init  = get_normal_depth(Q0, B, n_manning, S0)
    A       = np.full(nx, B * y_init)

    # Sediment transport state
    sed = _init_sediment_state(nx, z_bed, B, SED_GRAIN_SIZES, SED_BED_FRACTIONS) if SEDIMENT_TRANSPORT else None

    results    = []
    next_store = 0.0
    t          = 0.0

    print("Starting Diffusive Wave …")
    while t < t_end:
        # ── Hydraulic parameters from current Q via Manning ──────────────
        Q_pos = np.maximum(Q, 0.001)          # guard division; Q ≥ 0 here
        # Normal depth from Manning's equation (cell-centred)
        y     = np.array([get_normal_depth(qi, B, n_manning, S0) for qi in Q_pos])
        V     = Q_pos / (B * y)               # cross-section-averaged velocity
        ck    = (5.0 / 3.0) * V               # Seddon celerity
        D     = Q_pos / (2.0 * B * S0)        # Hayami diffusivity

        max_ck = float(np.max(ck))
        max_D  = float(np.max(D))

        # ── Timestep: Cr ≤ 0.9  AND  2d ≤ 0.9 ──────────────────────────
        dt_adv  = 0.9 * dx / max_ck if max_ck > 1e-8 else 2.0
        dt_diff = 0.9 * dx ** 2 / (2.0 * max_D) if max_D > 1e-8 else 2.0
        dt      = min(dt_adv, dt_diff, 2.0)
        if t + dt > t_end:
            dt = t_end - t

        Q_in_next = np.interp(t + dt, t_data, Q_data)

        # ── Explicit upwind advection + centred diffusion ────────────────
        # Scheme (interior cells i = 1 … nx-2):
        #   Q[i]^{n+1} = Q[i] - Cr[i]*(Q[i]-Q[i-1])
        #                      + d[i]*(Q[i+1] - 2Q[i] + Q[i-1])
        Cr  = ck * dt / dx
        d   = D  * dt / dx ** 2
        Q_new = Q.copy()
        # Interior nodes: upwind advection (cₖ > 0 ⟹ flow to the right)
        Q_new[1:-1] = (Q[1:-1]
                       - Cr[1:-1] * (Q[1:-1] - Q[:-2])
                       + d[1:-1]  * (Q[2:] - 2.0 * Q[1:-1] + Q[:-2]))

        # ── Boundary conditions ──────────────────────────────────────────
        Q_new[0]  = Q_in_next          # prescribed upstream inflow
        Q_new[-1] = Q_new[-2]          # zero-gradient (Neumann) outflow

        Q = np.maximum(Q_new, 0.0)

        # ── Depth from Manning's rating curve ────────────────────────────
        # In the Muskingum-Cunge model Q and A are always related by
        # Manning's equation (the model's core assumption: Sƒ = S₀).
        # Computing A from the rating curve is therefore internally
        # consistent with the physics and avoids the spurious depth
        # oscillations produced by a separate continuity update.
        A = B * np.array([get_normal_depth(qi, B, n_manning, S0) for qi in Q])

        # ── Coupled sediment transport step ─────────────────────────────
        if sed is not None:
            z_bed = _sediment_step(sed, A / B, Q, B, n_manning, SED_GRAIN_SIZES,
                                   dx, dt, nx, SED_INFLOW_BC, SED_INFLOW_CONC,
                                   formula=SED_FORMULA,
                                   feed_fractions=SED_FEED_FRACTIONS,
                                   current_time=t)

        t += dt
        if t >= next_store:
            sed_snap = {k: v.copy() for k, v in sed.items()} if sed else None
            results.append((x.copy(), z_bed.copy(), (A / B).copy(), Q.copy(), t, sed_snap))
            next_store += store_interval

    print("  ✓ Diffusive wave complete.")
    return results


# ╔══════════════════════════════════════════════════════════════╗
# ║  4.  POST-PROCESSING & VISUALISATION                        ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Solver colour / style definitions (highly distinct) ---
SOLVER_VIS = {
    'Dynamic (MacCormack)':     {'color': '#0072B2', 'ls': '-',   'lw': 2.2, 'marker': None},
    'Dynamic (Lax-Friedrichs)': {'color': '#E69F00', 'ls': '--',  'lw': 2.0, 'marker': None},
    'Dynamic (HLL)':            {'color': '#CC79A7', 'ls': '-.',  'lw': 2.0, 'marker': None},
    'Kinematic':                {'color': '#009E73', 'ls': ':',   'lw': 2.5, 'marker': None},
    'Diffusive':                {'color': '#D55E00', 'ls': (0,(3,1,1,1,1,1)), 'lw': 2.0, 'marker': None},
}


def _safe_show():
    """Display plot interactively, or save as PNG if no display available."""
    try:
        plt.show()
    except Exception:
        fname = 'figure.png'
        plt.savefig(fname, dpi=150)
        print(f"  (no display) Saved → {fname}")
        plt.close()


def _froude_number(y, Q, B, g=9.81):
    """
    Compute the Froude number at each node.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    The Froude number Fr = V / c is the ratio of flow velocity to
    gravity-wave celerity c = √(g·y):

        Fr = V / √(g·y) = Q / (B·y · √(g·y))

    Fr < 1  → subcritical (tranquil) flow — waves can travel upstream;
              backwater effects propagate against the flow direction.
    Fr = 1  → critical flow — minimum energy for a given Q; hydraulic
              control section; wave propagation upstream impossible.
    Fr > 1  → supercritical (rapid) flow — waves are swept downstream;
              downstream conditions cannot influence upstream flow.

    For the default S₀ = 0.01 channel, uniform flow has Fr ≈ 0.8–1.0,
    so the channel is near-critical and the flow regime is sensitive to
    any change in Q or geometry.
    """
    V = Q / (B * np.maximum(y, 0.01))
    return V / np.sqrt(g * np.maximum(y, 0.01))


def _energy_grade_line(z_bed, y, Q, B, g=9.81):
    """
    Compute the Energy Grade Line (EGL) elevation at each node.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    The total specific energy (per unit weight) at a cross section is:

        E = z_bed + y + V²/(2g)     [m above datum]

    where:
      z_bed        – potential energy (bed elevation above datum)
      y            – hydrostatic pressure head (depth of water)
      V²/(2g)      – velocity head (kinetic energy term)

    The EGL always slopes DOWNWARD in the direction of flow (energy is
    lost to friction).  The friction slope Sƒ is the negative gradient
    of the EGL:  Sƒ = −dE/dx.

    Interpretation
    ~~~~~~~~~~~~~~
    • The gap between the EGL and the Water Surface Elevation (WSE =
      z_bed + y) equals the velocity head V²/(2g).  Wide gaps indicate
      high velocities.
    • In uniform flow, EGL slope = bed slope = Sƒ = S₀.
    • In backwater (M1 profile), the EGL is flatter than S₀; in a
      drawdown (M2 profile), it is steeper.
    """
    V = Q / (B * np.maximum(y, 0.01))
    return z_bed + y + V ** 2 / (2.0 * g)


def _mass_balance(results, B, dt_store):
    """
    Compute a simple volume-based mass-balance error for the simulation.

    Method
    ~~~~~~
    Over each storage interval Δt_store, the volume fluxes at the
    upstream and downstream boundaries are approximated as:
        ΔV_in  = Q[0]  · Δt_store      (inflow)
        ΔV_out = Q[-1] · Δt_store      (outflow)

    The change in storage within the domain is:
        ΔS = (A_final − A_initial) · Δx   summed along the channel

    Mass balance error (as % of total inflow):
        err = |ΔV_in − ΔV_out − ΔS| / V_in × 100%

    A perfect solver would give err = 0%.  In practice:
      • MacCormack, Lax-Friedrichs, HLL: < 0.5% (inherently conservative
        or nearly so because the flux-difference form satisfies the
        telescoping property across all interior cells)
      • Kinematic wave: < 0.1% (the continuity equation is solved exactly)
      • Diffusive (Muskingum-Cunge): ~10–15% (known artefact — see solver
        docstring; the rating-curve A does not satisfy the discrete
        continuity equation exactly)

    Returns
    -------
    vol_in, vol_out, delta_storage, error_pct : float
    """
    if len(results) < 2:
        return 0, 0, 0, 0
    dx = results[0][0][1] - results[0][0][0]
    vol_in  = sum(s[3][0]  * dt_store for s in results)
    vol_out = sum(s[3][-1] * dt_store for s in results)
    A_init  = results[0][2]  * B
    A_final = results[-1][2] * B
    dS = np.sum((A_final - A_init) * dx)
    total = vol_in if vol_in != 0 else 1
    err = abs(vol_in - vol_out - dS) / total * 100
    return vol_in, vol_out, dS, err


def _collect_mid_observation_data(solvers, length, width, t_final, num_intervals=1000):
    """
    Sample stage and velocity at the middle of the profile at regular
    time intervals for each solver.
    
    Returns a dictionary with solver names as keys, each containing:
        - 'time': array of sample times
        - 'stage': array of stage values at mid-point
        - 'velocity': array of velocity values at mid-point
    """
    obs_x = length / 2.0
    results = {}
    
    for name, data in solvers.items():
        if not data:
            continue
        
        # Find the observation point index
        x_grid = data[0][0]
        obs_idx = np.argmin(np.abs(x_grid - obs_x))
        
        # Get all time steps from the data
        all_times = np.array([step[4] for step in data])
        
        # Create interpolation points
        target_times = np.linspace(0, t_final, num_intervals)
        
        # Collect stage and velocity at each time
        stage_series = np.array([step[1][obs_idx] + step[2][obs_idx] for step in data])
        vel_series = np.array([step[3][obs_idx] / (step[2][obs_idx] * width) for step in data])
        
        # Interpolate to regular intervals
        stage_interp = np.interp(target_times, all_times, stage_series)
        vel_interp = np.interp(target_times, all_times, vel_series)
        
        results[name] = {
            'time': target_times,
            'stage': stage_interp,
            'velocity': vel_interp
        }
    
    return results


def _plot_mid_observation(obs_data, length):
    """
    Create mid-observation comparison plots:
    - Top left: MacCormack stage vs other solver stages
    - Top right: MacCormack velocity vs other solver velocities
    - Bottom: Time series of stage for all solvers
    """
    if 'Dynamic (MacCormack)' not in obs_data:
        print("Warning: Dynamic (MacCormack) data not available for mid-observation plot.")
        return
    
    maccormack_data = obs_data['Dynamic (MacCormack)']
    mac_stage = maccormack_data['stage']
    mac_vel = maccormack_data['velocity']
    time_array = maccormack_data['time']
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax_stage = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])
    
    # Top-left: MacCormack stage vs other solver stages
    ax_stage.plot([mac_stage.min(), mac_stage.max()], 
                  [mac_stage.min(), mac_stage.max()], 
                  'k--', lw=1, alpha=0.5, label='1:1 Line')
    
    for name, data in obs_data.items():
        if name == 'Dynamic (MacCormack)':
            continue
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        ax_stage.scatter(mac_stage, data['stage'], 
                        c=vis['color'], alpha=0.6, s=20, label=name)
    
    ax_stage.set_xlabel('Dynamic (MacCormack) Stage (m)')
    ax_stage.set_ylabel('Other Solver Stage (m)')
    ax_stage.set_title('Stage Comparison at Mid-Channel')
    ax_stage.legend(fontsize=8)
    ax_stage.grid(True, alpha=0.3)
    
    # Top-right: MacCormack velocity vs other solver velocities
    ax_vel.plot([mac_vel.min(), mac_vel.max()], 
                [mac_vel.min(), mac_vel.max()], 
                'k--', lw=1, alpha=0.5, label='1:1 Line')
    
    for name, data in obs_data.items():
        if name == 'Dynamic (MacCormack)':
            continue
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        ax_vel.scatter(mac_vel, data['velocity'], 
                      c=vis['color'], alpha=0.6, s=20, label=name)
    
    ax_vel.set_xlabel('Dynamic (MacCormack) Velocity (m/s)')
    ax_vel.set_ylabel('Other Solver Velocity (m/s)')
    ax_vel.set_title('Velocity Comparison at Mid-Channel')
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    
    # Bottom: Time series of stage for all solvers
    for name, data in obs_data.items():
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        ax_ts.plot(data['time'], data['stage'], 
                   label=name, color=vis['color'], 
                   ls=vis['ls'], lw=vis['lw'])
    
    ax_ts.set_xlabel('Time (s)')
    ax_ts.set_ylabel('Stage (m)')
    ax_ts.set_title(f'Stage Time Series at Mid-Channel (x = {length/2:.0f} m)')
    ax_ts.legend(fontsize=8, ncol=2)
    ax_ts.grid(True, alpha=0.3)
    
    plt.suptitle('Mid-Channel Observation Analysis', fontsize=14, y=0.995)
    _safe_show()


def _plot_sediment_comparison(solvers, length, width):
    """
    Create sediment transport comparison plots for COMPARE mode.

    Figure 1: Cumulative bed change (Δη) for each solver at end of simulation.
    Figure 2: Bedload transport rate (Q_b) at peak discharge for each solver.
    Figure 3: Bed elevation evolution at mid-channel vs time for each solver.
    """
    obs_x = length / 2.0

    # --- Figure 1: Final bed change profile ---
    fig1, (ax_dz, ax_qb) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for name, data in solvers.items():
        if not data:
            continue
        last = data[-1]
        sed_state = last[5] if len(last) > 5 else None
        if sed_state is None:
            continue
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        x_grid = last[0]
        delta_eta = sed_state['delta_eta']
        ax_dz.plot(x_grid, delta_eta, label=name,
                   color=vis['color'], ls=vis['ls'], lw=vis['lw'])

    ax_dz.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_dz.set_ylabel('Cumulative Bed Change Δη (m)')
    ax_dz.set_title('Final Bed Change Profile (+ = deposition, − = erosion)')
    ax_dz.legend(fontsize=8, ncol=2)
    ax_dz.grid(True, alpha=0.3)

    # --- Bedload transport at peak discharge ---
    for name, data in solvers.items():
        if not data:
            continue
        # Find peak inflow step
        idx_pk = int(np.argmax([s[3][0] for s in data]))
        sed_pk = data[idx_pk][5] if len(data[idx_pk]) > 5 else None
        if sed_pk is None:
            continue
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        ax_qb.plot(data[idx_pk][0], sed_pk['Qb'], label=name,
                   color=vis['color'], ls=vis['ls'], lw=vis['lw'])

    ax_qb.set_xlabel('Distance along channel (m)')
    ax_qb.set_ylabel('Bedload Q_b (m³/s)')
    ax_qb.set_title('Bedload Transport at Peak Discharge')
    ax_qb.legend(fontsize=8, ncol=2)
    ax_qb.grid(True, alpha=0.3)
    fig1.tight_layout()
    _safe_show()

    # --- Figure 2: Bed elevation at mid-channel vs time ---
    fig2, ax_ts = plt.subplots(figsize=(12, 5))
    for name, data in solvers.items():
        if not data:
            continue
        x_grid = data[0][0]
        obs_idx = np.argmin(np.abs(x_grid - obs_x))
        times = []
        bed_at_mid = []
        for step in data:
            sed_st = step[5] if len(step) > 5 else None
            if sed_st is None:
                break
            times.append(step[4])
            bed_at_mid.append(sed_st['eta'][obs_idx])
        if times:
            vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
            ax_ts.plot(times, bed_at_mid, label=name,
                       color=vis['color'], ls=vis['ls'], lw=vis['lw'])

    ax_ts.set_xlabel('Time (s)')
    ax_ts.set_ylabel('Bed Elevation at Mid-Channel (m)')
    ax_ts.set_title(f'Bed Elevation Time Series (x = {obs_x:.0f} m)')
    ax_ts.legend(fontsize=8, ncol=2)
    ax_ts.grid(True, alpha=0.3)
    fig2.tight_layout()
    _safe_show()


def _plot_sediment_single(data, length, width, solver_name):
    """
    Create sediment transport plots for single-solver mode.

    Figure 1: Initial vs final bed profile with bed change.
    Figure 2: Bedload transport profiles at key times.
    Figure 3: Bed elevation time series at mid-channel.
    """
    if not data or data[0][5] is None:
        return

    x_grid = data[0][0]
    obs_x = length / 2.0
    obs_idx = np.argmin(np.abs(x_grid - obs_x))

    initial_bed = data[0][1].copy()
    final_bed = data[-1][5]['eta']
    delta_eta = data[-1][5]['delta_eta']

    t_series = np.array([s[4] for s in data])
    qin_series = np.array([s[3][0] for s in data])
    idx_peak = int(np.argmax(qin_series))

    # --- Figure 1: Bed profile change ---
    fig1, (ax_bed, ax_dz) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_bed.plot(x_grid, initial_bed, 'k-', lw=2, label='Initial Bed')
    ax_bed.plot(x_grid, final_bed, 'r-', lw=2, label='Final Bed')
    # Show final WSE
    final_wse = data[-1][1] + data[-1][2]
    ax_bed.plot(x_grid, final_wse, 'b--', lw=1.5, alpha=0.6, label='Final WSE')
    ax_bed.fill_between(x_grid, initial_bed, initial_bed.min() - 1,
                        color='#8B7355', alpha=0.2)
    ax_bed.set_ylabel('Elevation (m)')
    ax_bed.set_title(f'Bed Profile Evolution — {solver_name}')
    ax_bed.legend(fontsize=8)
    ax_bed.grid(True, alpha=0.3)

    ax_dz.fill_between(x_grid, delta_eta, 0,
                       where=delta_eta >= 0, color='#009E73', alpha=0.4, label='Deposition')
    ax_dz.fill_between(x_grid, delta_eta, 0,
                       where=delta_eta < 0, color='#D55E00', alpha=0.4, label='Erosion')
    ax_dz.plot(x_grid, delta_eta, 'k-', lw=1)
    ax_dz.axhline(0, color='gray', ls='--', lw=0.8)
    ax_dz.set_xlabel('Distance (m)')
    ax_dz.set_ylabel('Δη (m)')
    ax_dz.set_title('Cumulative Bed Change')
    ax_dz.legend(fontsize=8)
    ax_dz.grid(True, alpha=0.3)
    fig1.tight_layout()
    _safe_show()

    # --- Figure 2: Bedload transport at key times ---
    fig2, ax_qb = plt.subplots(figsize=(10, 5))
    prof_indices = [0, idx_peak, -1]
    prof_labels = ['Initial',
                   f'Peak Q (t={t_series[idx_peak]:.0f}s)',
                   f'Final (t={t_series[-1]:.0f}s)']
    prof_colors = ['#0072B2', '#D55E00', '#009E73']
    for i, idx in enumerate(prof_indices):
        sed_st = data[idx][5]
        if sed_st is not None:
            ax_qb.plot(x_grid, sed_st['Qb'], color=prof_colors[i],
                       lw=1.8, label=prof_labels[i])
    ax_qb.set_xlabel('Distance (m)')
    ax_qb.set_ylabel('Bedload Q_b (m³/s)')
    ax_qb.set_title('Bedload Transport Rate Profiles')
    ax_qb.legend(fontsize=8)
    ax_qb.grid(True, alpha=0.3)
    fig2.tight_layout()
    _safe_show()

    # --- Figure 3: Bed elevation time series at mid-channel ---
    fig3, ax_ts = plt.subplots(figsize=(10, 5))
    bed_mid = np.array([s[5]['eta'][obs_idx] for s in data if s[5] is not None])
    t_mid = np.array([s[4] for s in data if s[5] is not None])
    ax_ts.plot(t_mid, bed_mid, 'k-', lw=1.8)
    ax_ts.set_xlabel('Time (s)')
    ax_ts.set_ylabel('Bed Elevation (m)')
    ax_ts.set_title(f'Bed Elevation at Mid-Channel (x = {obs_x:.0f} m)')
    ax_ts.grid(True, alpha=0.3)
    fig3.tight_layout()
    _safe_show()

    # Sediment mass balance
    rho_s = SED_SPECIFIC_GRAVITY * 1000.0
    total_dep = np.sum(np.maximum(delta_eta, 0) * width * (x_grid[1] - x_grid[0]))
    total_ero = np.sum(np.maximum(-delta_eta, 0) * width * (x_grid[1] - x_grid[0]))
    net_change = np.sum(delta_eta * width * (x_grid[1] - x_grid[0]))
    _formula_names = {
        'MPM': 'Meyer-Peter & Müller',
        'WILCOCK_CROWE': 'Wilcock & Crowe (2003)',
        'PARKER_KLINGMAN': 'Parker, Klingeman & McLean (1982)',
    }
    _grain_str = ', '.join(f'{d*1000:.1f}' for d in SED_GRAIN_SIZES) + ' mm'
    print(f"\n  Sediment summary ({solver_name}):")
    print(f"    Formula:                 {_formula_names.get(SED_FORMULA, SED_FORMULA)}")
    print(f"    Grain sizes:             {_grain_str}  ({len(SED_GRAIN_SIZES)} fractions)")
    print(f"    Total deposition volume: {total_dep:.3f} m³")
    print(f"    Total erosion volume:    {total_ero:.3f} m³")
    print(f"    Net bed change volume:   {net_change:.3f} m³")
    print(f"    Max deposition:          {np.max(delta_eta):.4f} m")
    print(f"    Max erosion:             {np.min(delta_eta):.4f} m\n")


def _plot_sediment_output(solvers, length, width, t_final):
    """
    Create the comprehensive SEDIMENT output option plot.
    
    Layout (5 subplots):
    1. Top (full width): Final bed elevation change profile for all solvers
    2. Second row (full width): Cumulative sediment bed change volume along transect
    3. Third row, left: Rating curve at mid-observation (Q vs C, with shear stress)
    4. Third row, right: Time series of sediment flux at mid-observation
    5. Bottom (full width): Box plots of bed change binned at 1/10 channel length intervals
    """
    if not SEDIMENT_TRANSPORT:
        print("Warning: SEDIMENT output type requires SEDIMENT_TRANSPORT = True")
        return
    
    obs_x = length / 2.0
    
    # Create figure with custom grid: 4 rows, 2 columns
    fig = plt.figure(figsize=(16, 17))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.2, 1.2], hspace=0.40, wspace=0.3)
    ax_bed  = fig.add_subplot(gs[0, :])      # Row 0: bed elevation change
    ax_vol  = fig.add_subplot(gs[1, :])      # Row 1: cumulative volume
    ax_rate = fig.add_subplot(gs[2, 0])      # Row 2 left: rating curve
    ax_ts   = fig.add_subplot(gs[2, 1])      # Row 2 right: sediment flux time series
    ax_box  = fig.add_subplot(gs[3, :])      # Row 3: box plots by channel segment
    
    # ─── Subplot 1: Final bed elevation change profile (top, full width) ───
    # First pass: find global y-axis max across all solvers to avoid skewing
    all_delta_eta = []
    for name, data in solvers.items():
        if not data:
            continue
        last = data[-1]
        sed_state = last[5] if len(last) > 5 else None
        if sed_state is None:
            continue
        all_delta_eta.extend(sed_state['delta_eta'])
    
    if all_delta_eta:
        arr = np.array(all_delta_eta)
        vmin, vmax = np.percentile(arr, 2), np.percentile(arr, 98)
        # Add a small pad and guard against zero-range
        pad = max((vmax - vmin) * 0.1, 1e-6)
        y_lim_low, y_lim_high = vmin - pad, vmax + pad
    else:
        y_lim_low, y_lim_high = -1.0, 1.0
    
    for name, data in solvers.items():
        if not data:
            continue
        last = data[-1]
        sed_state = last[5] if len(last) > 5 else None
        if sed_state is None:
            continue
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        x_grid = last[0]
        delta_eta = sed_state['delta_eta']
        ax_bed.plot(x_grid, delta_eta, label=name,
                   color=vis['color'], ls=vis['ls'], lw=vis['lw'])
    
    ax_bed.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_bed.axvline(obs_x, color='gray', ls=':', lw=1, alpha=0.5)
    ax_bed.set_ylim(y_lim_low, y_lim_high)
    ax_bed.set_xlabel('Distance along channel (m)', fontsize=11)
    ax_bed.set_ylabel('Change in Bed Elevation (m)', fontsize=11)
    ax_bed.set_title('Final Bed Elevation Change Profile', fontsize=12, fontweight='bold')
    ax_bed.legend(fontsize=9, ncol=3, loc='best')
    ax_bed.grid(True, alpha=0.3)
    
    # ─── Subplot 2: Cumulative sediment bed change volume along transect ───
    for name, data in solvers.items():
        if not data:
            continue
        last = data[-1]
        sed_state = last[5] if len(last) > 5 else None
        if sed_state is None:
            continue
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        x_grid = last[0]
        delta_eta = sed_state['delta_eta']
        dx = x_grid[1] - x_grid[0]
        
        # Cumulative volume from upstream to downstream
        # Volume at each node = delta_eta * width * dx
        volume_increments = delta_eta * width * dx
        cumulative_volume = np.cumsum(volume_increments)
        
        ax_vol.plot(x_grid, cumulative_volume, label=name,
               color=vis['color'], ls=vis['ls'], lw=vis['lw'])
    
    ax_vol.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_vol.axvline(obs_x, color='gray', ls=':', lw=1, alpha=0.5)
    # Determine y-limits for cumulative volume using percentiles to avoid extreme outliers
    all_vols = []
    for line in ax_vol.get_lines():
        all_vols.extend(line.get_ydata())
    if all_vols:
        arrv = np.array(all_vols)
        vmin_v, vmax_v = np.percentile(arrv, 2), np.percentile(arrv, 98)
        pad_v = max((vmax_v - vmin_v) * 0.1, 1e-6)
        ax_vol.set_ylim(vmin_v - pad_v, vmax_v + pad_v)

    ax_vol.set_xlabel('Distance along channel (m)', fontsize=11)
    ax_vol.set_ylabel('Cumulative Bed Change Volume (m³)', fontsize=11)
    ax_vol.set_title('Cumulative Sediment Volume Along Channel', fontsize=12, fontweight='bold')
    ax_vol.legend(fontsize=9, ncol=3, loc='best')
    ax_vol.grid(True, alpha=0.3)
    
    # ─── Subplot 3: Rating curve at mid-observation (bottom left) ───
    # Collect data for rating curve and shear stress
    all_tau = []
    rating_data = {}
    
    for name, data in solvers.items():
        if not data:
            continue
        if len(data[0]) <= 5 or data[0][5] is None:
            continue
        
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        x_grid = data[0][0]
        obs_idx = np.argmin(np.abs(x_grid - obs_x))
        
        Q_vals = []
        C_vals = []
        tau_vals = []
        
        rho_s = SED_SPECIFIC_GRAVITY * 1000.0
        
        for step in data:
            sed_st = step[5] if len(step) > 5 else None
            if sed_st is None:
                break
            
            y_i = step[2][obs_idx]
            Q_i = step[3][obs_idx]
            Qb_i = sed_st['Qb'][obs_idx]
            
            # Depth-averaged sediment concentration: C = ρ_s * Q_b / Q
            if abs(Q_i) > 1e-6:
                C_i = rho_s * Qb_i / abs(Q_i)
            else:
                C_i = 0.0
            
            # Bed shear stress
            tau_star, tau_bed, _ = _sed_shields_parameter(
                np.array([y_i]), np.array([Q_i]), width, MANNING_N, SED_D50
            )
            
            Q_vals.append(abs(Q_i))
            C_vals.append(C_i)
            tau_vals.append(tau_bed[0])
        
        if Q_vals:
            ax_rate.scatter(Q_vals, C_vals, c=vis['color'], s=20, alpha=0.6, label=name)
            all_tau.extend(tau_vals)
            rating_data[name] = {'Q': Q_vals, 'C': C_vals, 'tau': tau_vals}
    
    ax_rate.set_xlabel('Total Discharge Q (m³/s)', fontsize=11)
    ax_rate.set_ylabel('Sediment Concentration C (kg/m³)', fontsize=11)
    ax_rate.set_title('Rating Curve at Mid-Channel', fontsize=12, fontweight='bold')
    ax_rate.legend(fontsize=8, loc='upper left')
    ax_rate.grid(True, alpha=0.3)
    
    # Second x-axis for shear stress (positioned below)
    ax_tau = ax_rate.twiny()
    ax_tau.set_xlabel('Boundary Shear Stress τ (Pa)', fontsize=10, color='#D55E00')
    ax_tau.tick_params(axis='x', labelcolor='#D55E00', top=False, labeltop=False, 
                      bottom=True, labelbottom=True)
    
    if all_tau:
        tau_min, tau_max = min(all_tau), max(all_tau)
        tau_pad = (tau_max - tau_min) * 0.1 if tau_max > tau_min else 0.1
        ax_tau.set_xlim(tau_min - tau_pad, tau_max + tau_pad)

    # Clip rating-curve y-axis (concentration) to ignore extreme outliers
    conc_vals = []
    for coll in ax_rate.collections:
        try:
            conc_vals.extend(coll.get_offsets()[:, 1])
        except Exception:
            pass
    if conc_vals:
        carr = np.array(conc_vals)
        cmin, cmax = np.percentile(carr, 2), np.percentile(carr, 98)
        padc = max((cmax - cmin) * 0.1, 1e-6)
        ax_rate.set_ylim(max(cmin - padc, 0.0), cmax + padc)
    
    ax_tau.spines['bottom'].set_position(('outward', 40))
    ax_tau.xaxis.set_ticks_position('bottom')
    ax_tau.xaxis.set_label_position('bottom')
    
    # ─── Subplot 4: Time series of sediment flux at mid-observation (bottom right) ───
    for name, data in solvers.items():
        if not data:
            continue
        if len(data[0]) <= 5 or data[0][5] is None:
            continue
        
        vis = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        x_grid = data[0][0]
        obs_idx = np.argmin(np.abs(x_grid - obs_x))
        
        times = []
        Qb_mid = []
        
        for step in data:
            sed_st = step[5] if len(step) > 5 else None
            if sed_st is None:
                break
            times.append(step[4])
            Qb_mid.append(sed_st['Qb'][obs_idx])
        
        if times:
            ax_ts.plot(times, Qb_mid, label=name,
                      color=vis['color'], ls=vis['ls'], lw=vis['lw'])

    # Clip sediment-flux y-axis to percentile range to avoid extreme spikes
    flux_vals = []
    for line in ax_ts.get_lines():
        flux_vals.extend(line.get_ydata())
    if flux_vals:
        arrf = np.array(flux_vals)
        fmin, fmax = np.percentile(arrf, 2), np.percentile(arrf, 98)
        padf = max((fmax - fmin) * 0.1, 1e-9)
        ax_ts.set_ylim(max(fmin - padf, 0.0), fmax + padf)
    
    ax_ts.set_xlabel('Time (s)', fontsize=11)
    ax_ts.set_ylabel('Sediment Flux Qb (m³/s)', fontsize=11)
    ax_ts.set_title(f'Sediment Flux Time Series at x = {obs_x:.0f} m', fontsize=12, fontweight='bold')
    ax_ts.legend(fontsize=8, ncol=1, loc='best')
    ax_ts.grid(True, alpha=0.3)
    
    # ─── Subplot 5: Box plots of bed change binned at 1/10 channel length ───
    n_bins = 10
    bin_edges   = np.linspace(0, length, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_span    = length / n_bins            # width of one bin [m]

    # Collect solvers that have valid final sediment data
    active_solvers = [
        name for name, data in solvers.items()
        if data and len(data[-1]) > 5 and data[-1][5] is not None
    ]
    n_sol = len(active_solvers)

    if n_sol > 0:
        # Distribute solver boxes evenly within each bin, using 70% of bin width
        box_w   = bin_span * 0.70 / n_sol
        half    = bin_span * 0.35 - box_w * 0.5
        offsets = np.linspace(-half, half, n_sol)

        for s_idx, name in enumerate(active_solvers):
            last      = solvers[name][-1]
            sed_state = last[5]
            vis       = SOLVER_VIS.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
            x_grid    = last[0]
            d_eta     = sed_state['delta_eta']

            bin_data  = []
            for b in range(n_bins):
                mask = (x_grid >= bin_edges[b]) & (x_grid < bin_edges[b + 1])
                vals = d_eta[mask]
                bin_data.append(vals.tolist() if len(vals) > 0 else [0.0])

            positions = bin_centers + offsets[s_idx]
            bp = ax_box.boxplot(
                bin_data,
                positions=positions,
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', lw=1.5),
                boxprops=dict(facecolor=vis['color'], alpha=0.55),
                whiskerprops=dict(color=vis['color'], lw=1.2),
                capprops=dict(color=vis['color'], lw=1.2),
            )
            # Proxy artist for legend
            ax_box.plot([], [], color=vis['color'], lw=6, alpha=0.55, label=name)

    ax_box.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_box.set_xlim(0, length)
    ax_box.set_xticks(bin_centers)
    ax_box.set_xticklabels([f'{c:.0f}' for c in bin_centers], fontsize=9)
    ax_box.set_xlabel('Distance along channel (m)', fontsize=11)
    ax_box.set_ylabel('Change in Bed Elevation (m)', fontsize=11)
    ax_box.set_title(
        f'Bed Elevation Change Distribution — {n_bins} Segments × {bin_span:.0f} m',
        fontsize=12, fontweight='bold'
    )
    if n_sol > 0:
        ax_box.legend(fontsize=8, ncol=min(n_sol, 5), loc='best')
    ax_box.grid(True, alpha=0.3, axis='y')

    # Percentile y-clip for box plot (same scheme as other panels)
    box_all_vals = []
    for name in active_solvers:
        box_all_vals.extend(solvers[name][-1][5]['delta_eta'])
    if box_all_vals:
        arrb = np.array(box_all_vals)
        bmin, bmax = np.percentile(arrb, 2), np.percentile(arrb, 98)
        padb = max((bmax - bmin) * 0.15, 1e-6)
        ax_box.set_ylim(bmin - padb, bmax + padb)

    # Build formula & grain info string for the main title
    _formula_names = {
        'MPM': 'Meyer-Peter & Müller',
        'WILCOCK_CROWE': 'Wilcock & Crowe (2003)',
        'PARKER_KLINGMAN': 'Parker, Klingeman & McLean (1982)',
    }
    _formula_label = _formula_names.get(SED_FORMULA, SED_FORMULA)
    _grain_str = ', '.join(f'{d*1000:.1f}' for d in SED_GRAIN_SIZES) + ' mm'
    _nfrac = len(SED_GRAIN_SIZES)
    _title_extra = f'{_formula_label}  |  {_nfrac} fraction{"s" if _nfrac > 1 else ""}: {_grain_str}'

    plt.suptitle(f'Sediment Transport Analysis\n{_title_extra}',
                 fontsize=13, fontweight='bold', y=0.998)
    _safe_show()


# ────────────────────────────────────────────────────────────────
#  Animation: frame-by-frame WSE longitudinal profile
# ────────────────────────────────────────────────────────────────

def build_animation(solvers, solver_vis, length, t_final, csv_path,
                    script_dir, fps=15):
    """
    Build a polished longitudinal-WSE animation and save it.

    Tries .mp4 (ffmpeg) → .gif (Pillow) → gives up gracefully.
    Each solver gets a unique colour, dash pattern, and line-width
    so that they remain distinguishable even in monochrome prints.
    """
    # Pick a reference solver for grid / bed data
    ref = None
    for data in solvers.values():
        if data:
            ref = data
            break
    if ref is None:
        print("  No solver data — skipping animation.")
        return

    x_grid = ref[0][0]
    z_bed  = ref[0][1]

    # Determine vertical range across ALL solvers
    wse_min =  1e9
    wse_max = -1e9
    for data in solvers.values():
        if not data:
            continue
        for step in data:
            wse = step[1] + step[2]
            wse_min = min(wse_min, np.min(wse))
            wse_max = max(wse_max, np.max(wse))
    pad = max((wse_max - wse_min) * 0.12, 0.3)

    # --- Figure layout ---
    fig = plt.figure(figsize=(13, 7))
    gs  = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], width_ratios=[3, 1],
                           hspace=0.30, wspace=0.25)
    ax_main  = fig.add_subplot(gs[0, :])
    ax_hydro = fig.add_subplot(gs[1, 0])
    ax_fr    = fig.add_subplot(gs[1, 1])

    # Main axis
    ax_main.set_xlim(0, length)
    ax_main.set_ylim(wse_min - pad, wse_max + pad)
    ax_main.set_xlabel('Distance along channel (m)')
    ax_main.set_ylabel('Elevation (m)')
    ax_main.fill_between(x_grid, z_bed, z_bed.min() - 2,
                         color='#8B7355', alpha=0.35, label='Channel bed')
    ax_main.plot(x_grid, z_bed, 'k-', lw=1.5)
    ax_main.grid(True, alpha=0.25)
    title_txt = ax_main.set_title('Water Surface Profile  —  t = 0.0 s',
                                  fontsize=13, fontweight='bold')

    # Build animated lines
    anim_lines = {}
    bed_lines  = {}      # per-solver animated bed lines (sediment mode)
    for name, vis in solver_vis.items():
        if name in solvers and solvers[name]:
            line, = ax_main.plot([], [], label=name,
                                 color=vis['color'], ls=vis['ls'], lw=vis['lw'])
            anim_lines[name] = line
            # Check if sediment data is present
            if SEDIMENT_TRANSPORT and len(solvers[name][0]) > 5 and solvers[name][0][5] is not None:
                bln, = ax_main.plot([], [], color=vis['color'], ls=':', lw=1.0, alpha=0.6)
                bed_lines[name] = bln
    ax_main.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.85)

    # Inset hydrograph
    try:
        df_bc = pd.read_csv(csv_path)
        t_bc  = df_bc.iloc[:, 0].values
        q_bc  = df_bc.iloc[:, 1].values
    except Exception:
        t_bc = np.linspace(0, t_final, 100)
        q_bc = np.zeros_like(t_bc)

    ax_hydro.fill_between(t_bc, q_bc, alpha=0.2, color='steelblue')
    ax_hydro.plot(t_bc, q_bc, 'steelblue', lw=1)
    ax_hydro.set_xlim(0, t_final)
    q_pad = max((max(q_bc) - min(q_bc)) * 0.1, 1)
    ax_hydro.set_ylim(min(q_bc) - q_pad, max(q_bc) + q_pad)
    ax_hydro.set_xlabel('Time (s)', fontsize=9)
    ax_hydro.set_ylabel('Q (m³/s)', fontsize=9)
    ax_hydro.set_title('Inflow Hydrograph', fontsize=9)
    ax_hydro.grid(True, alpha=0.25)
    time_marker = ax_hydro.axvline(0, color='red', lw=1.5, ls='--')

    # Froude-number panel
    ax_fr.set_xlim(0, length)
    ax_fr.set_ylim(0, 3)
    ax_fr.axhline(1.0, color='gray', ls='--', lw=0.8, label='Fr = 1')
    ax_fr.set_xlabel('Distance (m)', fontsize=9)
    ax_fr.set_ylabel('Fr', fontsize=9)
    ax_fr.set_title('Froude Number', fontsize=9)
    ax_fr.grid(True, alpha=0.25)

    fr_lines = {}
    for name, vis in solver_vis.items():
        if name in anim_lines:
            ln, = ax_fr.plot([], [], color=vis['color'], ls=vis['ls'], lw=1.2)
            fr_lines[name] = ln

    def init():
        for ln in anim_lines.values():
            ln.set_data([], [])
        for ln in bed_lines.values():
            ln.set_data([], [])
        for ln in fr_lines.values():
            ln.set_data([], [])
        time_marker.set_xdata([0])
        return (list(anim_lines.values()) + list(bed_lines.values())
                + list(fr_lines.values()) + [time_marker, title_txt])

    def update(frame_idx):
        current_time = 0.0
        for name, data in solvers.items():
            if name not in anim_lines:
                continue
            if data and frame_idx < len(data):
                step = data[frame_idx]
                wse  = step[1] + step[2]
                anim_lines[name].set_data(step[0], wse)
                fr = _froude_number(step[2], step[3], WIDTH)
                fr_lines[name].set_data(step[0], fr)
                current_time = step[4]
                # Update bed line if sediment is active
                if name in bed_lines and len(step) > 5 and step[5] is not None:
                    bed_lines[name].set_data(step[0], step[5]['eta'])
        time_marker.set_xdata([current_time])
        title_txt.set_text(f'Water Surface Profile  —  t = {current_time:.1f} s')
        return (list(anim_lines.values()) + list(bed_lines.values())
                + list(fr_lines.values()) + [time_marker, title_txt])

    num_frames = max(len(d) for d in solvers.values() if d)
    ani = FuncAnimation(fig, update, frames=num_frames,
                        init_func=init, blit=True, interval=50)

    # Save as MP4 using ffmpeg. Do NOT fall back to GIF — require mp4 output.
    base = os.path.join(script_dir, 'simulation_animation')
    try:
        import shutil
        ffmpeg_bin = shutil.which('ffmpeg')
        if not ffmpeg_bin:
            print("ERROR: 'ffmpeg' not found on PATH. To save as MP4, install ffmpeg and ensure it's on your PATH.")
            print("  Windows: https://ffmpeg.org/download.html — add to PATH, then re-run the script.")
            plt.close(fig)
            return
        
        # Create a custom progress callback for animation saving
        print("  Saving animation...")
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, metadata={'artist': '1DFlowModel'})
        
        total_frames = num_frames
        progress_bar_width = 50
        
        with writer.saving(fig, base + '.mp4', dpi=120):
            for frame_num in range(total_frames):
                update(frame_num)
                writer.grab_frame()
                
                # Display progress meter
                progress = (frame_num + 1) / total_frames
                filled = int(progress_bar_width * progress)
                bar = '█' * filled + '░' * (progress_bar_width - filled)
                percent = progress * 100
                print(f"\r  Writing animation: |{bar}| {percent:.1f}% ({frame_num+1}/{total_frames} frames)", 
                      end='', flush=True)
        
        print()  # New line after progress bar
        print(f"  Animation saved → {base}.mp4")
    except Exception as e:
        print(f"\nERROR saving animation as MP4: {e}")
        print("Ensure ffmpeg is installed and available on PATH. No GIF fallback is performed.")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
#  Solver comparison summary table
# ────────────────────────────────────────────────────────────────

def print_solver_summary(solvers, width, store_interval):
    """Print a tabular comparison of each solver's key metrics."""
    print("\n" + "=" * 72)
    print("  SOLVER COMPARISON SUMMARY")
    print("=" * 72)
    header = f"  {'Solver':<28s} {'Peak Q_out':>10s} {'Peak WSE':>10s} {'Max Fr':>8s} {'Mass Err':>10s}"
    print(header)
    print("  " + "-" * 68)
    for name, data in solvers.items():
        if not data:
            print(f"  {name:<28s}  {'(no data)':>10s}")
            continue
        q_out_max = max(s[3][-1] for s in data)
        wse_max   = max(np.max(s[1] + s[2]) for s in data)
        fr_max    = max(np.max(_froude_number(s[2], s[3], width)) for s in data)
        _, _, _, err = _mass_balance(data, width, store_interval)
        print(f"  {name:<28s} {q_out_max:10.2f} {wse_max:10.3f} {fr_max:8.3f} {err:9.2f}%")
    print("=" * 72 + "\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  5.  EXECUTION                                              ║
# ╚══════════════════════════════════════════════════════════════╝

if SOLVER_TYPE == 'COMPARE':
    print("\n╔════════════════════════════════════════════════╗")
    print("║  COMPARISON MODE — running all 5 solvers …    ║")
    print("╚════════════════════════════════════════════════╝\n")

    anim_interval = min(T_FINAL / 100.0, 10.0)
    print(f"Storage interval: {anim_interval:.2f} s\n")

    t0 = _timer.perf_counter()
    res_dyn = solve_dynamic_wave(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX,
                                 T_FINAL, CSV_PATH, BC_MODE, NORMAL_SLOPE,
                                 store_interval=anim_interval)
    res_lax = solve_dynamic_wave_lax(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX,
                                     T_FINAL, CSV_PATH, BC_MODE, NORMAL_SLOPE,
                                     store_interval=anim_interval)
    res_hll = solve_dynamic_wave_hll(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX,
                                     T_FINAL, CSV_PATH, BC_MODE, NORMAL_SLOPE,
                                     store_interval=anim_interval)
    res_kin = solve_kinematic_wave(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX,
                                  T_FINAL, CSV_PATH, store_interval=anim_interval)
    res_dif = solve_diffusive_wave(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX,
                                  T_FINAL, CSV_PATH, BC_MODE,
                                  store_interval=anim_interval)
    elapsed = _timer.perf_counter() - t0
    print(f"\nAll solvers finished in {elapsed:.1f} s.\n")

    solvers = {
        'Dynamic (MacCormack)':      res_dyn,
        'Dynamic (Lax-Friedrichs)':  res_lax,
        'Dynamic (HLL)':             res_hll,
        'Kinematic':                 res_kin,
        'Diffusive':                 res_dif,
    }

    # ── Summary table ──
    print_solver_summary(solvers, WIDTH, anim_interval)

    # ── Generate plots based on OUTPUT_TYPE ──
    if OUTPUT_TYPE == 'MID_OBSERVATION':
        print("\nGenerating mid-observation plots...")
        obs_data = _collect_mid_observation_data(solvers, LENGTH, WIDTH, T_FINAL, num_intervals=1000)
        _plot_mid_observation(obs_data, LENGTH)
    elif OUTPUT_TYPE == 'SEDIMENT':
        print("\nGenerating sediment output plots...")
        _plot_sediment_output(solvers, LENGTH, WIDTH, T_FINAL)
    else:
        # ── Static comparison plots (PROFILE_PLOTS mode) ──
        fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
        ax_wse, ax_vel, ax_fr = axes

        for name, data in solvers.items():
            if not data:
                continue
            vis    = SOLVER_VIS[name]
            x_grid = data[0][0]
            max_wse = np.full_like(x_grid, -1e9)
            max_vel = np.full_like(x_grid, -1e9)
            max_fr  = np.full_like(x_grid, -1e9)
            valid   = True
            for step in data:
                y_i, Q_i = step[2], step[3]
                if np.any(~np.isfinite(y_i)) or np.any(~np.isfinite(Q_i)):
                    valid = False
                    break
                max_wse = np.maximum(max_wse, step[1] + y_i)
                v_i     = np.abs(Q_i / (y_i * WIDTH))
                max_vel = np.maximum(max_vel, v_i)
                max_fr  = np.maximum(max_fr, _froude_number(y_i, Q_i, WIDTH))
            if not valid:
                continue
            ax_wse.plot(x_grid, max_wse, label=name, color=vis['color'],
                        ls=vis['ls'], lw=vis['lw'])
            ax_vel.plot(x_grid, max_vel, label=name, color=vis['color'],
                        ls=vis['ls'], lw=vis['lw'])
            ax_fr.plot(x_grid, max_fr,   label=name, color=vis['color'],
                       ls=vis['ls'], lw=vis['lw'])

        if res_dyn:
            ax_wse.plot(res_dyn[0][0], res_dyn[0][1], 'k-', lw=2, label='Bed Elevation')
            # EGL at peak for reference (MacCormack)
            idx_pk = int(np.argmax([s[3][0] for s in res_dyn]))
            egl = _energy_grade_line(res_dyn[idx_pk][1], res_dyn[idx_pk][2],
                                     res_dyn[idx_pk][3], WIDTH)
            ax_wse.plot(res_dyn[idx_pk][0], egl, 'k--', lw=1, alpha=0.5,
                        label='EGL (MacCormack @ peak Q)')

        ax_wse.set_ylabel('Max WSE (m)');        ax_wse.set_title('Peak Water-Surface Elevation Envelope')
        ax_vel.set_ylabel('Max Velocity (m/s)'); ax_vel.set_title('Peak Velocity Envelope')
        ax_fr.set_ylabel('Max Froude');          ax_fr.set_title('Peak Froude Number Envelope')
        ax_fr.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6, label='Fr = 1 (critical)')
        ax_fr.set_xlabel('Distance along channel (m)')
        for ax in axes:
            ax.legend(fontsize=8, ncol=2, framealpha=0.85)
            ax.grid(True, alpha=0.25)
        plt.tight_layout()
        _safe_show()

    # ── Sediment transport plots (if enabled) ──
    if SEDIMENT_TRANSPORT:
        print("\nGenerating sediment transport plots...")
        _plot_sediment_comparison(solvers, LENGTH, WIDTH)

    # ── Animation ──
    if SAVE_ANIMATION:
        print("Building animation (set SAVE_ANIMATION = False to skip) …")
        build_animation(solvers, SOLVER_VIS, LENGTH, T_FINAL, CSV_PATH,
                        SCRIPT_DIR, fps=ANIMATION_FPS)
    else:
        print("Animation skipped (SAVE_ANIMATION = False).")

else:
    # ── Single-solver execution ──
    if SOLVER_TYPE == 'KINEMATIC':
        data = solve_kinematic_wave(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX, T_FINAL, CSV_PATH)
    elif SOLVER_TYPE == 'DIFFUSIVE':
        data = solve_diffusive_wave(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX, T_FINAL, CSV_PATH, BC_MODE)
    elif SOLVER_TYPE == 'DYNAMIC_LAX':
        data = solve_dynamic_wave_lax(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX, T_FINAL,
                                      CSV_PATH, BC_MODE, NORMAL_SLOPE)
    elif SOLVER_TYPE == 'DYNAMIC_HLL':
        data = solve_dynamic_wave_hll(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX, T_FINAL,
                                      CSV_PATH, BC_MODE, NORMAL_SLOPE)
    else:
        data = solve_dynamic_wave(LENGTH, WIDTH, BED_SLOPE, MANNING_N, DX, T_FINAL,
                                  CSV_PATH, BC_MODE, NORMAL_SLOPE)

    if data:
        t_series   = np.array([s[4] for s in data])
        qin_series = np.array([s[3][0] for s in data])
        stage_out  = np.array([s[1][-1] + s[2][-1] for s in data])

        x_grid  = data[0][0]
        obs_x   = LENGTH / 2.0
        obs_idx = np.argmin(np.abs(x_grid - obs_x))

        obs_stage = np.array([s[1][obs_idx] + s[2][obs_idx] for s in data])
        obs_vel   = np.array([s[3][obs_idx] / (s[2][obs_idx] * WIDTH) for s in data])

        idx_init = 0
        idx_peak = int(np.argmax(qin_series))
        idx_end  = -1
        prof_idx    = [idx_init, idx_peak, idx_end]
        prof_labels = ['Initial',
                       f'Peak Q (t={t_series[idx_peak]:.0f}s)',
                       f'Final (t={t_series[idx_end]:.0f}s)']
        prof_colors = ['#0072B2', '#D55E00', '#009E73']

        # Plot 1 — Boundary conditions
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t_series, qin_series, 'b-', label='Inflow Q')
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Discharge (m³/s)', color='b')
        ax1.tick_params(axis='y', labelcolor='b'); ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(t_series, stage_out, 'r--', label='Outlet Stage')
        ax2.set_ylabel('Stage (m)', color='r'); ax2.tick_params(axis='y', labelcolor='r')
        plt.title('Simulated Boundary Conditions'); fig1.tight_layout(); _safe_show()

        # Plot 2 — Water-level profiles
        fig2, ax_wl = plt.subplots(figsize=(10, 5))
        ax_wl.fill_between(data[0][0], data[0][1], data[0][1].min() - 1,
                           color='#8B7355', alpha=0.3, label='Bed')
        ax_wl.plot(data[0][0], data[0][1], 'k-', lw=2)
        for i, idx in enumerate(prof_idx):
            wse = data[idx][1] + data[idx][2]
            ax_wl.plot(data[idx][0], wse, color=prof_colors[i], lw=1.8, label=prof_labels[i])
        # EGL at peak
        egl = _energy_grade_line(data[idx_peak][1], data[idx_peak][2],
                                 data[idx_peak][3], WIDTH)
        ax_wl.plot(data[idx_peak][0], egl, 'k--', lw=1, alpha=0.4, label='EGL @ peak')
        ax_wl.axvline(x=obs_x, color='gray', ls=':', label='Obs. Point')
        ax_wl.set_xlabel('Distance (m)'); ax_wl.set_ylabel('Elevation (m)')
        ax_wl.set_title('Water Level & Energy Grade Line Profiles')
        ax_wl.legend(fontsize=8); ax_wl.grid(True, alpha=0.3)
        fig2.tight_layout(); _safe_show()

        # Plot 3 — Velocity profiles
        fig3, ax_v = plt.subplots(figsize=(10, 5))
        for i, idx in enumerate(prof_idx):
            V_i = data[idx][3] / (data[idx][2] * WIDTH)
            ax_v.plot(data[idx][0], V_i, color=prof_colors[i], lw=1.8, label=prof_labels[i])
        ax_v.axvline(x=obs_x, color='gray', ls=':', label='Obs. Point')
        ax_v.set_xlabel('Distance (m)'); ax_v.set_ylabel('Velocity (m/s)')
        ax_v.set_title('Velocity Profiles'); ax_v.legend(); ax_v.grid(True, alpha=0.3)
        fig3.tight_layout(); _safe_show()

        # Plot 4 — Observation-point time series
        fig4, ax_o1 = plt.subplots(figsize=(10, 5))
        ax_o1.plot(t_series, obs_stage, 'g-', label='Stage')
        ax_o1.set_xlabel('Time (s)'); ax_o1.set_ylabel('Stage (m)', color='g')
        ax_o1.tick_params(axis='y', labelcolor='g'); ax_o1.grid(True, alpha=0.3)
        ax_o2 = ax_o1.twinx()
        ax_o2.plot(t_series, obs_vel, 'm--', label='Velocity')
        ax_o2.set_ylabel('Velocity (m/s)', color='m'); ax_o2.tick_params(axis='y', labelcolor='m')
        plt.title(f'Time Series at Observation Point (x = {obs_x:.0f} m)')
        fig4.tight_layout(); _safe_show()

        # Plot 5 — Froude number profile at peak
        fig5, ax_fr = plt.subplots(figsize=(10, 4))
        fr_peak = _froude_number(data[idx_peak][2], data[idx_peak][3], WIDTH)
        ax_fr.plot(data[idx_peak][0], fr_peak, '#D55E00', lw=1.8)
        ax_fr.axhline(1.0, color='gray', ls='--', lw=0.8, label='Fr = 1')
        ax_fr.fill_between(data[idx_peak][0], fr_peak, alpha=0.15, color='#D55E00')
        ax_fr.set_xlabel('Distance (m)'); ax_fr.set_ylabel('Froude Number')
        ax_fr.set_title(f'Froude Number Profile at Peak Discharge (t = {t_series[idx_peak]:.0f} s)')
        ax_fr.legend(); ax_fr.grid(True, alpha=0.3)
        fig5.tight_layout(); _safe_show()

        # Mass balance check
        store_dt = t_series[1] - t_series[0] if len(t_series) > 1 else 1.0
        vi, vo, ds, err = _mass_balance(data, WIDTH, store_dt)
        print(f"\n  Mass balance: In={vi:.0f} m³  Out={vo:.0f} m³  ΔS={ds:.0f} m³  Error={err:.2f}%\n")

        # Sediment transport plots (if enabled)
        if SEDIMENT_TRANSPORT:
            solver_name_sed = SOLVER_TYPE.replace('_', ' ').title()
            if solver_name_sed == 'Dynamic':
                solver_name_sed = 'Dynamic (MacCormack)'
            _plot_sediment_single(data, LENGTH, WIDTH, solver_name_sed)

        # Animation (single solver)
        if SAVE_ANIMATION:
            solver_name = SOLVER_TYPE.replace('_', ' ').title()
            if solver_name == 'Dynamic':
                solver_name = 'Dynamic (MacCormack)'
            single_dict = {solver_name: data}
            vis_dict = {solver_name: SOLVER_VIS.get(solver_name,
                        {'color': '#0072B2', 'ls': '-', 'lw': 2, 'marker': None})}
            print("Building animation …")
            build_animation(single_dict, vis_dict, LENGTH, T_FINAL, CSV_PATH,
                            SCRIPT_DIR, fps=ANIMATION_FPS)
