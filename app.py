"""
Ballistic Lead Calculator
3-DOF point-mass model · RK4 integration · relative-air-velocity drag
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

_logo = Image.open("logo.png")

st.set_page_config(
    page_title="Digitest Ballistic Lead Calculator",
    page_icon=_logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --accent: #58a6ff; --green: #3fb950; --text: #e6edf3; --subtext: #8b949e;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important; color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.5rem; }
h1 { color: var(--accent) !important; letter-spacing: 0.04em; }
.metric-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 0.55rem;
}
.metric-label { color: var(--subtext); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { color: var(--accent); font-size: 1.4rem; font-weight: 700; }
.metric-unit  { color: var(--subtext); font-size: 0.82rem; }
.section-header {
    border-left: 3px solid var(--accent); padding-left: 0.7rem;
    margin: 1.1rem 0 0.5rem; font-size: 0.9rem; font-weight: 600;
    color: var(--accent); text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
.stButton > button {
    background: linear-gradient(135deg,#1f6feb,#388bfd); color:#fff !important;
    border:none; border-radius:8px; font-weight:700; width:100%;
    padding:0.55rem 1.4rem; transition:opacity 0.2s;
}
.stButton > button:hover { opacity:0.85; }
.stDownloadButton > button {
    background: linear-gradient(135deg,#238636,#3fb950); color:#fff !important;
    border:none; border-radius:8px; font-weight:700; width:100%;
}
label { color: var(--subtext) !important; font-size: 0.82rem !important; }
input, .stNumberInput input {
    background-color:#21262d !important; color:var(--text) !important;
    border:1px solid var(--border) !important; border-radius:6px !important;
}
.subtitle {
    color:var(--subtext); font-size:0.93rem; margin-top:-0.8rem;
    margin-bottom:1.1rem; letter-spacing:0.03em;
}
.info-box {
    background:#1c2128; border:1px solid var(--border); border-radius:8px;
    padding:0.55rem 0.85rem; font-size:0.76rem; color:var(--subtext); margin-bottom:0.8rem;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

G         = 9.81    # m/s²    gravitational acceleration
R_AIR     = 287.05  # J/kg·K  specific gas constant for dry air
GAMMA_AIR = 1.4     # —       adiabatic index (cp/cv) for air

# Cd(Mach) reference table — np.interp extrapolates flat outside the range
MACH_TABLE = np.array([0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0])
CD_TABLE   = np.array([0.20, 0.22, 0.25, 0.32, 0.40, 0.38, 0.35, 0.31, 0.28, 0.25, 0.23])


# ═══════════════════════════════════════════════════════════════════════════════
#  ATMOSPHERE
# ═══════════════════════════════════════════════════════════════════════════════

def atmosphere(temp_c: float, pressure_hpa: float):
    """ISA-style local atmosphere.  Returns (rho [kg/m³], speed_of_sound [m/s])."""
    T   = temp_c + 273.15
    P   = pressure_hpa * 100.0
    rho = P / (R_AIR * T)
    c   = np.sqrt(GAMMA_AIR * R_AIR * T)
    return rho, c


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAG COEFFICIENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_cd(mach: float, cd_base: float, use_mach_table: bool = False) -> float:
    """
    use_mach_table=False  →  return cd_base (constant, user-supplied)
    use_mach_table=True   →  np.interp from MACH_TABLE / CD_TABLE
    """
    if use_mach_table:
        return float(np.interp(mach, MACH_TABLE, CD_TABLE))
    return cd_base


# ═══════════════════════════════════════════════════════════════════════════════
#  RK4 CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _derivs(state: np.ndarray, wind_vec: np.ndarray,
            rho: float, c: float,
            cd_base: float, area: float, mass: float,
            use_mach_table: bool = False) -> np.ndarray:
    """
    Time derivatives of the 3-DOF point-mass state vector [x, y, z, vx, vy, vz].

    Drag on relative-air velocity:
        v_rel  = v_bullet − wind_vector
        a_drag = −k·|v_rel|·v_rel    where k = ½ρCdA/m
    """
    vel   = state[3:6]
    v_rel = vel - wind_vec
    spd   = np.linalg.norm(v_rel)

    mach = spd / c if c > 0.0 else 0.0
    cd   = get_cd(mach, cd_base, use_mach_table)
    k    = 0.5 * rho * cd * area / mass

    a_drag = -k * spd * v_rel
    a_grav = np.array([0.0, -G, 0.0])

    return np.concatenate([vel, a_drag + a_grav])


def _rk4_step(state: np.ndarray, dt: float, **kw) -> np.ndarray:
    """Single 4th-order Runge-Kutta integration step."""
    k1 = _derivs(state,               **kw)
    k2 = _derivs(state + 0.5*dt*k1,   **kw)
    k3 = _derivs(state + 0.5*dt*k2,   **kw)
    k4 = _derivs(state + dt*k3,        **kw)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_3d(
        distance_m:     float,
        tilt_deg:       float,
        temp_c:         float,
        pressure_hpa:   float,
        wind_x:         float,
        wind_z:         float,
        bullet_mass_g:  float,
        bullet_diam_mm: float,
        muzzle_vel_ms:  float,
        cd_base:        float,
        dt:             float = 0.001,
        use_mach_table: bool  = False,
) -> dict:
    """
    3-DOF point-mass RK4 trajectory.  Returns tof, impact_y, impact_z,
    impact_velocity, impact_energy, impact_mach, avg_cd.
    """
    rho, c = atmosphere(temp_c, pressure_hpa)

    mass = bullet_mass_g / 1000.0
    area = np.pi * (bullet_diam_mm / 2000.0) ** 2

    wind_vec = np.array([wind_x, 0.0, wind_z])
    kw = dict(wind_vec=wind_vec, rho=rho, c=c,
              cd_base=cd_base, area=area, mass=mass,
              use_mach_table=use_mach_table)

    v0x = muzzle_vel_ms * np.cos(np.radians(tilt_deg))
    v0y = muzzle_vel_ms * np.sin(np.radians(tilt_deg))
    state = np.array([0.0, 0.0, 0.0, v0x, v0y, 0.0])

    t          = 0.0
    prev_state = state.copy()
    prev_t     = 0.0
    sum_cd     = 0.0
    step_count = 0

    for _ in range(500_000):
        v_r    = state[3:6] - wind_vec
        spd_r  = np.linalg.norm(v_r)
        mach_r = spd_r / c if c > 0.0 else 0.0
        sum_cd    += get_cd(mach_r, cd_base, use_mach_table)
        step_count += 1

        prev_state = state.copy()
        prev_t     = t
        state = _rk4_step(state, dt, **kw)
        t    += dt

        if state[0] >= distance_m:
            dx   = state[0] - prev_state[0]
            frac = (distance_m - prev_state[0]) / dx if dx > 1e-12 else 1.0
            imp  = prev_state + frac * (state - prev_state)
            tof  = prev_t + frac * dt
            spd  = float(np.linalg.norm(imp[3:6]))
            return {
                "tof":             tof,
                "impact_y":        float(imp[1]),
                "impact_z":        float(imp[2]),
                "impact_velocity": spd,
                "impact_energy":   0.5 * mass * spd**2,
                "impact_mach":     spd / c if c > 0.0 else 0.0,
                "avg_cd":          sum_cd / step_count if step_count else cd_base,
            }

        if state[1] < -(distance_m + 200.0):
            break

    spd = float(np.linalg.norm(state[3:6]))
    return {
        "tof":             t,
        "impact_y":        float(state[1]),
        "impact_z":        float(state[2]),
        "impact_velocity": spd,
        "impact_energy":   0.5 * mass * spd**2,
        "impact_mach":     spd / c if c > 0.0 else 0.0,
        "avg_cd":          sum_cd / step_count if step_count else cd_base,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  REQUIRED ELEVATION  (bisection root-solve)
# ═══════════════════════════════════════════════════════════════════════════════

def find_required_elevation(
        distance_m:     float,
        temp_c:         float,
        pressure_hpa:   float,
        wind_x:         float,
        wind_z:         float,
        bullet_mass_g:  float,
        bullet_diam_mm: float,
        muzzle_vel_ms:  float,
        cd_base:        float,
        tol:            float = 1e-3,
        max_iter:       int   = 60,
        use_mach_table: bool  = False,
) -> float:
    """
    Bisection: find tilt where impact_y = 0 within tol metres.
    Returns 0.0 if target is unreachable.
    """
    def residual(tilt: float) -> float:
        return simulate_3d(
            distance_m, tilt, temp_c, pressure_hpa,
            wind_x, wind_z,
            bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
            use_mach_table=use_mach_table,
        )["impact_y"]

    tilt_lo, f_lo = 0.0, residual(0.0)
    calls = 1

    tilt_hi = 1.0
    f_hi    = residual(tilt_hi)
    calls  += 1

    while f_hi <= 0.0 and tilt_hi < 85.0 and calls < max_iter:
        tilt_hi *= 2.0
        f_hi     = residual(tilt_hi)
        calls   += 1

    if f_hi <= 0.0:
        return 0.0

    while calls < max_iter:
        tilt_mid = 0.5 * (tilt_lo + tilt_hi)
        f_mid    = residual(tilt_mid)
        calls   += 1

        if abs(f_mid) < tol:
            return tilt_mid

        if f_lo * f_mid < 0.0:
            tilt_hi, f_hi = tilt_mid, f_mid
        else:
            tilt_lo, f_lo = tilt_mid, f_mid

    return 0.5 * (tilt_lo + tilt_hi)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scenario(
        distance_m, tilt_deg, temp_c, pressure_hpa,
        wind_x, wind_z,
        bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
        use_mach_table: bool = False,
) -> dict:
    """One trajectory + bisection elevation solve.  Returns all derived outputs."""
    main = simulate_3d(distance_m, tilt_deg, temp_c, pressure_hpa,
                       wind_x, wind_z,
                       bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
                       use_mach_table=use_mach_table)

    tof   = main["tof"]
    imp_y = main["impact_y"]
    imp_z = main["impact_z"]
    v_imp = main["impact_velocity"]
    e_imp = main["impact_energy"]

    drop       = distance_m * np.tan(np.radians(tilt_deg)) - imp_y
    wind_drift = imp_z

    required_elev = find_required_elevation(
        distance_m, temp_c, pressure_hpa,
        wind_x, wind_z,
        bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
        use_mach_table=use_mach_table,
    )

    return dict(
        tof=tof,
        impact_y=imp_y,
        impact_z=imp_z,
        drop=drop,
        wind_drift=wind_drift,
        required_elev=required_elev,
        vert_error_deg=tilt_deg - required_elev,
        vert_error_m=imp_y,
        impact_velocity=v_imp,
        impact_energy=e_imp,
        impact_mach=main["impact_mach"],
        avg_cd=main["avg_cd"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PK — Monte Carlo hit probability
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pk(
        distance_m:        float,
        lead_m:            float,
        target_radius_m:   float,
        burst_size:        int,
        dispersion_mrad:   float,
        tracking_error_m:  float,
        lead_error_coeff:  float,
        num_trials:        int,
        rng,
) -> dict:
    """
    Vectorized Monte Carlo Pk estimate for one (distance, lead) combination.

    Error model
    -----------
    dispersion_sigma_m   = distance_m × tan(dispersion_mrad / 1000)
    eff_tracking_error_m = tracking_error_m + lead_m × lead_error_coeff
    total_sigma_m        = sqrt(dispersion² + eff_tracking²)

    Each trial: burst_size rounds with independent normal X/Y errors.
    Trial is a hit if ANY round lands within target_radius_m.
    Pk (%) = successful_trials / num_trials × 100
    """
    dispersion_sigma_m   = distance_m * np.tan(dispersion_mrad / 1000.0)
    eff_tracking_error_m = tracking_error_m + abs(lead_m) * lead_error_coeff
    total_sigma_m        = np.sqrt(dispersion_sigma_m**2 + eff_tracking_error_m**2)

    errors_x = rng.normal(0.0, total_sigma_m, size=(num_trials, burst_size))
    errors_y = rng.normal(0.0, total_sigma_m, size=(num_trials, burst_size))
    hit_dist = np.sqrt(errors_x**2 + errors_y**2)
    trial_success = np.any(hit_dist <= target_radius_m, axis=1)

    return {
        "pk_percent":                float(np.mean(trial_success) * 100.0),
        "dispersion_sigma_m":        float(dispersion_sigma_m),
        "effective_tracking_error_m": float(eff_tracking_error_m),
        "total_sigma_m":             float(total_sigma_m),
    }


def _pk_row(pk_r: dict, burst_size: int, target_radius_m: float) -> dict:
    """Build the Pk column dict from a compute_pk result."""
    return {
        "Pk (%)":                  round(pk_r["pk_percent"],                    2),
        "Dispersion Sigma (m)":    round(pk_r["dispersion_sigma_m"],            3),
        "Tracking Error Eff. (m)": round(pk_r["effective_tracking_error_m"],    3),
        "Total Sigma (m)":         round(pk_r["total_sigma_m"],                 3),
        "Burst Size":              burst_size,
        "Target Radius (m)":       round(target_radius_m,                       2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SWEEP — Tek Tilt (1–500 km/h)
# ═══════════════════════════════════════════════════════════════════════════════

def sweep(
        distance_m, tilt_deg, temp_c, pressure_hpa,
        wind_x, wind_z,
        bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
        use_mach_table:   bool  = False,
        do_pk:            bool  = False,
        target_radius_m:  float = 1.0,
        burst_size:       int   = 25,
        dispersion_mrad:  float = 1.5,
        tracking_error_m: float = 0.5,
        lead_error_coeff: float = 0.02,
        num_trials:       int   = 1000,
        seed:             int   = 42,
) -> pd.DataFrame:
    """Trajectory computed ONCE.  Only Lead (and Pk) varies with target speed."""
    s = compute_scenario(
        distance_m, tilt_deg, temp_c, pressure_hpa,
        wind_x, wind_z,
        bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
        use_mach_table=use_mach_table,
    )

    rng = np.random.default_rng(seed) if do_pk else None
    rows = []
    for spd in range(1, 501):
        lead = (spd / 3.6) * s["tof"]
        row = {
            "Hedef Hızı (km/h)":      spd,
            "TOF (s)":                round(s["tof"],            4),
            "Lead (m)":               round(lead,                3),
            "Drop (m)":               round(s["drop"],           3),
            "Wind Drift (m)":         round(s["wind_drift"],     3),
            "Impact Y (m)":           round(s["impact_y"],       3),
            "Vertical Error (m)":     round(s["vert_error_m"],   3),
            "Required Elevation (°)": round(s["required_elev"],  4),
            "Girilen Tilt (°)":       round(tilt_deg,            4),
            "Dikey Hata (°)":         round(s["vert_error_deg"], 4),
            "Impact Velocity (m/s)":  round(s["impact_velocity"],2),
            "Impact Energy (J)":      round(s["impact_energy"],  1),
        }
        if do_pk:
            pk_r = compute_pk(
                distance_m, lead, target_radius_m,
                burst_size, dispersion_mrad,
                tracking_error_m, lead_error_coeff,
                num_trials, rng,
            )
            row.update(_pk_row(pk_r, burst_size, target_radius_m))
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SWEEP — Tilt Aralığı
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_tilt_range(
        distance_m,
        tilt_start, tilt_end, tilt_step,
        temp_c, pressure_hpa, wind_x, wind_z,
        bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
        use_mach_table:   bool  = False,
        do_pk:            bool  = False,
        target_radius_m:  float = 1.0,
        burst_size:       int   = 25,
        dispersion_mrad:  float = 1.5,
        tracking_error_m: float = 0.5,
        lead_error_coeff: float = 0.02,
        num_trials:       int   = 1000,
        seed:             int   = 42,
) -> pd.DataFrame:
    """
    Float-safe tilt generation:
        count       = int(round((tilt_end - tilt_start) / tilt_step)) + 1
        tilt_values = [tilt_start + i * tilt_step  for i in range(count)]
    Trajectory computed ONCE per tilt; Lead (and Pk) computed per speed.
    """
    count       = int(round((tilt_end - tilt_start) / tilt_step)) + 1
    tilt_values = [tilt_start + i * tilt_step for i in range(count)]

    rng = np.random.default_rng(seed) if do_pk else None
    rows = []

    for tilt in tilt_values:
        s = compute_scenario(
            distance_m, tilt, temp_c, pressure_hpa,
            wind_x, wind_z,
            bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
            use_mach_table=use_mach_table,
        )
        tof_r   = round(s["tof"],            4)
        drop_r  = round(s["drop"],           3)
        drift_r = round(s["wind_drift"],     3)
        impy_r  = round(s["impact_y"],       3)
        ve_m_r  = round(s["vert_error_m"],   3)
        req_r   = round(s["required_elev"],  4)
        tilt_r  = round(tilt,               4)
        dha_r   = round(s["vert_error_deg"], 4)
        vmp_r   = round(s["impact_velocity"],2)
        ene_r   = round(s["impact_energy"],  1)

        for spd in range(1, 501):
            lead = (spd / 3.6) * s["tof"]
            row = {
                "Hedef Hızı (km/h)":      spd,
                "Tilt (°)":               tilt_r,
                "TOF (s)":                tof_r,
                "Lead (m)":               round(lead, 3),
                "Drop (m)":               drop_r,
                "Wind Drift (m)":         drift_r,
                "Impact Y (m)":           impy_r,
                "Vertical Error (m)":     ve_m_r,
                "Required Elevation (°)": req_r,
                "Dikey Hata (°)":         dha_r,
                "Impact Velocity (m/s)":  vmp_r,
                "Impact Energy (J)":      ene_r,
            }
            if do_pk:
                pk_r = compute_pk(
                    distance_m, lead, target_radius_m,
                    burst_size, dispersion_mrad,
                    tracking_error_m, lead_error_coeff,
                    num_trials, rng,
                )
                row.update(_pk_row(pk_r, burst_size, target_radius_m))
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(df, x_col, y_col, color, ylabel):
    """Dark-themed matplotlib figure.  Caller is responsible for plt.close(fig)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    ax.plot(df[x_col], df[y_col], color=color, linewidth=1.8)
    ax.set_xlabel(x_col, color="#8b949e", fontsize=9)
    ax.set_ylabel(ylabel,  color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linestyle="--", linewidth=0.6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout(pad=0.8)
    return fig


def _show_fig(fig):
    """Render figure and release memory."""
    st.pyplot(fig)
    plt.close(fig)


def metric_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}'
        f'  <span class="metric-unit">{unit}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🎯 Ballistic Lead Calculator")
st.markdown(
    '<div class="subtitle">Target Intercept &amp; Elevation Analysis'
    ' &nbsp;·&nbsp; 3-DOF Point-Mass · RK4 · Relative-Air-Velocity Drag</div>',
    unsafe_allow_html=True,
)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Mühimmat ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mühimmat</div>', unsafe_allow_html=True)
    bullet_name = st.text_input("Mermi Adı",         value="Default 25mm")
    bullet_mass = st.number_input("Ağırlık (g)",      value=185.0, min_value=1.0,   step=0.1)
    bullet_diam = st.number_input("Çap (mm)",         value=25.0,  min_value=1.0,   step=0.1)
    muzzle_vel  = st.number_input("Namlu Hızı (m/s)", value=1000.0, min_value=1.0,  step=1.0)
    cd_val      = st.number_input(
        "Cd (sabit)", value=0.29, min_value=0.01, step=0.01, format="%.3f",
        help="Mach Dependent Cd kapalıyken bu değer kullanılır.",
    )
    use_mach_table = st.checkbox(
        "Mach Dependent Cd", value=False,
        help="Açıksa MACH_TABLE/CD_TABLE ile np.interp. Kapalıysa sabit Cd.",
    )

    # ── Analiz Modu ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Analiz Modu</div>', unsafe_allow_html=True)
    analysis_mode = st.radio(
        "", ["Tek Tilt", "Tilt Aralığı"],
        index=0, key="analysis_mode",
        label_visibility="collapsed",
    )

    # ── Koşullar ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Koşullar</div>', unsafe_allow_html=True)
    distance = st.number_input("Mesafe (m)", value=1000.0, min_value=10.0, step=10.0)

    if analysis_mode == "Tek Tilt":
        tilt_angle = st.number_input(
            "Tilt / Elevation (°)", value=2.0,
            min_value=-45.0, max_value=45.0, step=0.1,
        )
    else:
        tilt_start = st.number_input("Başlangıç Tilt (°)", value=1.0,  min_value=-45.0, max_value=45.0, step=0.1)
        tilt_end   = st.number_input("Bitiş Tilt (°)",     value=15.0, min_value=-45.0, max_value=45.0, step=0.1)
        tilt_step  = st.number_input("Tilt Adımı (°)",     value=0.5,  min_value=0.01,  max_value=10.0, step=0.1, format="%.2f")

    temperature = st.number_input("Sıcaklık (°C)", value=15.0,   min_value=-60.0, max_value=60.0, step=0.5)
    pressure    = st.number_input("Basınç (hPa)",  value=1013.25, min_value=800.0, step=0.5)

    # ── Rüzgar ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Rüzgar</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        '<b>Head/Tail Wind</b>: x ekseninde, mermi hareket yönünde.<br>'
        '<b>Crosswind</b>: z ekseninde, yanal sapma oluşturur.'
        '</div>',
        unsafe_allow_html=True,
    )
    ht_spd = st.number_input("Head/Tail Wind (m/s)", value=0.0, min_value=0.0, step=0.1)
    ht_dir = st.radio(
        "Head/Tail Yön",
        ["Karşıdan — Headwind (−x)", "Arkadan — Tailwind (+x)"],
        index=0, key="ht_dir",
    )
    wind_x = -ht_spd if ht_dir.startswith("Karşıdan") else +ht_spd

    cw_spd = st.number_input("Crosswind (m/s)", value=0.0, min_value=0.0, step=0.1)
    cw_dir = st.radio(
        "Crosswind Yönü",
        ["+Z yönünde (sağ sapma)", "−Z yönünde (sol sapma)"],
        index=0, key="cw_dir",
    )
    wind_z = +cw_spd if cw_dir.startswith("+Z") else -cw_spd

    # ── Pk / Vurulma Modeli ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pk / Vurulma Modeli</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        '<b>Basitleştirilmiş istatistiksel model:</b> Dairesel hedef geometrisi, '
        'bağımsız normal dağılımlı X/Y hata, burst içinde ≥1 isabet = başarılı deneme.'
        '</div>',
        unsafe_allow_html=True,
    )
    do_pk = st.checkbox("Pk hesapla", value=False, key="do_pk")

    target_radius  = st.number_input("Hedef yarıçapı (m)",           value=1.0,  min_value=0.1,  step=0.1, format="%.2f")
    burst_size     = st.number_input("Burst mermi sayısı",            value=25,   min_value=1,    step=1)
    disp_mrad      = st.number_input("Dispersion sigma (mrad)",       value=1.5,  min_value=0.01, step=0.1, format="%.2f")
    track_err      = st.number_input("Tracking/Radar error sigma (m)", value=0.5, min_value=0.0,  step=0.1, format="%.2f")
    lead_err_coef  = st.number_input("Lead error coefficient",        value=0.02, min_value=0.0,  step=0.005, format="%.3f")
    num_trials     = st.number_input("Monte Carlo deneme sayısı",     value=1000, min_value=10,   max_value=50000, step=100)
    pk_seed        = st.number_input("Random Seed",                   value=42,   min_value=0,    step=1)

    st.markdown("")
    calc_btn = st.button("⚡ Hesapla")


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

for _k in ("df", "mode", "mach_info"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════════

if calc_btn:

    _pk_kwargs = dict(
        do_pk=do_pk,
        target_radius_m=float(target_radius),
        burst_size=int(burst_size),
        dispersion_mrad=float(disp_mrad),
        tracking_error_m=float(track_err),
        lead_error_coeff=float(lead_err_coef),
        num_trials=int(num_trials),
        seed=int(pk_seed),
    )

    if analysis_mode == "Tek Tilt":
        _label = "Simülasyon çalışıyor (3D · RK4" + (" · Pk MC" if do_pk else "") + ")…"
        with st.spinner(_label):
            df_result = sweep(
                distance, tilt_angle, temperature, pressure,
                wind_x, wind_z,
                bullet_mass, bullet_diam, muzzle_vel, cd_val,
                use_mach_table=use_mach_table,
                **_pk_kwargs,
            )
            _, c_snd = atmosphere(temperature, pressure)
            initial_mach = muzzle_vel / c_snd
            _sim = simulate_3d(
                distance, tilt_angle, temperature, pressure,
                wind_x, wind_z,
                bullet_mass, bullet_diam, muzzle_vel, cd_val,
                use_mach_table=use_mach_table,
            )
        st.session_state["df"]        = df_result
        st.session_state["mode"]      = "Tek Tilt"
        st.session_state["mach_info"] = {
            "initial_mach": round(initial_mach,        3),
            "impact_mach":  round(_sim["impact_mach"],  3),
            "avg_cd":       round(_sim["avg_cd"],        4),
        }

    else:  # Tilt Aralığı
        valid = True
        if tilt_end < tilt_start:
            st.error("⚠ Bitiş Tilt, Başlangıç Tilt'ten küçük olamaz.")
            valid = False
        if tilt_step <= 0:
            st.error("⚠ Tilt Adımı sıfırdan büyük olmalıdır.")
            valid = False

        if valid:
            n_tilts = int(round((tilt_end - tilt_start) / tilt_step)) + 1
            _label = (
                f"Simülasyon çalışıyor ({n_tilts} tilt · 3D · RK4"
                + (" · Pk MC" if do_pk else "") + ")…"
            )
            with st.spinner(_label):
                df_result = sweep_tilt_range(
                    distance, tilt_start, tilt_end, tilt_step,
                    temperature, pressure, wind_x, wind_z,
                    bullet_mass, bullet_diam, muzzle_vel, cd_val,
                    use_mach_table=use_mach_table,
                    **_pk_kwargs,
                )
            st.session_state["df"]        = df_result
            st.session_state["mode"]      = "Tilt Aralığı"
            st.session_state["mach_info"] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SONUÇLAR
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state["df"] is None:
    st.info("Sol panelden parametreleri girin ve **⚡ Hesapla** butonuna basın.")
    st.stop()

df   = st.session_state["df"]
mode = st.session_state["mode"]

_has_pk = "Pk (%)" in df.columns

# ── TABLE FORMAT ──────────────────────────────────────────────────────────────
BASE_FMT: dict = {
    "TOF (s)":                 "{:.4f}",
    "Lead (m)":                "{:.3f}",
    "Drop (m)":                "{:.3f}",
    "Wind Drift (m)":          "{:.3f}",
    "Impact Y (m)":            "{:.3f}",
    "Vertical Error (m)":      "{:.3f}",
    "Required Elevation (°)":  "{:.4f}",
    "Dikey Hata (°)":          "{:.4f}",
    "Impact Velocity (m/s)":   "{:.2f}",
    "Impact Energy (J)":       "{:.1f}",
    "Pk (%)":                  "{:.2f}",
    "Dispersion Sigma (m)":    "{:.3f}",
    "Tracking Error Eff. (m)": "{:.3f}",
    "Total Sigma (m)":         "{:.3f}",
    "Target Radius (m)":       "{:.2f}",
}


# ─────────────────────────────────────────────────────────────────────────────
#  TEK TILT MODE
# ─────────────────────────────────────────────────────────────────────────────

if mode == "Tek Tilt":

    s        = df.iloc[0]
    lead_100 = float(df.loc[df["Hedef Hızı (km/h)"] == 100, "Lead (m)"].iloc[0])
    mi       = st.session_state.get("mach_info") or {}

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Senaryo Sonuçları</div>',
                unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns(3)
    r2a, r2b, r2c = st.columns(3)
    r3a, r3b, r3c = st.columns(3)
    r4a, r4b, r4c = st.columns(3)

    with r1a: metric_card("TOF",                 f"{s['TOF (s)']:.4f}",                    "s")
    with r1b: metric_card("Lead @ 100 km/h",      f"{lead_100:.3f}",                         "m")
    with r1c: metric_card("Drop",                 f"{s['Drop (m)']:.3f}",                    "m")
    with r2a: metric_card("Wind Drift (z-axis)",  f"{s['Wind Drift (m)']:.3f}",              "m")
    with r2b: metric_card("Required Elevation",   f"{s['Required Elevation (°)']:.4f}",      "°")
    with r2c: metric_card("Vertical Error",       f"{s['Vertical Error (m)']:.3f}",          "m")
    with r3a: metric_card("Initial Mach",         f"{mi.get('initial_mach', '—')}",          "")
    with r3b: metric_card("Impact Mach",          f"{mi.get('impact_mach', '—')}",           "")
    with r3c: metric_card("Ortalama Cd",          f"{mi.get('avg_cd', '—')}",                "")
    with r4a: metric_card("Impact Velocity",      f"{s['Impact Velocity (m/s)']:.2f}",       "m/s")
    with r4b: metric_card("Impact Energy",        f"{s['Impact Energy (J)']:.1f}",           "J")

    if _has_pk:
        row_100   = df.loc[df["Hedef Hızı (km/h)"] == 100].iloc[0]
        pk_100    = float(row_100["Pk (%)"])
        sigma_100 = float(row_100["Total Sigma (m)"])
        with r4c: metric_card("Pk @ 100 km/h",  f"{pk_100:.2f}",   "%")
        r5a, r5b, _ = st.columns(3)
        with r5a: metric_card("Total Sigma @ 100 km/h", f"{sigma_100:.3f}", "m")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Grafikler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'TOF, Drop, Wind Drift, Required Elevation ve Impact Velocity hedef hızından bağımsızdır. '
        'Sadece <b>Lead = v_hedef × TOF</b> doğrusal olarak değişir.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    with col1:
        st.markdown("**Lead vs Hedef Hızı**")
        _show_fig(make_figure(df, "Hedef Hızı (km/h)", "Lead (m)",               "#58a6ff", "Lead (m)"))
    with col2:
        st.markdown("**Required Elevation vs Hedef Hızı**")
        _show_fig(make_figure(df, "Hedef Hızı (km/h)", "Required Elevation (°)", "#f78166", "Elevation (°)"))
    with col3:
        st.markdown("**Drop vs Hedef Hızı**")
        _show_fig(make_figure(df, "Hedef Hızı (km/h)", "Drop (m)",               "#3fb950", "Drop (m)"))
    with col4:
        st.markdown("**Wind Drift vs Hedef Hızı**")
        _show_fig(make_figure(df, "Hedef Hızı (km/h)", "Wind Drift (m)",         "#d2a8ff", "Wind Drift (m)"))
    with col5:
        st.markdown("**Impact Velocity vs Hedef Hızı**")
        _show_fig(make_figure(df, "Hedef Hızı (km/h)", "Impact Velocity (m/s)", "#ffa657", "Vel (m/s)"))

    if _has_pk:
        with col6:
            st.markdown("**Pk vs Hedef Hızı**")
            _show_fig(make_figure(df, "Hedef Hızı (km/h)", "Pk (%)", "#e3b341", "Pk (%)"))

    st.divider()

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sonuç Tablosu (1–500 km/h)</div>',
                unsafe_allow_html=True)
    search = st.text_input("Hız filtrele (km/h):", placeholder="örn. 100",
                           key="search_single")
    show_df = df.copy()
    if search.strip():
        try:
            show_df = show_df[show_df["Hedef Hızı (km/h)"] == float(search.strip())]
        except ValueError:
            pass

    fmt = {k: v for k, v in BASE_FMT.items() if k in show_df.columns}
    fmt["Girilen Tilt (°)"] = "{:.4f}"
    st.dataframe(show_df.style.format(fmt), use_container_width=True, height=430)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ sonuc.csv İndir", csv_bytes, "sonuc.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
#  TILT ARALIĞI MODE
# ─────────────────────────────────────────────────────────────────────────────

else:

    n_tilts  = df["Tilt (°)"].nunique()
    n_rows   = len(df)
    tmin     = df["Tilt (°)"].min()
    tmax     = df["Tilt (°)"].max()
    tof_min  = df["TOF (s)"].min()
    tof_max  = df["TOF (s)"].max()
    drop_min = df["Drop (m)"].min()
    drop_max = df["Drop (m)"].max()

    st.markdown(
        f'<div class="section-header">Tilt Aralığı Sonuçları '
        f'({n_tilts} tilt · {n_rows:,} kombinasyon)</div>',
        unsafe_allow_html=True,
    )

    r1a, r1b, r1c = st.columns(3)
    with r1a: metric_card("Tilt Aralığı", f"{tmin:.2f} – {tmax:.2f}",         "°")
    with r1b: metric_card("TOF Aralığı",  f"{tof_min:.4f} – {tof_max:.4f}",   "s")
    with r1c: metric_card("Drop Aralığı", f"{drop_min:.3f} – {drop_max:.3f}", "m")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Grafikler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Impact Y, Drop ve Impact Velocity hedef hızından bağımsızdır. '
        'Seçilen hız tabloyu filtreler ve Lead/Pk değerleri o hıza göre hesaplanmıştır.'
        '</div>',
        unsafe_allow_html=True,
    )

    viz_speed = st.number_input(
        "Grafikte hedef hızı (km/h)", value=100,
        min_value=1, max_value=500, step=1, key="viz_speed",
    )
    chart_df = (
        df[df["Hedef Hızı (km/h)"] == viz_speed]
        .sort_values("Tilt (°)")
        .reset_index(drop=True)
    )

    if chart_df.empty:
        st.warning(f"{viz_speed} km/h için veri bulunamadı.")
    else:
        gc1, gc2 = st.columns(2)
        gc3, gc4 = st.columns(2)

        with gc1:
            st.markdown(f"**Tilt vs Impact Y** *(hedef hızı {viz_speed} km/h)*")
            _show_fig(make_figure(chart_df, "Tilt (°)", "Impact Y (m)",         "#58a6ff", "Impact Y (m)"))
        with gc2:
            st.markdown("**Tilt vs Vertical Error**")
            _show_fig(make_figure(chart_df, "Tilt (°)", "Vertical Error (m)",   "#f78166", "Vert. Error (m)"))
        with gc3:
            st.markdown("**Tilt vs Drop**")
            _show_fig(make_figure(chart_df, "Tilt (°)", "Drop (m)",             "#3fb950", "Drop (m)"))
        with gc4:
            st.markdown("**Tilt vs Impact Velocity**")
            _show_fig(make_figure(chart_df, "Tilt (°)", "Impact Velocity (m/s)", "#ffa657", "Vel (m/s)"))

        if _has_pk:
            gc5, gc6 = st.columns(2)
            with gc5:
                st.markdown(f"**Tilt vs Pk (%)** *(hedef hızı {viz_speed} km/h)*")
                _show_fig(make_figure(chart_df, "Tilt (°)", "Pk (%)",          "#e3b341", "Pk (%)"))
            with gc6:
                st.markdown("**Tilt vs Total Sigma (m)**")
                _show_fig(make_figure(chart_df, "Tilt (°)", "Total Sigma (m)", "#79c0ff", "Total Sigma (m)"))

    st.divider()

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sonuç Tablosu</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Hız filtresi: belirli bir hedef hızındaki tüm tilt değerlerini gösterir.'
        '</div>',
        unsafe_allow_html=True,
    )
    search = st.text_input("Hız filtrele (km/h):", placeholder="örn. 100",
                           key="search_range")
    show_df = df.copy()
    if search.strip():
        try:
            show_df = show_df[show_df["Hedef Hızı (km/h)"] == float(search.strip())]
        except ValueError:
            pass

    fmt = {k: v for k, v in BASE_FMT.items() if k in show_df.columns}
    fmt["Tilt (°)"] = "{:.4f}"
    st.dataframe(show_df.style.format(fmt), use_container_width=True, height=430)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ sonuc.csv İndir", csv_bytes, "sonuc.csv", "text/csv")
