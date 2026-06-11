"""
Ballistic Lead Calculator
3-DOF point-mass model · RK4 integration · relative-air-velocity drag
"""

import base64
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
import plotly.graph_objects as go

_logo     = Image.open("logo.png")
_logo_b64 = base64.b64encode(open("logo.png", "rb").read()).decode()

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
#  AMMO DATABASE
#  Yeni mühimmat eklemek için buraya giriş ekleyin — başka hiçbir yerde
#  sabit mühimmat ismi kullanılmamaktadır.
# ═══════════════════════════════════════════════════════════════════════════════

AMMO_DB = {
    "M53 API": {
        "display":        "M53 API — Armor Piercing Incendiary",
        "type":           "API (Armor Piercing Incendiary)",
        "description_tr": "Zırh delici, yangın etkili",
        # mass_g = projectile weight — balistik hesapta kullanılan değer
        "mass_g":              105.0,
        "diam_mm":             20.0,
        "length_mm":           168.0,
        "muzzle_ms":           1051.56,
        "cd":                  0.29,
        "cartridge_weight_g":  260.0,
        "projectile_weight_g": 105.0,
        "max_pressure_psi":    60500,
        "armor_penetration":   "100 m mesafede, 25° eğimde 20 mm zırh delme",
        "accuracy":            "600 yd mesafede ortalama 15 inç yarıçap",
    },
    "M56 A3 HEI": {
        "display":        "M56 A3 HEI — High Explosive Incendiary",
        "type":           "HEI (High Explosive Incendiary)",
        "description_tr": "Yüksek patlayıcılı, yangın etkili",
        # mass_g = projectile weight — balistik hesapta kullanılan değer
        "mass_g":              101.0,
        "diam_mm":             20.0,
        "length_mm":           168.02,
        "muzzle_ms":           1030.0,
        "cd":                  0.29,
        "cartridge_weight_g":  258.0,
        "projectile_weight_g": 101.0,
        "muzzle_tolerance_ms":      "±15 m/s",
        "velocity_std_ms":          12.19,
        "avg_pressure_kg_cm2":      4254,
        "dispersion":               "232.4 m mesafede maksimum 16.15 cm",
        "bullet_pull_force_kgf":    "499–1179 kgf",
        "movement_time_ms":         "maksimum 4 ms",
        "case_model":               "M103",
        "case_material":            "Pirinç (CuZn30)",
        "projectile_body_material": "Çelik (Ç1040)",
        "fuze":                     "M505 A3",
        "primer":                   "M52 A3 B1 Elektrikli Kapsül",
        "link_type":                "M12 veya M14 Mayon",
        "propellant":               "Küresel Barut",
        "weapon":                   "M39, M61, M197",
    },
    "M55 A2 TP": {
        "display":        "M55 A2 TP — Target Practice",
        "type":           "TP (Target Practice)",
        "description_tr": "Eğitim ve hedef atışı mühimmatı",
        # mass_g = projectile weight — balistik hesapta kullanılan değer
        "mass_g":              99.0,
        "diam_mm":             20.0,
        "length_mm":           168.0,
        "muzzle_ms":           1030.22,
        "cd":                  0.29,
        "cartridge_weight_g":  255.0,
        "projectile_weight_g": 99.0,
        "propellant_weight_g": 43.0,
        "case_length_mm":      102.0,
        "case_material":       "Brass",
        "primer":              "M52A3B1 electric",
        "chamber_pressure_psi": 60500,
        "dispersion":           "15 in @ 600 yd",
    },
    # ── Buraya ek mühimmatlar eklenebilir ──────────────────────────────────
}


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
#  TRAJECTORY POINTS — for 3D animation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_trajectory_points(
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
        aim_z_m:        float = 0.0,
) -> dict:
    """
    Same RK4 physics as simulate_3d but collects every step.
    aim_z_m: lateral lead distance — adds yaw angle so v0z points toward lead.
      yaw_rad = atan2(aim_z_m, distance_m)
      v0x = V * cos(tilt) * cos(yaw)
      v0y = V * sin(tilt)
      v0z = V * cos(tilt) * sin(yaw)
    Returns {"t", "x", "y", "z", "tof", "impact_y", "impact_z"}.
    """
    rho, c = atmosphere(temp_c, pressure_hpa)
    mass   = bullet_mass_g / 1000.0
    area   = np.pi * (bullet_diam_mm / 2000.0) ** 2
    wind_vec = np.array([wind_x, 0.0, wind_z])
    kw = dict(wind_vec=wind_vec, rho=rho, c=c,
              cd_base=cd_base, area=area, mass=mass,
              use_mach_table=use_mach_table)

    _elev = np.radians(tilt_deg)
    if abs(aim_z_m) > 1e-9:
        _yaw = np.arctan2(aim_z_m, distance_m)
        v0x  = muzzle_vel_ms * np.cos(_elev) * np.cos(_yaw)
        v0y  = muzzle_vel_ms * np.sin(_elev)
        v0z  = muzzle_vel_ms * np.cos(_elev) * np.sin(_yaw)
    else:
        v0x = muzzle_vel_ms * np.cos(_elev)
        v0y = muzzle_vel_ms * np.sin(_elev)
        v0z = 0.0
    state = np.array([0.0, 0.0, 0.0, v0x, v0y, v0z])

    t_list = [0.0]; x_list = [0.0]; y_list = [0.0]; z_list = [0.0]
    t = 0.0

    for _ in range(500_000):
        prev_state = state.copy()
        prev_t     = t
        state = _rk4_step(state, dt, **kw)
        t    += dt

        if state[0] >= distance_m:
            dx   = state[0] - prev_state[0]
            frac = (distance_m - prev_state[0]) / dx if dx > 1e-12 else 1.0
            imp  = prev_state + frac * (state - prev_state)
            tof  = prev_t + frac * dt
            t_list.append(tof)
            x_list.append(float(imp[0]))
            y_list.append(float(imp[1]))
            z_list.append(float(imp[2]))
            return {
                "t": t_list, "x": x_list, "y": y_list, "z": z_list,
                "tof": tof, "impact_y": float(imp[1]), "impact_z": float(imp[2]),
            }

        t_list.append(t)
        x_list.append(float(state[0]))
        y_list.append(float(state[1]))
        z_list.append(float(state[2]))

        if state[1] < -(distance_m + 200.0):
            break

    return {
        "t": t_list, "x": x_list, "y": y_list, "z": z_list,
        "tof": t, "impact_y": float(state[1]), "impact_z": float(state[2]),
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


def make_3d_animation(
        traj, tof, impact_y, impact_z, lead_m,
        distance_m, target_speed_ms, wind_x, wind_z,
        required_elev, drop, wind_drift,
        ammo_name, speed_kmh, tilt_deg,
        entered_tilt_deg=None,
        scale_mode: str = "visual",
) -> go.Figure:
    """
    Animated Plotly 3D figure.
    Coordinate mapping to Plotly axes:
      X = downrange (physics x)
      Y = lateral   (physics z)
      Z = vertical  (physics y)

    scale_mode = "real"   → aspectmode="data"   (true physical proportions)
    scale_mode = "visual" → aspectmode="manual" (exaggerated Y/Z for readability)

    uirevision="keep-camera" prevents Play/Pause/slider from resetting the
    user's chosen viewpoint. Frames contain NO layout/camera keys.
    """
    if entered_tilt_deg is None:
        entered_tilt_deg = tilt_deg

    t_arr = np.array(traj["t"])
    x_arr = np.array(traj["x"])
    y_arr = np.array(traj["y"])   # physics vertical  → Plotly Z
    z_arr = np.array(traj["z"])   # physics lateral   → Plotly Y
    n = len(t_arr)

    # ── Aspect ratio & camera based on scale mode ─────────────────────────────
    if scale_mode == "real":
        _aspect_mode = "data"
        _ar = dict(x=1.0, y=1.0, z=1.0)   # ignored by "data" mode
        _init_cam = dict(eye=dict(x=1.6, y=1.2, z=0.8))
        _mode_note = (
            "Bu görünüm eksenleri fiziksel oranda gösterir; "
            "küçük Y/Z sapmaları zor görünebilir."
        )
    else:  # "visual"
        _aspect_mode = "manual"
        _x_r = 4.0
        _z_r = max(1.2, min(2.5, abs(lead_m) / max(distance_m, 1.0) * 8.0))
        _y_r = 1.2
        _ar = dict(x=_x_r, y=_y_r, z=_z_r)
        _init_cam = dict(eye=dict(x=1.8, y=1.4, z=0.9))
        _mode_note = (
            "Bu görünüm okunabilirlik için Y/Z eksenlerini görsel olarak büyütür; "
            "fiziksel ölçüler eksen değerlerinde doğrudur."
        )

    step = max(1, n // 80)
    frame_idx = list(range(0, n, step))
    if frame_idx[-1] != n - 1:
        frame_idx.append(n - 1)

    # ── Static traces ─────────────────────────────────────────────────────────
    ghost_bullet = go.Scatter3d(
        x=x_arr, y=z_arr, z=y_arr, mode="lines",
        line=dict(color="#58a6ff", width=2, dash="dash"), opacity=0.35,
        name="Mermi Yolu",
    )
    ghost_target = go.Scatter3d(
        x=[distance_m, distance_m], y=[0.0, lead_m], z=[0.0, 0.0],
        mode="lines+markers",
        line=dict(color="#f78166", width=2, dash="dash"), opacity=0.35,
        marker=dict(color="#f78166", size=4),
        name="Hedef Yolu",
    )
    lead_pt = go.Scatter3d(
        x=[distance_m], y=[lead_m], z=[0.0], mode="markers",
        marker=dict(symbol="diamond", color="#e3b341", size=10),
        name=f"Lead ({lead_m:.2f} m)",
    )
    impact_pt = go.Scatter3d(
        x=[distance_m], y=[impact_z], z=[impact_y], mode="markers",
        marker=dict(symbol="x", color="#ff7b72", size=10),
        name=f"İmpact Y={impact_y:.2f} m  Z={impact_z:.2f} m",
    )
    vmiss = go.Scatter3d(
        x=[distance_m, distance_m], y=[impact_z, impact_z], z=[0.0, impact_y],
        mode="lines",
        line=dict(color="#ff7b72", width=2, dash="dot"),
        name=f"Vertical Miss {impact_y:.2f} m",
    )
    # Animated traces — indices 5 & 6 (matched in every frame)
    anim_bullet = go.Scatter3d(
        x=[x_arr[0]], y=[z_arr[0]], z=[y_arr[0]], mode="markers+lines",
        marker=dict(color="#58a6ff", size=7),
        line=dict(color="#58a6ff", width=3), name="Mermi",
    )
    anim_target = go.Scatter3d(
        x=[distance_m], y=[0.0], z=[0.0], mode="markers+lines",
        marker=dict(color="#f78166", size=7),
        line=dict(color="#f78166", width=3), name="Hedef",
    )

    extra = []
    if abs(wind_x) > 0.01 or abs(wind_z) > 0.01:
        wmag = np.sqrt(wind_x**2 + wind_z**2)
        wlen = min(distance_m * 0.12, 50.0)
        mx   = distance_m / 2
        extra.append(go.Scatter3d(
            x=[mx, mx + wind_x / wmag * wlen],
            y=[0.0, wind_z / wmag * wlen],
            z=[0.0, 0.0],
            mode="lines+text",
            line=dict(color="#79c0ff", width=4),
            text=["", f"Rüzgar {wmag:.1f} m/s"],
            textfont=dict(color="#79c0ff", size=10),
            name="Rüzgar",
        ))

    # ── Animation frames — only data, NO layout/camera keys ──────────────────
    # Omitting layout keys in frames is what keeps the user's camera intact.
    frames = []
    for fi in frame_idx:
        tb = max(0, fi - 20)
        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=x_arr[tb:fi+1], y=z_arr[tb:fi+1], z=y_arr[tb:fi+1],
                    mode="markers+lines",
                    marker=dict(color="#58a6ff", size=5),
                    line=dict(color="#58a6ff", width=3),
                ),
                go.Scatter3d(
                    x=np.full(fi - tb + 1, distance_m),
                    y=target_speed_ms * t_arr[tb:fi+1],
                    z=np.zeros(fi - tb + 1),
                    mode="markers+lines",
                    marker=dict(color="#f78166", size=5),
                    line=dict(color="#f78166", width=3),
                ),
            ],
            traces=[5, 6],
            name=str(fi),
        ))

    ann_text = (
        f"Girilen Tilt: {entered_tilt_deg:.4f}°  |  Required Elevation: {required_elev:.4f}°<br>"
        f"Vertical Miss (Impact Y): {impact_y:.3f} m  |  TOF: {tof:.4f} s<br>"
        f"Lead = Hedef Hızı × TOF: {lead_m:.3f} m  |  Lateral Aim Z: {lead_m:.3f} m<br>"
        f"Wind Drift: {wind_drift:.3f} m  |  Hedef Hızı: {speed_kmh} km/h<br>"
        f"<i>{_mode_note}</i>"
    )

    static_traces = [ghost_bullet, ghost_target, lead_pt, impact_pt, vmiss,
                     anim_bullet, anim_target] + extra

    # ── Camera presets (relayout — never reset user's current view) ───────────
    _cam_iso  = dict(eye=dict(x=1.5,   y=1.5,   z=0.8))
    _cam_top  = dict(eye=dict(x=0.01,  y=0.01,  z=2.5))
    _cam_side = dict(eye=dict(x=0.01,  y=-2.5,  z=0.4))
    _cam_rear = dict(eye=dict(x=-2.5,  y=0.01,  z=0.4))
    _cam_traj = dict(eye=dict(x=-2.0,  y=0.8,   z=0.5))   # behind shooter

    cam_menu = dict(
        type="buttons", showactive=False,
        direction="right",
        x=0.0, xanchor="left",
        y=1.08, yanchor="top",
        bgcolor="#161b22", bordercolor="#30363d",
        font=dict(color="#e6edf3", size=11),
        buttons=[
            dict(label="İzometrik",        method="relayout",
                 args=[{"scene.camera": _cam_iso}]),
            dict(label="Üstten",           method="relayout",
                 args=[{"scene.camera": _cam_top}]),
            dict(label="Yandan",           method="relayout",
                 args=[{"scene.camera": _cam_side}]),
            dict(label="Arkadan",          method="relayout",
                 args=[{"scene.camera": _cam_rear}]),
            dict(label="Mermiyi Takip Et", method="relayout",
                 args=[{"scene.camera": _cam_traj}]),
        ],
    )

    play_menu = dict(
        type="buttons", showactive=False,
        y=0, x=0.5, xanchor="center",
        bgcolor="#161b22", bordercolor="#30363d",
        font=dict(color="#e6edf3", size=12),
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": 30, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )

    fig = go.Figure(
        data=static_traces,
        frames=frames,
        layout=go.Layout(
            # uirevision keeps user's camera / zoom / pan across Plotly.react() calls.
            # Any constant string works — as long as it doesn't change between renders
            # the camera state is preserved when animation frames are applied.
            uirevision="keep-camera",
            title=dict(
                text=(f"3D Atış Simülasyonu — {ammo_name} — {speed_kmh} km/h "
                      f"— Tilt {entered_tilt_deg:.2f}°"),
                font=dict(color="#58a6ff", size=14),
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            scene=dict(
                bgcolor="#0d1117",
                aspectmode=_aspect_mode,
                aspectratio=_ar,
                camera=_init_cam,   # initial view only — uirevision preserves after that
                xaxis=dict(title="X — Downrange (m)", color="#8b949e",
                           gridcolor="#30363d", zerolinecolor="#30363d"),
                yaxis=dict(title="Z — Lateral (m)",   color="#8b949e",
                           gridcolor="#30363d", zerolinecolor="#30363d"),
                zaxis=dict(title="Y — Vertical (m)",  color="#8b949e",
                           gridcolor="#30363d", zerolinecolor="#30363d"),
            ),
            legend=dict(font=dict(color="#e6edf3"), bgcolor="#161b22",
                        bordercolor="#30363d"),
            annotations=[dict(
                text=ann_text, align="left", showarrow=False,
                xref="paper", yref="paper", x=0.01, y=0.98,
                bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                font=dict(color="#8b949e", size=11),
            )],
            updatemenus=[cam_menu, play_menu],
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Frame: "},
                pad={"t": 50},
                steps=[dict(
                    args=[[f.name],
                          {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    label=str(i), method="animate",
                ) for i, f in enumerate(frames)],
            )],
            height=660,
            margin=dict(l=0, r=0, t=80, b=80),
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — must be initialised BEFORE sidebar widgets render
# ═══════════════════════════════════════════════════════════════════════════════

_FIRST_KEY = next(iter(AMMO_DB))

if "ammo_sel_prev" not in st.session_state:
    _a0 = AMMO_DB[_FIRST_KEY]
    st.session_state.update({
        "ammo_selector":    _FIRST_KEY,
        "ammo_input_name":  _a0["display"],
        "ammo_input_mass":  _a0["mass_g"],
        "ammo_input_diam":  _a0["diam_mm"],
        "ammo_input_len":   _a0["length_mm"],
        "ammo_input_vel":   _a0["muzzle_ms"],
        "ammo_input_cd":    _a0["cd"],
        "ammo_sel_prev":    _FIRST_KEY,
    })

for _k in ("df", "mode", "mach_info", "ammo_snapshot", "sim_params",
           "sim3d_fig_single", "sim3d_fig_range"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

_hcol_logo, _hcol_title = st.columns([2, 8])
with _hcol_logo:
    st.markdown(
        f'<img src="data:image/png;base64,{_logo_b64}" '
        f'style="width:200px; margin-top:1.2rem; margin-left:2rem;" />',
        unsafe_allow_html=True,
    )
with _hcol_title:
    st.markdown("# Digitest Ballistic Lead Calculator")
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

    # ── Mühimmat Seçimi ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mühimmat Seçimi</div>',
                unsafe_allow_html=True)

    selected_ammo = st.selectbox(
        "Mühimmat", list(AMMO_DB.keys()), key="ammo_selector",
    )

    # Seçim değiştiğinde input alanlarını güncelle (widget'lar henüz render edilmedi)
    if selected_ammo != st.session_state["ammo_sel_prev"]:
        _a = AMMO_DB[selected_ammo]
        st.session_state.update({
            "ammo_input_name": _a["display"],
            "ammo_input_mass": _a["mass_g"],
            "ammo_input_diam": _a["diam_mm"],
            "ammo_input_len":  _a["length_mm"],
            "ammo_input_vel":  _a["muzzle_ms"],
            "ammo_input_cd":   _a["cd"],
            "ammo_sel_prev":   selected_ammo,
        })

    # Info kartı — seçili mühimmatın özellikleri (dinamik: sadece mevcut alanlar gösterilir)
    _ai = AMMO_DB[selected_ammo]

    def _ai_get(key, fmt=None):
        v = _ai.get(key)
        if v is None:
            return None
        return fmt.format(v) if fmt else str(v)

    _info_lines = [f"<b>{selected_ammo}</b>"]
    if _ai.get("type"):
        _info_lines.append(f"<b>Tip:</b> {_ai['type']}")
    if _ai.get("description_tr"):
        _info_lines.append(f"<b>Açıklama:</b> {_ai['description_tr']}")
    _info_lines.append("")  # boş satır

    # Fizik parametreleri — her zaman mevcut
    _phys = (
        f"<b>Mermi Ağırlığı (Balistik):</b> {_ai['mass_g']} g &nbsp;·&nbsp; "
        f"<b>Çap:</b> {_ai['diam_mm']} mm &nbsp;·&nbsp; "
        f"<b>Uzunluk:</b> {_ai['length_mm']} mm"
    )
    _info_lines.append(_phys)
    _vel_line = f"<b>Namlu Hızı:</b> {_ai['muzzle_ms']} m/s"
    if _ai.get("muzzle_tolerance_ms"):
        _vel_line += f"  (tolerans: {_ai['muzzle_tolerance_ms']})"
    _info_lines.append(_vel_line)
    if _ai.get("velocity_std_ms") is not None:
        _info_lines.append(f"<b>Hız Std Sapması:</b> {_ai['velocity_std_ms']} m/s")

    # İsteğe bağlı balistik/yapısal alanlar
    _OPTIONAL_FIELDS = [
        ("cartridge_weight_g",      "<b>Fişek Ağırlığı:</b> {} g"),
        ("projectile_weight_g",     "<b>Projectile Weight:</b> {} g"),
        ("propellant_weight_g",     "<b>Barut Ağırlığı:</b> {} g"),
        ("case_length_mm",          "<b>Kovan Boyu:</b> {} mm"),
        ("chamber_pressure_psi",    "<b>Namlu Basıncı:</b> {} psi"),
        ("max_pressure_psi",        "<b>Maks Basınç:</b> {} psi"),
        ("avg_pressure_kg_cm2",     "<b>Ort. Basınç:</b> {} kg/cm²"),
        ("dispersion",              "<b>Dağılım:</b> {}"),
        ("bullet_pull_force_kgf",   "<b>İrtibat Kuvveti:</b> {}"),
        ("movement_time_ms",        "<b>Harekete Geçme:</b> {}"),
        ("armor_penetration",       "<b>Zırh Delme:</b> {}"),
        ("accuracy",                "<b>Hassasiyet:</b> {}"),
        ("case_model",              "<b>Kovan Model:</b> {}"),
        ("case_material",           "<b>Kovan Malzeme:</b> {}"),
        ("projectile_body_material","<b>Mermi Gövde:</b> {}"),
        ("fuze",                    "<b>Tapa:</b> {}"),
        ("nose_type",               "<b>Mermi Tapası:</b> {}"),
        ("primer",                  "<b>Kapsül:</b> {}"),
        ("link_type",               "<b>Mayon:</b> {}"),
        ("propellant",              "<b>Barut:</b> {}"),
        ("weapon",                  "<b>Kullanılan Silah:</b> {}"),
    ]
    _info_lines.append("")
    for _field, _tmpl in _OPTIONAL_FIELDS:
        _v = _ai.get(_field)
        if _v is not None:
            _info_lines.append(_tmpl.format(_v))

    st.markdown(
        '<div class="info-box">' + "<br>".join(_info_lines) + "</div>",
        unsafe_allow_html=True,
    )

    # ── Mühimmat (düzenlenebilir parametreler) ────────────────────────────────
    st.markdown('<div class="section-header">Mühimmat</div>', unsafe_allow_html=True)
    bullet_name = st.text_input("Mermi Adı",         key="ammo_input_name")
    bullet_mass = st.number_input("Ağırlık (g)",      min_value=1.0,  step=0.1,   key="ammo_input_mass")
    bullet_diam = st.number_input("Çap (mm)",         min_value=1.0,  step=0.1,   key="ammo_input_diam")
    bullet_len  = st.number_input("Uzunluk (mm)",     min_value=0.1,  step=0.1,   key="ammo_input_len",
                                   help="Kayıt amaçlı — fizik hesabında kullanılmaz.")
    muzzle_vel  = st.number_input("Namlu Hızı (m/s)", min_value=1.0,  step=1.0,   key="ammo_input_vel")
    cd_val      = st.number_input("Cd (sabit)",       min_value=0.01, step=0.01,  key="ammo_input_cd",
                                   format="%.3f",
                                   help="Mach Dependent Cd kapalıyken bu değer kullanılır.")
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

    temperature = st.number_input("Sıcaklık (°C)", value=15.0,    min_value=-60.0, max_value=60.0, step=0.5)
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

    target_radius = st.number_input("Hedef yarıçapı (m)",            value=1.0,  min_value=0.1,  step=0.1,   format="%.2f")
    burst_size    = st.number_input("Burst mermi sayısı",             value=25,   min_value=1,    step=1)
    disp_mrad     = st.number_input("Dispersion sigma (mrad)",        value=1.5,  min_value=0.01, step=0.1,   format="%.2f")
    track_err     = st.number_input("Tracking/Radar error sigma (m)", value=0.5,  min_value=0.0,  step=0.1,   format="%.2f")
    lead_err_coef = st.number_input("Lead error coefficient",         value=0.02, min_value=0.0,  step=0.005, format="%.3f")
    num_trials    = st.number_input("Monte Carlo deneme sayısı",      value=1000, min_value=10,   max_value=50000, step=100)
    pk_seed       = st.number_input("Random Seed",                    value=42,   min_value=0,    step=1)

    st.markdown("")
    calc_btn = st.button("⚡ Hesapla")


# ═══════════════════════════════════════════════════════════════════════════════
#  HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════════

if calc_btn:

    # Anlık mühimmat snapshot'ı — kullanıcının o anki (düzenlenmiş) değerlerini saklar
    _ammo_snap = {
        "name":      selected_ammo,
        "display":   AMMO_DB[selected_ammo]["display"],
        "type":      AMMO_DB[selected_ammo]["type"],
        "mass_g":    float(bullet_mass),
        "diam_mm":   float(bullet_diam),
        "length_mm": float(bullet_len),
        "muzzle_ms": float(muzzle_vel),
    }

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
        st.session_state["df"]            = df_result
        st.session_state["mode"]          = "Tek Tilt"
        st.session_state["mach_info"]     = {
            "initial_mach": round(initial_mach,        3),
            "impact_mach":  round(_sim["impact_mach"],  3),
            "avg_cd":       round(_sim["avg_cd"],        4),
        }
        st.session_state["ammo_snapshot"] = _ammo_snap
        st.session_state["sim_params"] = dict(
            distance_m=float(distance), tilt_deg=float(tilt_angle),
            temp_c=float(temperature), pressure_hpa=float(pressure),
            wind_x=float(wind_x), wind_z=float(wind_z),
            bullet_mass_g=float(bullet_mass), bullet_diam_mm=float(bullet_diam),
            muzzle_vel_ms=float(muzzle_vel), cd_base=float(cd_val),
            use_mach_table=bool(use_mach_table),
        )
        st.session_state["sim3d_fig_single"] = None
        st.session_state["sim3d_fig_range"]  = None

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
            st.session_state["df"]            = df_result
            st.session_state["mode"]          = "Tilt Aralığı"
            st.session_state["mach_info"]     = None
            st.session_state["ammo_snapshot"] = _ammo_snap
            st.session_state["sim_params"] = dict(
                distance_m=float(distance), tilt_deg=None,
                temp_c=float(temperature), pressure_hpa=float(pressure),
                wind_x=float(wind_x), wind_z=float(wind_z),
                bullet_mass_g=float(bullet_mass), bullet_diam_mm=float(bullet_diam),
                muzzle_vel_ms=float(muzzle_vel), cd_base=float(cd_val),
                use_mach_table=bool(use_mach_table),
            )
            st.session_state["sim3d_fig_single"] = None
            st.session_state["sim3d_fig_range"]  = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SONUÇLAR
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state["df"] is None:
    st.info("Sol panelden parametreleri girin ve **⚡ Hesapla** butonuna basın.")
    st.stop()

df        = st.session_state["df"]
mode      = st.session_state["mode"]
ammo_snap = st.session_state.get("ammo_snapshot") or {}

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


def _ammo_metric_cards():
    """Senaryo üstüne mühimmat bilgi kartları ekler (her iki modda da kullanılır)."""
    if not ammo_snap:
        return
    am1, am2, am3 = st.columns(3)
    am4, am5, _   = st.columns(3)
    with am1: metric_card("Mühimmat",       ammo_snap.get("name", "—"),                  "")
    with am2: metric_card("Mühimmat Tipi",  ammo_snap.get("type", "—"),                  "")
    with am3: metric_card("Mermi Uzunluğu", f"{ammo_snap.get('length_mm', 0):.1f}",      "mm")
    with am4: metric_card("Mermi Ağırlığı", f"{ammo_snap.get('mass_g', 0):.1f}",         "g")
    with am5: metric_card("Namlu Hızı",     f"{ammo_snap.get('muzzle_ms', 0):.2f}",      "m/s")


def _build_csv(base_df: pd.DataFrame) -> bytes:
    """CSV'ye mühimmat metadata kolonlarını ekleyerek döndürür."""
    export = base_df.copy()
    export.insert(len(export.columns), "Ammo Name",            ammo_snap.get("name", ""))
    export.insert(len(export.columns), "Ammo Type",            ammo_snap.get("type", ""))
    export.insert(len(export.columns), "Bullet Mass (g)",      ammo_snap.get("mass_g", ""))
    export.insert(len(export.columns), "Bullet Diameter (mm)", ammo_snap.get("diam_mm", ""))
    export.insert(len(export.columns), "Bullet Length (mm)",   ammo_snap.get("length_mm", ""))
    export.insert(len(export.columns), "Muzzle Velocity (m/s)", ammo_snap.get("muzzle_ms", ""))
    return export.to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  TEK TILT MODE
# ─────────────────────────────────────────────────────────────────────────────

if mode == "Tek Tilt":

    s        = df.iloc[0]
    lead_100 = float(df.loc[df["Hedef Hızı (km/h)"] == 100, "Lead (m)"].iloc[0])
    mi       = st.session_state.get("mach_info") or {}

    # ── Mühimmat kartları ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mühimmat</div>', unsafe_allow_html=True)
    _ammo_metric_cards()

    st.divider()

    # ── Senaryo sonuç kartları ────────────────────────────────────────────────
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

    st.download_button("⬇ sonuc.csv İndir", _build_csv(df), "sonuc.csv", "text/csv")

    # ── 3D Atış Simülasyonu ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">3D Atış Simülasyonu</div>',
                unsafe_allow_html=True)

    _sp = st.session_state.get("sim_params")
    if _sp:
        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            _sim3d_speed_s = st.number_input(
                "Hedef Hızı (km/h)", value=100, min_value=1, max_value=500, step=1,
                key="sim3d_speed_single",
            )
        with _sc2:
            _use_lead_s    = st.checkbox("Lead uygulanmış atış",       value=True,  key="sim3d_lead_s")
        with _sc3:
            _use_req_elev_s = st.checkbox("Required elevation ile göster", value=False, key="sim3d_req_s")

        _scale_lbl_s = st.radio(
            "Ölçek Modu", ["Görsel Ölçek", "Gerçek Ölçek"],
            index=0, horizontal=True, key="sim3d_scale_s",
        )
        _scale_mode_s = "visual" if _scale_lbl_s == "Görsel Ölçek" else "real"

        if st.button("🎯 3D Animasyon Oluştur", key="sim3d_btn_single"):
            with st.spinner("3D simülasyon hesaplanıyor…"):
                _row0    = df.iloc[0]
                _req_e   = float(_row0["Required Elevation (°)"])
                _drp     = float(_row0["Drop (m)"])
                _wdr     = float(_row0["Wind Drift (m)"])
                _ent_tlt = _sp["tilt_deg"]
                _sim_tlt = _req_e if _use_req_elev_s else _ent_tlt
                _tgt_ms  = _sim3d_speed_s / 3.6

                # Pass 1 — straight shot (aim_z=0) to get TOF → compute lead
                _traj0 = simulate_trajectory_points(
                    _sp["distance_m"], _sim_tlt,
                    _sp["temp_c"], _sp["pressure_hpa"],
                    _sp["wind_x"], _sp["wind_z"],
                    _sp["bullet_mass_g"], _sp["bullet_diam_mm"],
                    _sp["muzzle_vel_ms"], _sp["cd_base"],
                    use_mach_table=_sp["use_mach_table"],
                    aim_z_m=0.0,
                )
                _lead_s = _tgt_ms * _traj0["tof"]

                # Pass 2 — with yaw toward lead if checkbox enabled
                _aim_z = _lead_s if _use_lead_s else 0.0
                _traj  = simulate_trajectory_points(
                    _sp["distance_m"], _sim_tlt,
                    _sp["temp_c"], _sp["pressure_hpa"],
                    _sp["wind_x"], _sp["wind_z"],
                    _sp["bullet_mass_g"], _sp["bullet_diam_mm"],
                    _sp["muzzle_vel_ms"], _sp["cd_base"],
                    use_mach_table=_sp["use_mach_table"],
                    aim_z_m=_aim_z,
                )
                _fig3d = make_3d_animation(
                    traj=_traj,
                    tof=_traj["tof"],
                    impact_y=_traj["impact_y"],
                    impact_z=_traj["impact_z"],
                    lead_m=_lead_s,
                    distance_m=_sp["distance_m"],
                    target_speed_ms=_tgt_ms,
                    wind_x=_sp["wind_x"],
                    wind_z=_sp["wind_z"],
                    required_elev=_req_e,
                    drop=_drp,
                    wind_drift=_wdr,
                    ammo_name=ammo_snap.get("name", "—"),
                    speed_kmh=_sim3d_speed_s,
                    tilt_deg=_sim_tlt,
                    entered_tilt_deg=_ent_tlt,
                    scale_mode=_scale_mode_s,
                )
                st.session_state["sim3d_fig_single"] = _fig3d

        _fig3d_s = st.session_state.get("sim3d_fig_single")
        if _fig3d_s is not None:
            st.plotly_chart(_fig3d_s, use_container_width=True)
    else:
        st.info("Önce ⚡ Hesapla butonuna basın.")


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

    # ── Mühimmat kartları ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mühimmat</div>', unsafe_allow_html=True)
    _ammo_metric_cards()

    st.divider()

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

    st.download_button("⬇ sonuc.csv İndir", _build_csv(df), "sonuc.csv", "text/csv")

    # ── 3D Atış Simülasyonu ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">3D Atış Simülasyonu</div>',
                unsafe_allow_html=True)

    _sp = st.session_state.get("sim_params")
    if _sp:
        _unique_tilts = sorted(df["Tilt (°)"].unique().tolist())
        _tilt_labels  = [f"{t:.4f}°" for t in _unique_tilts]

        _rc1, _rc2 = st.columns(2)
        with _rc1:
            _sel_tilt_lbl = st.selectbox("Tilt Seçin", _tilt_labels, key="sim3d_tilt_range")
        with _rc2:
            _sim3d_speed_r = st.number_input(
                "Hedef Hızı (km/h)", value=100, min_value=1, max_value=500, step=1,
                key="sim3d_speed_range",
            )
        _sel_tilt_val = _unique_tilts[_tilt_labels.index(_sel_tilt_lbl)]

        _rc3, _rc4 = st.columns(2)
        with _rc3:
            _use_lead_r     = st.checkbox("Lead uygulanmış atış",           value=True,  key="sim3d_lead_r")
        with _rc4:
            _use_req_elev_r = st.checkbox("Required elevation ile göster",  value=False, key="sim3d_req_r")

        _scale_lbl_r = st.radio(
            "Ölçek Modu", ["Görsel Ölçek", "Gerçek Ölçek"],
            index=0, horizontal=True, key="sim3d_scale_r",
        )
        _scale_mode_r = "visual" if _scale_lbl_r == "Görsel Ölçek" else "real"

        if st.button("🎯 3D Animasyon Oluştur", key="sim3d_btn_range"):
            with st.spinner("3D simülasyon hesaplanıyor…"):
                _trows = df[
                    (df["Tilt (°)"] == _sel_tilt_val) &
                    (df["Hedef Hızı (km/h)"] == _sim3d_speed_r)
                ]
                if not _trows.empty:
                    _req_e = float(_trows.iloc[0]["Required Elevation (°)"])
                    _drp   = float(_trows.iloc[0]["Drop (m)"])
                    _wdr   = float(_trows.iloc[0]["Wind Drift (m)"])
                else:
                    _req_e = 0.0; _drp = 0.0; _wdr = 0.0

                _ent_tlt = _sel_tilt_val
                _sim_tlt = _req_e if _use_req_elev_r else _ent_tlt
                _tgt_ms  = _sim3d_speed_r / 3.6

                # Pass 1 — straight shot to get TOF → compute lead
                _traj0 = simulate_trajectory_points(
                    _sp["distance_m"], _sim_tlt,
                    _sp["temp_c"], _sp["pressure_hpa"],
                    _sp["wind_x"], _sp["wind_z"],
                    _sp["bullet_mass_g"], _sp["bullet_diam_mm"],
                    _sp["muzzle_vel_ms"], _sp["cd_base"],
                    use_mach_table=_sp["use_mach_table"],
                    aim_z_m=0.0,
                )
                _lead_r = _tgt_ms * _traj0["tof"]

                # Pass 2 — with yaw if lead checkbox enabled
                _aim_z = _lead_r if _use_lead_r else 0.0
                _traj  = simulate_trajectory_points(
                    _sp["distance_m"], _sim_tlt,
                    _sp["temp_c"], _sp["pressure_hpa"],
                    _sp["wind_x"], _sp["wind_z"],
                    _sp["bullet_mass_g"], _sp["bullet_diam_mm"],
                    _sp["muzzle_vel_ms"], _sp["cd_base"],
                    use_mach_table=_sp["use_mach_table"],
                    aim_z_m=_aim_z,
                )
                _fig3d = make_3d_animation(
                    traj=_traj,
                    tof=_traj["tof"],
                    impact_y=_traj["impact_y"],
                    impact_z=_traj["impact_z"],
                    lead_m=_lead_r,
                    distance_m=_sp["distance_m"],
                    target_speed_ms=_tgt_ms,
                    wind_x=_sp["wind_x"],
                    wind_z=_sp["wind_z"],
                    required_elev=_req_e,
                    drop=_drp,
                    wind_drift=_wdr,
                    ammo_name=ammo_snap.get("name", "—"),
                    speed_kmh=_sim3d_speed_r,
                    tilt_deg=_sim_tlt,
                    entered_tilt_deg=_ent_tlt,
                    scale_mode=_scale_mode_r,
                )
                st.session_state["sim3d_fig_range"] = _fig3d

        _fig3d_r = st.session_state.get("sim3d_fig_range")
        if _fig3d_r is not None:
            st.plotly_chart(_fig3d_r, use_container_width=True)
    else:
        st.info("Önce ⚡ Hesapla butonuna basın.")
