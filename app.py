"""
Ballistic Lead Calculator
3D point-mass model · RK4 integration · relative-air-velocity drag
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

st.set_page_config(
    page_title="Ballistic Lead Calculator",
    page_icon="🎯",
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ATMOSPHERE
# ═══════════════════════════════════════════════════════════════════════════════

def atmosphere(temp_c: float, pressure_hpa: float):
    """
    ISA-style local atmosphere.

    Returns
    -------
    rho : float   air density  [kg/m³]
    c   : float   speed of sound [m/s]
    """
    T   = temp_c + 273.15       # K
    P   = pressure_hpa * 100.0  # Pa
    rho = P / (R_AIR * T)
    c   = np.sqrt(GAMMA_AIR * R_AIR * T)
    return rho, c


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAG COEFFICIENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_cd(mach: float, cd_base: float) -> float:
    """
    Mach-dependent drag coefficient.

    Currently returns the constant cd_base.
    To add a Cd(Mach) table, replace this function body with an interpolation
    over a user-supplied (mach_array, cd_array) pair.

    Parameters
    ----------
    mach    : local Mach number of the bullet relative to air
    cd_base : baseline Cd from UI input
    """
    return cd_base


# ═══════════════════════════════════════════════════════════════════════════════
#  RK4 CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _derivs(state: np.ndarray, wind_vec: np.ndarray,
            rho: float, c: float,
            cd_base: float, area: float, mass: float) -> np.ndarray:
    """
    Time derivatives of the 6-DOF point-mass state vector.

    State layout : [x, y, z, vx, vy, vz]

    Coordinate system:
        x  = downrange  (forward, in direction of fire)
        y  = vertical   (up positive)
        z  = lateral    (right positive when viewed from behind the gun)

    Physics:
        v_rel  = bullet velocity − wind vector   (bullet velocity relative to air)
        a_drag = −(k · |v_rel|) · v_rel          where k = ½ρCdA/m
        a_grav = [0, −g, 0]
        a      = a_drag + a_grav
    """
    vel   = state[3:6]
    v_rel = vel - wind_vec
    spd   = np.linalg.norm(v_rel)

    mach = spd / c if c > 0.0 else 0.0
    cd   = get_cd(mach, cd_base)
    k    = 0.5 * rho * cd * area / mass

    a_drag = -k * spd * v_rel
    a_grav = np.array([0.0, -G, 0.0])
    a      = a_drag + a_grav

    return np.concatenate([vel, a])


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
) -> dict:
    """
    3D point-mass ballistic trajectory with RK4 integration.

    Wind convention
    ---------------
    wind_x > 0  tailwind  (same direction as bullet → reduces effective drag)
    wind_x < 0  headwind  (opposing bullet → increases effective drag)
    wind_z > 0  rightward crosswind  (pushes bullet in +z direction)
    wind_z < 0  leftward  crosswind

    Impact detection
    ----------------
    When x crosses distance_m, the exact impact point is found by linear
    interpolation between the last two RK4 states.

    Returns
    -------
    dict with keys: tof, impact_y, impact_z, impact_velocity, impact_energy
    """
    rho, c = atmosphere(temp_c, pressure_hpa)

    mass = bullet_mass_g / 1000.0
    area = np.pi * (bullet_diam_mm / 2000.0) ** 2   # (d/2 in m)² · π

    wind_vec = np.array([wind_x, 0.0, wind_z])
    kw = dict(wind_vec=wind_vec, rho=rho, c=c,
              cd_base=cd_base, area=area, mass=mass)

    v0x = muzzle_vel_ms * np.cos(np.radians(tilt_deg))
    v0y = muzzle_vel_ms * np.sin(np.radians(tilt_deg))
    state = np.array([0.0, 0.0, 0.0, v0x, v0y, 0.0])

    t          = 0.0
    prev_state = state.copy()
    prev_t     = 0.0

    for _ in range(500_000):
        prev_state = state.copy()
        prev_t     = t

        state = _rk4_step(state, dt, **kw)
        t    += dt

        if state[0] >= distance_m:
            # Linear interpolation to find precise impact point
            dx   = state[0] - prev_state[0]
            frac = (distance_m - prev_state[0]) / dx if dx > 1e-12 else 1.0
            imp  = prev_state + frac * (state - prev_state)
            tof  = prev_t + frac * dt

            v_imp = imp[3:6]
            spd   = float(np.linalg.norm(v_imp))
            return {
                "tof":             tof,
                "impact_y":        float(imp[1]),
                "impact_z":        float(imp[2]),
                "impact_velocity": spd,
                "impact_energy":   0.5 * mass * spd**2,
            }

        if state[1] < -(distance_m + 200.0):
            break

    # Fallback: bullet did not reach target range
    spd = float(np.linalg.norm(state[3:6]))
    return {
        "tof":             t,
        "impact_y":        float(state[1]),
        "impact_z":        float(state[2]),
        "impact_velocity": spd,
        "impact_energy":   0.5 * mass * spd**2,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO  (two simulations: full conditions + horizontal-no-wind reference)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scenario(
        distance_m, tilt_deg, temp_c, pressure_hpa,
        wind_x, wind_z,
        bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base,
) -> dict:
    """
    Derive all ballistic outputs from two trajectory runs.

    Sim 1 — given tilt + wind      → TOF, impact_y, impact_z, v_impact
    Sim 2 — tilt=0°, no wind       → required-elevation reference drop
    """
    main = simulate_3d(distance_m, tilt_deg, temp_c, pressure_hpa,
                       wind_x, wind_z,
                       bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base)

    ref  = simulate_3d(distance_m, 0.0, temp_c, pressure_hpa,
                       0.0, 0.0,
                       bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base)

    tof   = main["tof"]
    imp_y = main["impact_y"]
    imp_z = main["impact_z"]        # ← this IS the wind drift (z-axis lateral)
    v_imp = main["impact_velocity"]
    e_imp = main["impact_energy"]

    # Drop: vertical distance below the line-of-departure at target range
    #   line-of-departure height at range d = d · tan(tilt)
    drop = distance_m * np.tan(np.radians(tilt_deg)) - imp_y

    # Wind Drift: lateral (z-axis) displacement at impact
    wind_drift = imp_z

    # Required Elevation: angle to compensate pure-gravity drop (horizontal, no wind)
    drop_horiz    = max(-ref["impact_y"], 0.0)
    required_elev = np.degrees(np.arctan2(drop_horiz, distance_m))

    # Vertical error in degrees and metres
    vert_error_deg = tilt_deg - required_elev
    vert_error_m   = imp_y       # bullet's actual y at target (+ = above, − = below)

    return dict(
        tof=tof,
        impact_y=imp_y,
        impact_z=imp_z,
        drop=drop,
        wind_drift=wind_drift,
        required_elev=required_elev,
        vert_error_deg=vert_error_deg,
        vert_error_m=vert_error_m,
        impact_velocity=v_imp,
        impact_energy=e_imp,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TARGET-SPEED SWEEP  1–500 km/h
# ═══════════════════════════════════════════════════════════════════════════════

def sweep(distance_m, tilt_deg, temp_c, pressure_hpa,
          wind_x, wind_z,
          bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base) -> pd.DataFrame:
    """
    Trajectory is computed ONCE.  Only Lead = v_target × TOF varies per row,
    because the target's speed does not affect the bullet's trajectory.
    """
    s = compute_scenario(distance_m, tilt_deg, temp_c, pressure_hpa,
                         wind_x, wind_z,
                         bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd_base)

    rows = []
    for spd in range(1, 501):
        lead = (spd / 3.6) * s["tof"]
        rows.append({
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
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(df, x_col, y_col, color, ylabel):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    ax.plot(df[x_col], df[y_col], color=color, linewidth=1.8)
    ax.set_xlabel(x_col, color="#8b949e", fontsize=9)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linestyle="--", linewidth=0.6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout(pad=0.8)
    return fig


def metric_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value} <span class="metric-unit">{unit}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🎯 Ballistic Lead Calculator")
st.markdown(
    '<div class="subtitle">Target Intercept &amp; Elevation Analysis'
    ' &nbsp;·&nbsp; 3D Point-Mass · RK4 · Relative-Air-Velocity Drag</div>',
    unsafe_allow_html=True,
)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Ammo ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mühimmat</div>', unsafe_allow_html=True)
    bullet_name = st.text_input("Mermi Adı",                value="Default 25mm")
    bullet_mass = st.number_input("Ağırlık (g)",            value=185.0,   min_value=1.0,  step=0.1)
    bullet_diam = st.number_input("Çap (mm)",               value=25.0,    min_value=1.0,  step=0.1)
    muzzle_vel  = st.number_input("Namlu Hızı (m/s)",       value=1000.0,  min_value=1.0,  step=1.0)
    cd_val      = st.number_input("Cd (sabit)", value=0.29, min_value=0.01, step=0.01, format="%.3f",
                                  help="get_cd(mach) fonksiyonu ile Mach'a bağımlı tablo eklenebilir.")

    # ── Conditions ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Koşullar</div>', unsafe_allow_html=True)
    distance    = st.number_input("Mesafe (m)",           value=1000.0,   min_value=10.0,  step=10.0)
    tilt_angle  = st.number_input("Tilt / Elevation (°)", value=2.0,
                                  min_value=-45.0, max_value=45.0, step=0.1)
    temperature = st.number_input("Sıcaklık (°C)",        value=15.0,
                                  min_value=-60.0, max_value=60.0, step=0.5)
    pressure    = st.number_input("Basınç (hPa)",         value=1013.25,  min_value=800.0, step=0.5)

    # ── Wind ──────────────────────────────────────────────────────────────────
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
        ["Sağdan (+z)", "Soldan (−z)"],
        index=0, key="cw_dir",
    )
    wind_z = +cw_spd if cw_dir.startswith("Sağdan") else -cw_spd

    st.markdown("")
    calc_btn = st.button("⚡ Hesapla")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if calc_btn:
    with st.spinner("Simülasyon çalışıyor (3D · RK4)…"):
        df = sweep(distance, tilt_angle, temperature, pressure,
                   wind_x, wind_z,
                   bullet_mass, bullet_diam, muzzle_vel, cd_val)

    s = df.iloc[0]   # all trajectory cols are constant; lead from row 100
    lead_100 = float(df.loc[df["Hedef Hızı (km/h)"] == 100, "Lead (m)"].iloc[0])

    # ── METRIC CARDS ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Senaryo Sonuçları</div>',
                unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns(3)
    r2a, r2b, r2c = st.columns(3)
    r3a, r3b, _   = st.columns(3)

    with r1a: metric_card("TOF",                   f"{s['TOF (s)']:.4f}",                     "s")
    with r1b: metric_card("Lead @ 100 km/h",        f"{lead_100:.3f}",                          "m")
    with r1c: metric_card("Drop",                   f"{s['Drop (m)']:.3f}",                     "m")
    with r2a: metric_card("Wind Drift (z-axis)",    f"{s['Wind Drift (m)']:.3f}",               "m")
    with r2b: metric_card("Required Elevation",     f"{s['Required Elevation (°)']:.4f}",       "°")
    with r2c: metric_card("Vertical Error",         f"{s['Vertical Error (m)']:.3f}",           "m")
    with r3a: metric_card("Impact Velocity",        f"{s['Impact Velocity (m/s)']:.2f}",        "m/s")
    with r3b: metric_card("Impact Energy",          f"{s['Impact Energy (J)']:.1f}",            "J")

    st.divider()

    # ── CHARTS ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Grafikler</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'TOF, Drop, Wind Drift, Required Elevation ve Impact Velocity hedef hızından bağımsızdır '
        '(yalnızca mesafe, mühimmat ve ortam koşullarına bağlıdır). '
        'Sadece <b>Lead = v_hedef × TOF</b> doğrusal olarak değişir.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, _    = st.columns(2)

    with col1:
        st.markdown("**Lead vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Lead (m)", "#58a6ff", "Lead (m)"))

    with col2:
        st.markdown("**Required Elevation vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Required Elevation (°)", "#f78166", "Elevation (°)"))

    with col3:
        st.markdown("**Drop vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Drop (m)", "#3fb950", "Drop (m)"))

    with col4:
        st.markdown("**Wind Drift vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Wind Drift (m)", "#d2a8ff", "Wind Drift (m)"))

    with col5:
        st.markdown("**Impact Velocity vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Impact Velocity (m/s)", "#ffa657", "Vel (m/s)"))

    st.divider()

    # ── TABLE ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sonuç Tablosu (1–500 km/h)</div>',
                unsafe_allow_html=True)

    search = st.text_input("Hız filtrele (km/h):", placeholder="örn. 100")
    show_df = df.copy()
    if search.strip():
        try:
            val     = float(search.strip())
            show_df = show_df[show_df["Hedef Hızı (km/h)"] == val]
        except ValueError:
            pass

    fmt = {
        "TOF (s)":                "{:.4f}",
        "Lead (m)":               "{:.3f}",
        "Drop (m)":               "{:.3f}",
        "Wind Drift (m)":         "{:.3f}",
        "Impact Y (m)":           "{:.3f}",
        "Vertical Error (m)":     "{:.3f}",
        "Required Elevation (°)": "{:.4f}",
        "Girilen Tilt (°)":       "{:.4f}",
        "Dikey Hata (°)":         "{:.4f}",
        "Impact Velocity (m/s)":  "{:.2f}",
        "Impact Energy (J)":      "{:.1f}",
    }
    st.dataframe(show_df.style.format(fmt), use_container_width=True, height=430)

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ sonuc.csv İndir",
        data=csv_bytes,
        file_name="sonuc.csv",
        mime="text/csv",
    )

else:
    st.info("Sol panelden parametreleri girin ve **⚡ Hesapla** butonuna basın.")
