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
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --text: #e6edf3;
    --subtext: #8b949e;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.5rem; }
h1 { color: var(--accent) !important; letter-spacing: 0.04em; }
h2, h3 { color: var(--text) !important; }
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.metric-label { color: var(--subtext); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { color: var(--accent); font-size: 1.5rem; font-weight: 700; }
.metric-unit  { color: var(--subtext); font-size: 0.85rem; }
.section-header {
    border-left: 3px solid var(--accent);
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.6rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.4rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #238636 0%, #3fb950 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    width: 100%;
}
label, .stSlider label { color: var(--subtext) !important; font-size: 0.82rem !important; }
input, .stNumberInput input {
    background-color: #21262d !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.subtitle {
    color: var(--subtext);
    font-size: 0.95rem;
    margin-top: -0.8rem;
    margin-bottom: 1.2rem;
    letter-spacing: 0.03em;
}
.info-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #8b949e;
    margin-bottom: 0.8rem;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

G       = 9.81    # m/s²
R_AIR   = 287.05  # J/(kg·K)


def simulate(distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
             bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd):
    """
    2D Euler integration (x = downrange, y = vertical up).

    Drag is computed on the bullet's velocity RELATIVE TO AIR:
        vx_rel = vx - wind_ms   (wind_ms > 0 → tailwind, < 0 → headwind)
        vy_rel = vy
        v_rel  = sqrt(vx_rel² + vy_rel²)
        ax     = -k · v_rel · vx_rel
        ay     = -g - k · v_rel · vy_rel

    Returns (tof, y_at_target_range).
    """
    rho = (pressure_hpa * 100.0) / (R_AIR * (temp_c + 273.15))
    m   = bullet_mass_g / 1000.0
    A   = np.pi * (bullet_diam_mm / 2000.0) ** 2   # (d/2 in metres)² · π
    k   = 0.5 * cd * rho * A / m

    v0x = muzzle_vel_ms * np.cos(np.radians(tilt_deg))
    v0y = muzzle_vel_ms * np.sin(np.radians(tilt_deg))

    dt  = 0.0005
    x = y = 0.0
    vx, vy = v0x, v0y
    t  = 0.0
    tof = None

    for _ in range(2_000_000):
        vx_r = vx - wind_ms   # relative velocity to air (x-component)
        vy_r = vy
        v_r  = np.sqrt(vx_r * vx_r + vy_r * vy_r)

        vx += (-k * v_r * vx_r) * dt
        vy += (-G - k * v_r * vy_r) * dt
        x  += vx * dt
        y  += vy * dt
        t  += dt

        if x >= distance_m:
            tof = t
            break
        if y < -(distance_m + 200.0):  # fell far below, abort
            tof = t
            break

    if tof is None:
        tof = distance_m / max(v0x, 1e-9)

    return tof, y


def compute_scenario(distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
                     bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd):
    """
    Run three simulations once and derive all output quantities.

    Sim 1 — given conditions (tilt + wind)   → tof, y_w
    Sim 2 — same tilt, zero wind             → y_0   (wind-drift reference)
    Sim 3 — tilt = 0, zero wind              → y_h   (required-elevation reference)
    """
    sim_kw = dict(temp_c=temp_c, pressure_hpa=pressure_hpa,
                  bullet_mass_g=bullet_mass_g, bullet_diam_mm=bullet_diam_mm,
                  muzzle_vel_ms=muzzle_vel_ms, cd=cd)

    tof, y_w = simulate(distance_m, tilt_deg, **sim_kw, wind_ms=wind_ms)
    _,   y_0 = simulate(distance_m, tilt_deg, **sim_kw, wind_ms=0.0)
    _,   y_h = simulate(distance_m, 0.0,      **sim_kw, wind_ms=0.0)

    # Bullet drop: actual y deviation below the line of departure
    # line-of-departure height at range d = d·tan(tilt)
    drop = distance_m * np.tan(np.radians(tilt_deg)) - y_w

    # Wind drift: vertical displacement caused by wind
    # (positive → tailwind lifts bullet; negative → headwind adds extra drop)
    wind_drift = y_w - y_0

    # Required elevation: angle to compensate gravity-only drop for horizontal shot
    drop_horiz   = max(-y_h, 0.0)
    required_elev = np.degrees(np.arctan2(drop_horiz, distance_m))

    # Vertical error in degrees (aim angle vs required)
    vert_error_deg = tilt_deg - required_elev

    # Vertical error in metres (how far bullet hits above/below target at y = 0)
    vert_error_m = y_w

    return dict(
        tof=tof,
        y_impact=y_w,
        drop=drop,
        wind_drift=wind_drift,
        required_elev=required_elev,
        vert_error_deg=vert_error_deg,
        vert_error_m=vert_error_m,
    )


def sweep(distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
          bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd):
    """
    Target-speed sweep 1–500 km/h.
    Trajectory is computed ONCE (TOF/drop/drift don't depend on target speed).
    Only lead = v_target × TOF varies per row.
    """
    s = compute_scenario(distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
                         bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd)

    rows = []
    for spd in range(1, 501):
        lead = (spd / 3.6) * s['tof']
        rows.append({
            "Hedef Hızı (km/h)":      spd,
            "TOF (s)":                round(s['tof'], 4),
            "Lead (m)":               round(lead, 3),
            "Mermi Düşüşü (m)":      round(s['drop'], 3),
            "Wind Drift (m)":         round(s['wind_drift'], 3),
            "Impact Y (m)":           round(s['y_impact'], 3),
            "Required Elevation (°)": round(s['required_elev'], 4),
            "Girilen Tilt (°)":       round(tilt_deg, 4),
            "Dikey Hata (°)":         round(s['vert_error_deg'], 4),
            "Vertical Error (m)":     round(s['vert_error_m'], 3),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(df, x_col, y_col, color, ylabel):
    fig, ax = plt.subplots(figsize=(6, 3.2))
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


def metric_card(label, value, unit=""):
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} <span class="metric-unit">{unit}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🎯 Ballistic Lead Calculator")
st.markdown('<div class="subtitle">Target Intercept &amp; Elevation Analysis</div>',
            unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-header">Mühimmat Parametreleri</div>',
                unsafe_allow_html=True)
    bullet_name = st.text_input("Mermi Adı", value="Default 25mm")
    bullet_mass = st.number_input("Ağırlık (g)",               value=185.0,   min_value=1.0,   step=0.1)
    bullet_diam = st.number_input("Çap (mm)",                  value=25.0,    min_value=1.0,   step=0.1)
    muzzle_vel  = st.number_input("Namlu Çıkış Hızı (m/s)",   value=1000.0,  min_value=1.0,   step=1.0)
    cd_val      = st.number_input("Cd (Sürüklenme Katsayısı)", value=0.29,    min_value=0.01,  step=0.01, format="%.3f")

    st.markdown('<div class="section-header">Ortam Koşulları</div>',
                unsafe_allow_html=True)
    distance    = st.number_input("Mesafe (m)",           value=1000.0, min_value=10.0,  step=10.0)
    tilt_angle  = st.number_input("Tilt / Elevation (°)", value=2.0,    min_value=-45.0, max_value=45.0, step=0.1)
    temperature = st.number_input("Hava Sıcaklığı (°C)", value=15.0,   min_value=-60.0, max_value=60.0, step=0.5)
    pressure    = st.number_input("Basınç (hPa)",         value=1013.25,min_value=800.0, step=0.5)

    wind_spd = st.number_input("Rüzgar Hızı (m/s)", value=0.0, min_value=0.0, step=0.1)
    wind_dir = st.radio(
        "Rüzgar Yönü",
        ["Karşıdan (Headwind −)", "Arkadan (Tailwind +)"],
        index=0,
        help="Headwind mermiye karşı eser (drag artar). Tailwind merminin arkasından eser (drag azalır).",
    )
    # wind_ms > 0 → tailwind, < 0 → headwind
    wind_ms = wind_spd if wind_dir.startswith("Arkadan") else -wind_spd

    st.markdown("")
    calc_btn = st.button("⚡ Hesapla")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if calc_btn:
    with st.spinner("Simülasyon çalışıyor…"):
        df = sweep(distance, tilt_angle, temperature, pressure, wind_ms,
                   bullet_mass, bullet_diam, muzzle_vel, cd_val)

    # Scenario-level values are constant across rows; read from first row
    s = df.iloc[0]
    lead_100 = df.loc[df["Hedef Hızı (km/h)"] == 100, "Lead (m)"].iloc[0]

    # ── METRIC CARDS ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Senaryo Sonuçları</div>',
                unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    with r1c1: metric_card("TOF",                   f"{s['TOF (s)']:.4f}",                "s")
    with r1c2: metric_card("Lead @ 100 km/h",       f"{lead_100:.3f}",                    "m")
    with r1c3: metric_card("Mermi Düşüşü",          f"{s['Mermi Düşüşü (m)']:.3f}",      "m")
    with r2c1: metric_card("Wind Drift",            f"{s['Wind Drift (m)']:.3f}",         "m")
    with r2c2: metric_card("Required Elevation",    f"{s['Required Elevation (°)']:.4f}", "°")
    with r2c3: metric_card("Vertical Error",        f"{s['Vertical Error (m)']:.3f}",     "m")

    st.divider()

    # ── CHARTS ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Grafikler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">TOF, Mermi Düşüşü, Elevation ve Wind Drift hedef hızından bağımsızdır '
        '(yalnızca mesafe, mühimmat ve ortam koşullarına bağlıdır). '
        'Sadece <b>Lead</b> doğrusal olarak değişir.</div>',
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2)
    g3, g4 = st.columns(2)

    with g1:
        st.markdown("**Lead vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Lead (m)", "#58a6ff", "Lead (m)"))

    with g2:
        st.markdown("**Required Elevation vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Required Elevation (°)", "#f78166", "Elevation (°)"))

    with g3:
        st.markdown("**Mermi Düşüşü vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Mermi Düşüşü (m)", "#3fb950", "Düşüş (m)"))

    with g4:
        st.markdown("**Vertical Error vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Vertical Error (m)", "#d2a8ff", "Vert. Error (m)"))

    st.divider()

    # ── TABLE ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sonuç Tablosu</div>', unsafe_allow_html=True)

    search = st.text_input("Tabloda ara (hız filtrele):", placeholder="örn. 100")
    show_df = df.copy()
    if search.strip():
        try:
            val = float(search.strip())
            show_df = show_df[show_df["Hedef Hızı (km/h)"] == val]
        except ValueError:
            pass

    fmt = {
        "TOF (s)":                "{:.4f}",
        "Lead (m)":               "{:.3f}",
        "Mermi Düşüşü (m)":      "{:.3f}",
        "Wind Drift (m)":         "{:.3f}",
        "Impact Y (m)":           "{:.3f}",
        "Required Elevation (°)": "{:.4f}",
        "Girilen Tilt (°)":       "{:.4f}",
        "Dikey Hata (°)":         "{:.4f}",
        "Vertical Error (m)":     "{:.3f}",
    }
    st.dataframe(show_df.style.format(fmt), use_container_width=True, height=420)

    # ── CSV DOWNLOAD ────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ sonuc.csv İndir",
        data=csv_bytes,
        file_name="sonuc.csv",
        mime="text/csv",
    )

else:
    st.info("Sol panelden parametreleri girin ve **⚡ Hesapla** butonuna basın.")
