import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io

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
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


def compute_ballistics(distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
                       bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd,
                       target_speed_kmh):
    g = 9.81
    R = 287.05
    T = temp_c + 273.15
    P = pressure_hpa * 100.0

    rho = P / (R * T)

    bullet_mass = bullet_mass_g / 1000.0
    bullet_diam = bullet_diam_mm / 1000.0
    A = np.pi * (bullet_diam / 2) ** 2

    k = 0.5 * cd * rho * A / bullet_mass

    v0x = muzzle_vel_ms * np.cos(np.radians(tilt_deg))
    v0y = muzzle_vel_ms * np.sin(np.radians(tilt_deg))

    dt = 0.0005
    x, y, vx, vy = 0.0, 0.0, v0x, v0y
    t = 0.0
    tof = None

    for _ in range(2_000_000):
        v = np.sqrt(vx**2 + vy**2)
        ax = -k * v * vx - wind_ms * k * abs(vx)
        ay = -g - k * v * vy

        vx += ax * dt
        vy += ay * dt
        x  += vx * dt
        y  += vy * dt
        t  += dt

        if x >= distance_m:
            tof = t
            break
        if y < -500:
            tof = t
            break

    if tof is None:
        tof = distance_m / v0x if v0x > 0 else 0.0

    drop = 0.5 * g * tof ** 2

    target_speed_ms = target_speed_kmh / 3.6
    lead = target_speed_ms * tof

    required_elev_rad = np.arctan(drop / distance_m) if distance_m > 0 else 0.0
    required_elev_deg = np.degrees(required_elev_rad)

    vertical_error = tilt_deg - required_elev_deg

    return tof, lead, drop, required_elev_deg, vertical_error


def sweep(distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
          bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd):
    speeds = np.arange(1, 501, 1)
    rows = []
    for spd in speeds:
        tof, lead, drop, req_elev, vert_err = compute_ballistics(
            distance_m, tilt_deg, temp_c, pressure_hpa, wind_ms,
            bullet_mass_g, bullet_diam_mm, muzzle_vel_ms, cd, spd
        )
        rows.append({
            "Hedef Hızı (km/h)": spd,
            "TOF (s)": round(tof, 4),
            "Lead (m)": round(lead, 3),
            "Mermi Düşüşü (m)": round(drop, 3),
            "Gerekli Elevation (°)": round(req_elev, 4),
            "Girilen Tilt (°)": round(tilt_deg, 4),
            "Dikey Hata (°)": round(vert_err, 4),
        })
    return pd.DataFrame(rows)


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


# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Ballistic Lead Calculator")
st.markdown('<div class="subtitle">Target Intercept &amp; Elevation Analysis</div>', unsafe_allow_html=True)
st.divider()

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Mühimmat Parametreleri</div>', unsafe_allow_html=True)
    bullet_name   = st.text_input("Mermi Adı", value="Default 25mm")
    bullet_mass   = st.number_input("Ağırlık (g)", value=185.0, min_value=1.0, step=0.1)
    bullet_diam   = st.number_input("Çap (mm)",    value=25.0,  min_value=1.0, step=0.1)
    muzzle_vel    = st.number_input("Namlu Çıkış Hızı (m/s)", value=1000.0, min_value=1.0, step=1.0)
    cd            = st.number_input("Cd (Sürüklenme Katsayısı)", value=0.29, min_value=0.01, step=0.01, format="%.3f")

    st.markdown('<div class="section-header">Ortam Koşulları</div>', unsafe_allow_html=True)
    distance   = st.number_input("Mesafe (m)",           value=1000.0, min_value=10.0,   step=10.0)
    tilt_angle = st.number_input("Tilt / Elevation (°)", value=2.0,    min_value=-45.0,  max_value=45.0, step=0.1)
    temperature= st.number_input("Hava Sıcaklığı (°C)",  value=15.0,  min_value=-60.0,  max_value=60.0, step=0.5)
    pressure   = st.number_input("Basınç (hPa)",          value=1013.25, min_value=800.0, step=0.5)
    wind_speed = st.number_input("Rüzgar Hızı (m/s)",    value=0.0,   min_value=0.0,    step=0.1)

    st.markdown("")
    calc_btn = st.button("⚡ Hesapla")

# ── MAIN ────────────────────────────────────────────────────────────────────
if calc_btn:
    with st.spinner("Hesaplanıyor…"):
        df = sweep(distance, tilt_angle, temperature, pressure, wind_speed,
                   bullet_mass, bullet_diam, muzzle_vel, cd)

    ref_row = df[df["Hedef Hızı (km/h)"] == 100].iloc[0]

    st.markdown('<div class="section-header">Referans Sonuçlar (100 km/h)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("TOF", f"{ref_row['TOF (s)']:.4f}", "s")
    with c2: metric_card("Lead", f"{ref_row['Lead (m)']:.3f}", "m")
    with c3: metric_card("Mermi Düşüşü", f"{ref_row['Mermi Düşüşü (m)']:.3f}", "m")
    with c4: metric_card("Dikey Hata", f"{ref_row['Dikey Hata (°)']:.4f}", "°")

    st.divider()

    # ── CHARTS ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Grafikler</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    g3, g4 = st.columns(2)

    with g1:
        st.markdown("**Lead vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Lead (m)", "#58a6ff", "Lead (m)"))

    with g2:
        st.markdown("**Gerekli Elevation vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Gerekli Elevation (°)", "#f78166", "Elevation (°)"))

    with g3:
        st.markdown("**Mermi Düşüşü vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Mermi Düşüşü (m)", "#3fb950", "Düşüş (m)"))

    with g4:
        st.markdown("**Dikey Hata vs Hedef Hızı**")
        st.pyplot(make_figure(df, "Hedef Hızı (km/h)", "Dikey Hata (°)", "#d2a8ff", "Dikey Hata (°)"))

    st.divider()

    # ── TABLE ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sonuç Tablosu</div>', unsafe_allow_html=True)

    search = st.text_input("Tabloda ara (hız filtrele):", placeholder="örn. 100")
    show_df = df.copy()
    if search.strip():
        try:
            val = float(search.strip())
            show_df = show_df[show_df["Hedef Hızı (km/h)"] == val]
        except ValueError:
            pass

    st.dataframe(
        show_df.style.format({
            "TOF (s)": "{:.4f}",
            "Lead (m)": "{:.3f}",
            "Mermi Düşüşü (m)": "{:.3f}",
            "Gerekli Elevation (°)": "{:.4f}",
            "Girilen Tilt (°)": "{:.4f}",
            "Dikey Hata (°)": "{:.4f}",
        }),
        use_container_width=True,
        height=400,
    )

    # ── CSV DOWNLOAD ──────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ sonuc.csv İndir",
        data=csv_bytes,
        file_name="sonuc.csv",
        mime="text/csv",
    )

else:
    st.info("Sol panelden parametreleri girin ve **⚡ Hesapla** butonuna basın.")
