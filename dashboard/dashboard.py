import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import os

config_dir = os.path.join(os.path.dirname(__file__), ".streamlit")
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, "config.toml")
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        f.write('[theme]\nbase="light"\nprimaryColor="#3b82f6"\nbackgroundColor="#f9fafb"\nsecondaryBackgroundColor="#ffffff"\ntextColor="#111827"\nfont="sans serif"\n')

st.set_page_config(
    page_title="E-Commerce Payment Analysis",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, .stMetric,
    .stSelectbox, .stDataFrame, h1, h2, h3, h4, h5, h6, p, label,
    .stSidebar, [data-testid="stMetric"] {
        font-family: 'Poppins', sans-serif !important;
        color: #111827;
    }

    .stApp { background-color: #f9fafb; }
    [data-testid="stAppViewContainer"] { background-color: #f9fafb; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
        color: #111827;
    }

    /* ── Metric ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #111827 !important;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 4px;
        letter-spacing: 0.01em;
    }
    .section-sub {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-bottom: 20px;
    }

    /* ── Step badge ── */
    .step-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Insight / warning boxes ── */
    .insight-box {
        background: #f0f9ff;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.84rem;
        color: #1e40af;
        margin-top: 12px;
    }
    .warn-box {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.84rem;
        color: #92400e;
        margin-top: 12px;
    }
    .success-box {
        background: #f0fdf4;
        border-left: 3px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.84rem;
        color: #166534;
        margin-top: 12px;
    }

    /* ── Stat pill ── */
    .stat-pill {
        display: inline-block;
        background: #f3f4f6;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.82rem;
        color: #374151;
        margin: 4px 4px 4px 0;
    }
    .stat-pill b { color: #111827; }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 28px 0;
    }

    /* ── Nav (sidebar) ── */
    .nav-item {
        display: block;
        padding: 7px 12px;
        border-radius: 8px;
        font-size: 0.83rem;
        color: #374151;
        text-decoration: none;
        margin-bottom: 2px;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .dataframe { border-radius: 8px; overflow: hidden; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colours ───────────────────────────────────────────────────────────────────
PALETTE  = ["#3b82f6", "#93c5fd", "#bfdbfe", "#dbeafe"]
LINE_COL = "#1d4ed8"


# ── Helpers ───────────────────────────────────────────────────────────────────
def bare_fig(w=7, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax


def style_ax(ax, grid_axis="y"):
    ax.tick_params(axis="both", labelsize=8, colors="#6b7280")
    if grid_axis == "y":
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    else:
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data
def process(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # --- wrangling stats (before cleaning) ---
    n_raw        = len(df)
    n_missing    = int(df.isnull().sum().sum())
    n_dup        = int(df.duplicated().sum())
    inaccurate   = df[df["payment_value"] <= 0]
    not_def      = df[df["payment_type"] == "not_defined"]

    # outlier counts per numeric col
    outlier_stats = {}
    for col in ["payment_sequential", "payment_installments", "payment_value"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        outlier_stats[col] = {
            "count": int(mask.sum()),
            "pct":   round(mask.mean() * 100, 2),
            "upper": round(q3 + 1.5 * iqr, 2),
        }

    # --- cleaning ---
    df = df[df["payment_value"] > 0]
    df = df[df["payment_type"] != "not_defined"]
    n_clean = len(df)

    # --- segmentation ---
    def segment(row):
        if row["payment_value"] > 500:
            return "High Spender"
        elif row["payment_installments"] > 12:
            return "Long-term Installment"
        elif row["payment_installments"] <= 1:
            return "Instant Payer"
        else:
            return "Regular Installment"

    df["payment_segment"] = df.apply(segment, axis=1)

    wrangling_meta = {
        "n_raw": n_raw, "n_clean": n_clean,
        "n_missing": n_missing, "n_dup": n_dup,
        "n_inaccurate": len(inaccurate),
        "n_not_defined": len(not_def),
        "outlier_stats": outlier_stats,
        "inaccurate_df": inaccurate,
        "not_def_df": not_def,
    }
    return df, wrangling_meta


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💳 Payment Analysis")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("---")
    st.markdown("**Navigasi**")
    for label in [
        "📊 Overview",
        "🔧 Data Wrangling",
        "🔍 EDA & Visualisasi",
        "🧩 Analisis Lanjutan",
        "✅ Kesimpulan",
    ]:
        st.markdown(f"<span class='nav-item'>{label}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;color:#9ca3af;'>"
        "Dataset: <b>E-Commerce Public Dataset</b><br>"
        "Author: Yahya Ahmad<br>"
        "ID Dicoding: cdcc919d6y0337"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Gate ──────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown(
        """
        <div style="margin-top:80px;text-align:center;">
            <div style="font-size:3rem;">💳</div>
            <h2 style="font-weight:600;color:#111827;margin:12px 0 6px;">E-Commerce Payment Dashboard</h2>
            <p style="color:#6b7280;font-size:0.9rem;">
                Upload file <b>order_payments_dataset.csv</b> melalui sidebar untuk memulai.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────
df_raw = load_raw(uploaded)
df, wm  = process(df_raw)

payment_counts    = df["payment_type"].value_counts()
payment_values    = df.groupby("payment_type")["payment_value"].sum().sort_values(ascending=False)
installment_trend = df.groupby("payment_installments")["payment_value"].mean().reset_index()
segment_counts    = df["payment_segment"].value_counts().reset_index()
segment_counts.columns = ["payment_segment", "count"]


# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style="margin-bottom:28px;">
        <h1 style="font-size:1.7rem;font-weight:700;color:#111827;margin:0;">
            E-Commerce Payment Dashboard
        </h1>
        <p style="font-size:0.87rem;color:#6b7280;margin:4px 0 0;">
            Analisis metode pembayaran dan perilaku transaksi pelanggan
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Transaksi (Raw)",   f"{wm['n_raw']:,}")
k2.metric("Total Transaksi (Clean)", f"{wm['n_clean']:,}")
k3.metric("Total Nilai (BRL)",       f"{df['payment_value'].sum():,.0f}")
k4.metric("Rata-rata Nilai",         f"{df['payment_value'].mean():,.2f}")
k5.metric("Metode Pembayaran",       payment_counts.shape[0])

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA WRANGLING
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">🔧 Data Wrangling</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-sub">Gathering → Assessing → Cleaning</p>',
    unsafe_allow_html=True,
)

# ── 1.1 Gathering ─────────────────────────────────────────────────────────────
st.markdown('<span class="step-badge">Step 1 — Gathering Data</span>', unsafe_allow_html=True)

with st.expander("Lihat preview dataset mentah (5 baris pertama)", expanded=True):
    st.dataframe(df_raw.head(), use_container_width=True, hide_index=True)

col_g1, col_g2, col_g3 = st.columns(3)
col_g1.markdown(
    f"<div class='stat-pill'>🗂 Baris: <b>{wm['n_raw']:,}</b></div>",
    unsafe_allow_html=True,
)
col_g2.markdown(
    f"<div class='stat-pill'>📋 Kolom: <b>{df_raw.shape[1]}</b></div>",
    unsafe_allow_html=True,
)
col_g3.markdown(
    f"<div class='stat-pill'>🔤 Tipe data: <b>{df_raw.dtypes.nunique()} tipe</b></div>",
    unsafe_allow_html=True,
)

# dtypes table
buf = io.StringIO()
df_raw.info(buf=buf)
dtype_df = pd.DataFrame({
    "Kolom": df_raw.columns,
    "Tipe Data": df_raw.dtypes.astype(str).values,
    "Non-Null": df_raw.notnull().sum().values,
    "Null": df_raw.isnull().sum().values,
})
with st.expander("Info kolom & tipe data"):
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="insight-box">
    💡 Dataset memiliki <b>5 kolom utama</b>: <code>order_id</code>, <code>payment_sequential</code>,
    <code>payment_type</code>, <code>payment_installments</code>, dan <code>payment_value</code>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── 1.2 Assessing ─────────────────────────────────────────────────────────────
st.markdown('<span class="step-badge">Step 2 — Assessing Data</span>', unsafe_allow_html=True)

a1, a2, a3, a4 = st.columns(4)
a1.metric("Missing Values",    wm["n_missing"],    delta="✅ Tidak ada" if wm["n_missing"] == 0 else None)
a2.metric("Data Duplikat",     wm["n_dup"],        delta="✅ Tidak ada" if wm["n_dup"] == 0 else None)
a3.metric("Data Tidak Akurat", wm["n_inaccurate"], delta="⚠️ Dihapus")
a4.metric("Not Defined",       wm["n_not_defined"],delta="⚠️ Dihapus")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Outlier summary
st.markdown("**Deteksi Outlier (IQR Method)**")
ols = wm["outlier_stats"]
ol_cols = st.columns(3)
for i, (col, stats) in enumerate(ols.items()):
    with ol_cols[i]:
        st.markdown(
            f"""
            <div class="card" style="padding:14px 18px;">
                <div style="font-size:0.75rem;font-weight:600;color:#6b7280;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:6px;">{col}</div>
                <div style="font-size:1.3rem;font-weight:700;color:#111827;">{stats['pct']}%</div>
                <div style="font-size:0.78rem;color:#9ca3af;">{stats['count']:,} baris outlier</div>
                <div style="font-size:0.78rem;color:#9ca3af;">Upper fence: {stats['upper']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Boxplot
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
fig.patch.set_facecolor("#ffffff")
num_cols = ["payment_sequential", "payment_installments", "payment_value"]
for i, (ax, col) in enumerate(zip(axes, num_cols)):
    ax.set_facecolor("#ffffff")
    bp = ax.boxplot(
        df_raw[col].dropna(),
        patch_artist=True,
        medianprops=dict(color=LINE_COL, linewidth=2),
        boxprops=dict(facecolor="#dbeafe", color="#93c5fd"),
        whiskerprops=dict(color="#93c5fd"),
        capprops=dict(color="#93c5fd"),
        flierprops=dict(marker="o", color="#3b82f6", alpha=0.3, markersize=3),
        widths=0.5,
    )
    ax.set_title(col, fontsize=8.5, fontweight="600", color="#374151", pad=8)
    ax.tick_params(labelsize=7, colors="#6b7280")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig.suptitle("Boxplot Deteksi Outlier", fontsize=10, fontweight="600", color="#111827", y=1.02)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown(
    """
    <div class="warn-box">
    ⚠️ <b>Outlier:</b> Ditemukan outlier pada ketiga kolom numerik. Outlier ini <b>tidak dihapus</b>
    karena mencerminkan keberagaman transaksi yang wajar (transaksi bernilai tinggi, cicilan panjang,
    atau penggunaan multi-metode pembayaran).
    </div>
    """,
    unsafe_allow_html=True,
)

if wm["n_inaccurate"] > 0:
    with st.expander(f"Lihat {wm['n_inaccurate']} data tidak akurat (payment_value ≤ 0)"):
        st.dataframe(wm["inaccurate_df"], use_container_width=True, hide_index=True)

if wm["n_not_defined"] > 0:
    with st.expander(f"Lihat {wm['n_not_defined']} data 'not_defined'"):
        st.dataframe(wm["not_def_df"], use_container_width=True, hide_index=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── 1.3 Cleaning ──────────────────────────────────────────────────────────────
st.markdown('<span class="step-badge">Step 3 — Cleaning Data</span>', unsafe_allow_html=True)

cl1, cl2, cl3 = st.columns(3)
cl1.markdown(
    f"<div class='stat-pill'>🗑 Dihapus (value ≤ 0): <b>{wm['n_inaccurate']} baris</b></div>",
    unsafe_allow_html=True,
)
cl2.markdown(
    f"<div class='stat-pill'>🗑 Dihapus (not_defined): <b>{wm['n_not_defined']} baris</b></div>",
    unsafe_allow_html=True,
)
cl3.markdown(
    f"<div class='stat-pill'>✅ Sisa data bersih: <b>{wm['n_clean']:,} baris</b></div>",
    unsafe_allow_html=True,
)

# Before vs After bar
fig, ax = bare_fig(w=8, h=3)
cats  = ["Sebelum Cleaning", "Setelah Cleaning"]
vals  = [wm["n_raw"], wm["n_clean"]]
bars  = ax.barh(cats, vals, color=["#93c5fd", "#3b82f6"], edgecolor="none", height=0.45)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 200, bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}", va="center", ha="left", fontsize=8.5, color="#374151")
ax.set_title("Jumlah Baris Sebelum vs Sesudah Cleaning",
             fontsize=9.5, fontweight="600", color="#111827", pad=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
style_ax(ax, "x")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown(
    f"""
    <div class="success-box">
    ✅ <b>Cleaning selesai.</b> Dihapus {wm['n_inaccurate']} baris dengan <code>payment_value ≤ 0</code>
    dan {wm['n_not_defined']} baris dengan <code>payment_type = not_defined</code>.
    Data bersih: <b>{wm['n_clean']:,} baris</b> siap dianalisis.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Lihat 5 baris data setelah cleaning"):
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EDA & VISUALISASI
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">🔍 EDA & Visualisasi</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-sub">Menjawab dua pertanyaan bisnis utama</p>',
    unsafe_allow_html=True,
)

# ── Pertanyaan 1 ──────────────────────────────────────────────────────────────
st.markdown('<span class="step-badge">Pertanyaan 1 — Metode Pembayaran</span>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:0.84rem;color:#4b5563;margin-bottom:14px;'>"
    "Metode pembayaran mana yang paling dominan dan berkontribusi nilai transaksi terbesar?"
    "</p>",
    unsafe_allow_html=True,
)

# EDA summary table
eda1 = pd.DataFrame({
    "Metode": payment_counts.index,
    "Jumlah Transaksi": payment_counts.values,
    "Total Nilai (BRL)": [payment_values.get(m, 0) for m in payment_counts.index],
})
eda1["% Transaksi"] = (eda1["Jumlah Transaksi"] / eda1["Jumlah Transaksi"].sum() * 100).map("{:.1f}%".format)
eda1["Total Nilai (BRL)"] = eda1["Total Nilai (BRL)"].map("{:,.0f}".format)
eda1["Jumlah Transaksi"] = eda1["Jumlah Transaksi"].map("{:,}".format)
st.dataframe(eda1.reset_index(drop=True), use_container_width=True, hide_index=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    fig, ax = bare_fig(6, 3.8)
    bars = ax.bar(payment_counts.index, payment_counts.values,
                  color=PALETTE, edgecolor="none", width=0.55)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 400, f"{int(h):,}",
                ha="center", va="bottom", fontsize=8, color="#374151")
    ax.set_title("Frekuensi Transaksi per Metode", fontsize=9.5, fontweight="600",
                 color="#111827", pad=10)
    ax.set_ylabel("Jumlah Transaksi", fontsize=8, color="#6b7280")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = bare_fig(6, 3.8)
    pv = payment_values.reindex(payment_counts.index, fill_value=0)
    bars = ax.bar(pv.index, pv.values, color=PALETTE, edgecolor="none", width=0.55)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 30000, f"{h/1e6:.2f}M",
                ha="center", va="bottom", fontsize=8, color="#374151")
    ax.set_title("Total Nilai Transaksi per Metode (BRL)", fontsize=9.5, fontweight="600",
                 color="#111827", pad=10)
    ax.set_ylabel("Total Nilai (BRL)", fontsize=8, color="#6b7280")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown(
    """
    <div class="insight-box">
    💡 <b>Insight:</b> Credit Card mendominasi dengan jumlah transaksi terbanyak sekaligus kontribusi nilai
    terbesar (~12,5 juta BRL). Boleto menjadi alternatif populer kedua, sementara Voucher dan Debit Card
    bersifat komplementer. Promo cashback / poin ekstra bagi pengguna kartu kredit dapat memperkuat loyalitas.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Pertanyaan 2 ──────────────────────────────────────────────────────────────
st.markdown('<span class="step-badge">Pertanyaan 2 — Cicilan & Nilai Transaksi</span>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:0.84rem;color:#4b5563;margin-bottom:14px;'>"
    "Apakah terdapat hubungan antara jumlah cicilan dengan rata-rata nilai transaksi?"
    "</p>",
    unsafe_allow_html=True,
)

with st.expander("Lihat tabel rata-rata nilai per jumlah cicilan"):
    show_it = installment_trend.copy()
    show_it.columns = ["Jumlah Cicilan", "Rata-rata Nilai (BRL)"]
    show_it["Rata-rata Nilai (BRL)"] = show_it["Rata-rata Nilai (BRL)"].map("{:,.2f}".format)
    st.dataframe(show_it, use_container_width=True, hide_index=True)

fig, ax = bare_fig(10, 3.8)
ax.plot(
    installment_trend["payment_installments"],
    installment_trend["payment_value"],
    color=LINE_COL, linewidth=2.2, marker="o", markersize=5,
    markerfacecolor="#ffffff", markeredgewidth=1.8, markeredgecolor=LINE_COL,
)
ax.fill_between(installment_trend["payment_installments"],
                installment_trend["payment_value"], alpha=0.08, color=LINE_COL)
ax.set_title("Rata-rata Nilai Transaksi berdasarkan Jumlah Cicilan",
             fontsize=9.5, fontweight="600", color="#111827", pad=10)
ax.set_xlabel("Jumlah Cicilan", fontsize=8, color="#6b7280")
ax.set_ylabel("Rata-rata Nilai (BRL)", fontsize=8, color="#6b7280")
style_ax(ax)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown(
    """
    <div class="insight-box">
    💡 <b>Insight:</b> Terdapat <b>korelasi positif</b> antara jumlah cicilan dan rata-rata nilai transaksi.
    Cicilan 1–3× → transaksi &lt; 150 BRL; cicilan &gt; 8× → transaksi bernilai lebih tinggi.
    Skema cicilan bunga rendah untuk produk premium efektif mendongkrak Average Order Value (AOV).
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ANALISIS LANJUTAN
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">🧩 Analisis Lanjutan — Segmentasi Pelanggan</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-sub">Manual grouping berdasarkan pola perilaku pembayaran</p>',
    unsafe_allow_html=True,
)

# Segmentation logic explanation
st.markdown(
    """
    <div class="card">
    <div style="font-size:0.84rem;font-weight:600;color:#374151;margin-bottom:10px;">
        🏷️ Logika Segmentasi
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.82rem;color:#4b5563;">
        <div>🔵 <b>High Spender</b> — payment_value &gt; 500 BRL</div>
        <div>🟣 <b>Long-term Installment</b> — payment_installments &gt; 12</div>
        <div>🟢 <b>Instant Payer</b> — payment_installments ≤ 1</div>
        <div>⚪ <b>Regular Installment</b> — cicilan 2–12×</div>
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# EDA table for segments
total_seg = segment_counts["count"].sum()
seg_display = segment_counts.sort_values("count", ascending=False).copy()
seg_display["% Share"] = (seg_display["count"] / total_seg * 100).map("{:.1f}%".format)
seg_display["Rata-rata Nilai (BRL)"] = seg_display["payment_segment"].map(
    df.groupby("payment_segment")["payment_value"].mean().map("{:,.2f}".format)
)
seg_display["Rata-rata Cicilan"] = seg_display["payment_segment"].map(
    df.groupby("payment_segment")["payment_installments"].mean().map("{:.1f}".format)
)
seg_display.columns = ["Segmen", "Jumlah", "% Share", "Rata-rata Nilai (BRL)", "Rata-rata Cicilan"]
st.dataframe(seg_display.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Charts
seg_sorted = segment_counts.sort_values("count", ascending=True)
col_ch, col_pie = st.columns([3, 2], gap="large")

with col_ch:
    fig, ax = bare_fig(7, 3.5)
    bars = ax.barh(seg_sorted["payment_segment"], seg_sorted["count"],
                   color=PALETTE[::-1], edgecolor="none", height=0.5)
    for b in bars:
        w = b.get_width()
        ax.text(w + 200, b.get_y() + b.get_height() / 2,
                f"{int(w):,}", va="center", ha="left", fontsize=8, color="#374151")
    ax.set_title("Distribusi Segmen Pelanggan", fontsize=9.5, fontweight="600",
                 color="#111827", pad=10)
    ax.set_xlabel("Jumlah Transaksi", fontsize=8, color="#6b7280")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    style_ax(ax, "x")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_pie:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#ffffff")
    wedges, texts, autotexts = ax.pie(
        seg_sorted["count"],
        labels=seg_sorted["payment_segment"],
        autopct="%1.1f%%",
        colors=PALETTE[::-1],
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in texts:
        t.set_fontsize(7.5)
        t.set_color("#374151")
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("#ffffff")
        at.set_fontweight("600")
    ax.set_title("Proporsi Segmen (%)", fontsize=9.5, fontweight="600",
                 color="#111827", pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Avg value per segment bar
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
avg_val_seg = df.groupby("payment_segment")["payment_value"].mean().sort_values(ascending=False)

fig, ax = bare_fig(10, 3.2)
bars = ax.bar(avg_val_seg.index, avg_val_seg.values,
              color=PALETTE, edgecolor="none", width=0.5)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 4,
            f"BRL {h:,.0f}", ha="center", va="bottom", fontsize=8, color="#374151")
ax.set_title("Rata-rata Nilai Transaksi per Segmen (BRL)",
             fontsize=9.5, fontweight="600", color="#111827", pad=10)
ax.set_ylabel("Rata-rata Nilai (BRL)", fontsize=8, color="#6b7280")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
style_ax(ax)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown(
    """
    <div class="insight-box">
    💡 <b>Insight:</b>
    <ul style="margin:6px 0 0;padding-left:18px;line-height:1.9;">
        <li><b>Instant Payer</b> mendominasi volume — pelanggan likuiditas tunai tinggi.
            Diskon eksklusif / cashback dapat mempertahankan segmen ini.</li>
        <li><b>High Spender</b> memiliki rata-rata nilai transaksi tertinggi — target ideal
            untuk program loyalitas premium atau penawaran produk eksklusif.</li>
        <li><b>Long-term Installment</b> paling sedikit, namun menunjukkan ketergantungan
            kredit — tawarkan cicilan bunga rendah untuk mempertahankan mereka.</li>
        <li><b>Regular Installment</b> adalah segmen tengah yang stabil — cocok untuk
            cross-selling produk dengan harga menengah.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — KESIMPULAN
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">✅ Kesimpulan</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(
        """
        <div class="card">
        <div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:10px;">
            💳 Pertanyaan 1: Metode Pembayaran Dominan
        </div>
        <ul style="font-size:0.82rem;color:#4b5563;padding-left:16px;margin:0;line-height:1.9;">
            <li><b>Credit Card</b> — metode paling dominan (±76% transaksi)</li>
            <li>Kontribusi nilai terbesar (~12,5 juta BRL)</li>
            <li><b>Boleto</b> menjadi alternatif utama (non-kartu)</li>
            <li>Voucher & Debit Card bersifat komplementer</li>
            <li>Rekomendasi: promo cashback / poin ekstra untuk pengguna kartu kredit</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
        <div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:10px;">
            📈 Pertanyaan 2: Cicilan & Nilai Transaksi
        </div>
        <ul style="font-size:0.82rem;color:#4b5563;padding-left:16px;margin:0;line-height:1.9;">
            <li>Korelasi <b>positif</b> antara jumlah cicilan dan nilai transaksi</li>
            <li>Cicilan 1–3× → produk harga terjangkau (&lt; 150 BRL)</li>
            <li>Cicilan &gt; 8× → produk bernilai lebih tinggi</li>
            <li>Fitur cicilan efektif mendongkrak AOV</li>
            <li>Rekomendasi: skema cicilan bunga rendah untuk produk premium</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="card" style="background:#f8fafc;">
    <div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:10px;">
        🧩 Analisis Lanjutan: Segmentasi Pelanggan
    </div>
    <div style="font-size:0.82rem;color:#4b5563;line-height:1.9;">
        Empat segmen strategis berhasil diidentifikasi dari data: <b>Instant Payer</b> (dominan),
        <b>High Spender</b> (nilai tertinggi), <b>Regular Installment</b> (segmen stabil),
        dan <b>Long-term Installment</b> (minoritas). Masing-masing segmen memiliki karakteristik
        dan strategi pemasaran yang berbeda untuk memaksimalkan revenue dan loyalitas pelanggan.
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align:center;font-size:0.78rem;color:#d1d5db;padding:24px 0 8px;'>"
    "E-Commerce Payment Dashboard · Yahya Ahmad · Dicoding Data Analysis Project"
    "</div>",
    unsafe_allow_html=True,
)