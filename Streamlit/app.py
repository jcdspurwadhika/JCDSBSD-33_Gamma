import streamlit as st
import pandas as pd
 
from utils.model_helper import load_model, predict
from utils.constants import (
    FEATURE_ORDER, JOBS, MARITALS, EDUCATIONS,
    DEFAULTS, HOUSINGS, LOANS, CONTACTS,
    MONTHS, DAYS, POUTCOMES, SAMPLE_ROW
)
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Telemarketing Predictor",
    page_icon="🏦",
    layout="wide"
)
 
# ── Load model ────────────────────────────────────────────────────────────────
pipeline, best_score = load_model()
 
 
# ── Helper ────────────────────────────────────────────────────────────────────
def result_label(pred: int) -> str:
    return "Akan Berlangganan" if pred == 1 else "Tidak Akan Berlangganan"
 
 
# ── Header ────────────────────────────────────────────────────────────────────
st.title("Bank Telemarketing Predictor")
st.caption(f"Model: LogisticRegression · F2-Score terbaik: **{best_score:.4f}**")
 
with st.expander("Tentang aplikasi ini"):
    st.markdown("""
    Aplikasi ini memprediksi apakah seorang nasabah **akan berlangganan deposito berjangka**
    berdasarkan data kampanye telemarketing.
 
    **Model:** LogisticRegression dengan SMOTE untuk menangani *imbalanced data*.
    Fitur **duration** sengaja dihapus dari data untuk menghindari *data leakage*,
    karena durasi panggilan hanya diketahui **setelah** panggilan selesai — sehingga
    tidak tersedia saat prediksi dilakukan secara real-world.
    **Metrik utama:** F2-Score (mengutamakan meminimalkan *False Negative*).
 
    **Cara pakai:**
    - **Single Prediction** — isi form lalu klik *Predict*.
    - **Batch Upload (CSV)** — upload file CSV berisi banyak nasabah sekaligus.
      Unduh template di tab Batch untuk memastikan format kolom sudah benar.
    """)
 
tab_single, tab_batch = st.tabs(["🔍 Single Prediction", "📄 Batch Upload (CSV)"])
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Input Data Nasabah")
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("**Info Nasabah**")
        age     = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=35)
        job     = st.selectbox("Pekerjaan", JOBS)
        marital = st.selectbox("Status Pernikahan", MARITALS)
        edu     = st.selectbox("Pendidikan", EDUCATIONS)
 
    with col_b:
        st.markdown("**Status Keuangan**")
        default = st.selectbox("Kredit Macet (default)?", DEFAULTS)
        housing = st.selectbox("Pinjaman Rumah (housing loan)?", HOUSINGS)
        loan    = st.selectbox("Pinjaman Pribadi (personal loan)?", LOANS)
 
    st.divider()
    col_c, col_d = st.columns(2)
 
    with col_c:
        st.markdown("**Info Kampanye**")
        contact     = st.selectbox("Tipe Kontak", CONTACTS)
        month       = st.selectbox("Bulan Kontak Terakhir", MONTHS)
        day_of_week = st.selectbox("Hari Kontak Terakhir", DAYS)
        campaign    = st.number_input("Jumlah Kontak di Kampanye Ini", min_value=1, value=2)
 
    with col_d:
        st.markdown("**Riwayat Kontak**")
        pdays    = st.number_input("Hari Sejak Kontak Terakhir (999 = belum pernah)",
                                   min_value=0, max_value=999, value=999)
        previous = st.number_input("Jumlah Kontak Sebelum Kampanye Ini", min_value=0, value=0)
        poutcome = st.selectbox("Hasil Kampanye Sebelumnya", POUTCOMES)
        was_prev = st.radio(
            "Pernah Dihubungi Sebelumnya?", [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )
 
    st.divider()
    st.markdown("**Indikator Ekonomi**")
 
    eco1, eco2, eco3 = st.columns(3)
    eco4, eco5       = st.columns(2)
 
    with eco1:
        emp_var_rate = st.number_input(
            "Employment Variation Rate", value=3.3, step=0.1, format="%.1f",
            help="Tingkat variasi pekerjaan kuartalan"
        )
    with eco2:
        cons_price_idx = st.number_input(
            "Consumer Price Index", value=93.000, step=0.001, format="%.3f",
            help="Indeks harga konsumen bulanan"
        )
    with eco3:
        cons_conf_idx = st.number_input(
            "Consumer Confidence Index", value=-36.7, step=0.1, format="%.1f",
            help="Indeks kepercayaan konsumen bulanan"
        )
    with eco4:
        euribor3m = st.number_input(
            "Euribor 3 Month Rate", value=3.0, step=0.001, format="%.3f",
            help="Suku bunga euribor 3 bulan harian"
        )
    with eco5:
        nr_employed = st.number_input(
            "Number of Employees (ribuan)", value=5191.0, step=0.1, format="%.1f",
            help="Jumlah karyawan — indikator kuartalan"
        )
 
    st.divider()
 
    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": edu,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "campaign": campaign, "pdays": pdays,
            "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "nr.employed": nr_employed, "was_previously_contacted": was_prev
        }])
 
        preds, probas = predict(input_df, pipeline)
 
        st.subheader("Hasil Prediksi")
        r1, r2 = st.columns(2)
        with r1:
            if preds[0] == 1:
                st.success(result_label(preds[0]))
            else:
                st.error(result_label(preds[0]))
        with r2:
            st.metric("Probabilitas Berlangganan", f"{probas[0]*100:.1f}%")
 
# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Upload
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Upload File CSV")
 
    template_df = pd.DataFrame([SAMPLE_ROW])[FEATURE_ORDER]
    st.download_button(
        label="⬇Download Template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="template_telemarketing.csv",
        mime="text/csv"
    )
    st.info("Pastikan kolom CSV sesuai dengan template di atas. Urutan kolom boleh berbeda.")
 
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
 
    if uploaded is not None:
        try:
            try:
                batch_df = pd.read_csv(uploaded)
                if batch_df.shape[1] == 1:
                    uploaded.seek(0)
                    batch_df = pd.read_csv(uploaded, sep=';')
            except:
                uploaded.seek(0)
                batch_df = pd.read_csv(uploaded, sep=';')
 
            if 'was_previously_contacted' not in batch_df.columns:
                batch_df['was_previously_contacted'] = (batch_df['pdays'] != 999).astype(int)
 
            if 'y' in batch_df.columns:
                batch_df = batch_df.drop(columns=['y'])
 
            st.write(f"**{len(batch_df)} baris** berhasil dibaca.")
            st.dataframe(batch_df.head(), use_container_width=True)
 
            missing = [c for c in FEATURE_ORDER if c not in batch_df.columns]
            if missing:
                st.error(f"Kolom berikut tidak ditemukan: {missing}")
            else:
                if st.button("Predict Semua", type="primary"):
                    preds, probas = predict(batch_df, pipeline)
 
                    result_df = batch_df.copy()
                    result_df["Prediksi"]         = preds
                    result_df["Label"]            = result_df["Prediksi"].map(
                        {1: "Akan Berlangganan", 0: "Tidak"}
                    )
                    result_df["Probabilitas (%)"] = (probas * 100).round(1)
 
                    display_cols = ["Label", "Probabilitas (%)"] + FEATURE_ORDER
                    st.subheader("Hasil Prediksi Batch")
                    st.dataframe(result_df[display_cols], use_container_width=True)
 
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Total Nasabah", len(preds))
                    s2.metric("Prediksi Berlangganan", int(preds.sum()))
                    s3.metric("Prediksi Tidak", int(len(preds) - preds.sum()))
 
                    st.download_button(
                        label="⬇Download Hasil CSV",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error membaca file: {e}")
