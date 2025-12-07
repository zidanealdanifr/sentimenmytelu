import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import nltk
import time
import joblib  

# --- 0. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen My TelU",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .stDataFrame { width: 100%; }
    h1, h2, h3 { color: #8B0000; } /* Merah Maroon TelU */
    
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. SETUP & RESOURCE ---
@st.cache_resource
def setup_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = setup_resources()

# --- FUNGSI LOAD MODEL DARI all_models.pkl ---
@st.cache_resource
def load_trained_models():
    try:
        # 1. Load Vectorizer
        vec = joblib.load('vectorizer.pkl')
        
        # 2. Load Paket Model (all_models.pkl)
        # Isinya dictionary: {'svm': model_svm, 'nb': model_nb}
        models_pack = joblib.load('all_models.pkl')
        
        # 3. Ekstrak Model
        nb = models_pack['nb']
        svm = models_pack['svm']
        
        return vec, nb, svm, None
    except Exception as e:
        return None, None, None, str(e)

# --- 2. FUNGSI LOAD DATA (Untuk Tampilan Tabel) ---
@st.cache_data
def load_dataset():
    # Load Ulasan
    try:
        df = pd.read_csv('ulasan_mytelu.csv')
        if 'userName' in df.columns:
            df.rename(columns={'userName': 'Nama'}, inplace=True)
    except FileNotFoundError:
        return None, None, None, "File 'ulasan_mytelu.csv' tidak ditemukan."

    # Load Normalisasi
    norm_dict = {}
    df_norm_display = pd.DataFrame()
    
    try:
        df_norm = pd.read_csv('normalization.csv')
        df_norm_display = df_norm.copy()
        
        if 'gaul' in df_norm.columns:
            norm_dict = dict(zip(df_norm['gaul'], df_norm['baku']))
        else:
            norm_dict = dict(zip(df_norm.iloc[:, 0], df_norm.iloc[:, 1]))
            df_norm_display.columns = ['gaul', 'baku']
    except FileNotFoundError:
        return None, None, None, "File 'normalization.csv' tidak ditemukan."

    return df, norm_dict, df_norm_display, None

# --- 3. FUNGSI PRE-PROCESSING ---
def cleaning_process_detailed(text, norm_dict):
    if not isinstance(text, str): return "", "", "", "", ""
    
    # Case Folding
    cf = text.lower()
    
    # Normalization (Logic)
    tokens_raw = word_tokenize(cf)
    norm_tokens = [norm_dict.get(t, t) for t in tokens_raw]
    norm_str = " ".join(norm_tokens) 
    
    # Tokenizing & Filtering
    tokens_clean = [t for t in norm_tokens if t.isalpha()]
    tok_str = str(tokens_clean)
    
    # Stemming
    stems = [stemmer.stem(t) for t in tokens_clean]
    stem_str = " ".join(stems)
    
    return cf, norm_str, tok_str, stem_str

# ==========================================
# UI UTAMA APLIKASI
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.info("""
    Penelitian ini bertujuan untuk menganalisis sentimen ulasan pengguna aplikasi **"My TelU"** serta membandingkan performa algoritma **Naïve Bayes** dan **Support Vector Machine (SVM)**.
    """)
    
    st.markdown("""
    **📂 Dataset & Pre-processing**
    Menggunakan **138 ulasan bersih** melalui tahapan:
    * Case Folding 
    * Normalization
    * Tokenizing
    * Stemming
    
    **⚙️ Metode Evaluasi**
    *K-Fold Cross-Validation* dengan variasi skenario **K = 3, 5, dan 10**.
    """) 
    
    st.divider()
    st.caption("Dibuat dengan ❤️ untuk My TelU")

# --- HEADER AREA ---
st.title("🎓 Analisis Sentimen Ulasan My TelU")
st.markdown("Dashboard monitoring sentimen pengguna aplikasi My TelU berdasarkan ulasan Google Play Store.")

st.divider()

# Load Data untuk UI
df_raw, norm_dict, df_norm_view, error_msg = load_dataset()

# Load Model untuk Prediksi 
vec_pkl, nb_pkl, svm_pkl, pkl_error = load_trained_models()

if error_msg:
    st.error(error_msg)
    st.stop()

# --- METRIC DASHBOARD ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Ulasan", f"{len(df_raw)}")
with c2:
    pos_raw = len(df_raw[df_raw['score'] > 3])
    st.metric("Sentimen Positif", f"{pos_raw}")
with c3:
    neg_raw = len(df_raw[df_raw['score'] < 3])
    st.metric("Sentimen Negatif", f"{neg_raw}")

st.write("") # Spacer

# --- STATE MANAGEMENT ---
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = False

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🛠️ Pre-processing", 
    "📈 Evaluasi Model", 
    "☁️ Word Cloud", 
    "🤖 Klasifikasi Live"
])

# --- TAB 1: PRE-PROCESSING ---
with tab1:
    st.subheader("Dataset & Kamus")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("**Dataset Ulasan**")
        st.dataframe(df_raw[['Nama', 'content', 'score']], use_container_width=True)
    
    with col_d2:
        st.write("**Kamus Normalisasi**")
        st.dataframe(df_norm_view, use_container_width=True)
            
    st.divider()
    st.subheader("Tahapan Pre-processing")
    
    process_btn = st.button("🚀 Jalankan Pre-processing", type="primary")
    
    if process_btn:
        with st.status("Sedang memproses data...", expanded=True) as status:
            time.sleep(1) 
            st.write("• Melakukan Case Folding...")
            st.write("• Normalisasi kata gaul...")
            st.write("• Tokenizing & Filtering...")
            st.write("• Stemming kata dasar...")
            
            # Proses Data
            df_proc = df_raw.copy()
            df_proc = df_proc[df_proc['score'] != 3].copy()
            df_proc['label'] = df_proc['score'].apply(lambda x: 0 if x > 3 else 1)
            df_proc.reset_index(drop=True, inplace=True)
            
            res_cf, res_norm, res_tok, res_stem = [], [], [], []
            
            for txt in df_proc['content']:
                cf, nm, tk, stm = cleaning_process_detailed(txt, norm_dict)
                res_cf.append(cf)
                res_norm.append(nm)
                res_tok.append(tk)
                res_stem.append(stm)
            
            df_proc['Case Folding'] = res_cf
            df_proc['Normalization'] = res_norm
            df_proc['Tokenizing'] = res_tok
            df_proc['Stemming'] = res_stem
            
            st.session_state.df_processed = df_proc
            st.session_state.is_processed = True
            
            status.update(label="Pre-processing Selesai!", state="complete", expanded=False)
    
    if st.session_state.is_processed:
        st.success("✅ Data berhasil dibersihkan.")
        st.write("Berikut adalah hasil dari setiap tahapan pre-processing:")
        cols_display = ['content', 'Case Folding', 'Normalization', 'Tokenizing', 'Stemming', 'label']
        st.dataframe(st.session_state.df_processed[cols_display], use_container_width=True)

# --- TAB 2: EVALUASI MODEL ---
with tab2:
    st.subheader("Hasil Evaluasi")
    
    if not st.session_state.is_processed:
        st.warning("⚠️ Mohon jalankan Pre-processing di Tab 1 terlebih dahulu.")
    else:
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            model_display = st.selectbox("Pilih Algoritma Model:", ["Naïve Bayes", "SVM"])
        with c_opt2:
            k_selected = st.radio("Pilih Skenario K-Fold:", [3, 5, 10], horizontal=True)
        
        st.write("")
        show_eval = st.button("Tampilkan Hasil Evaluasi")
        
        st.divider()
        
        if show_eval:
            try:
                df_eval_full = pd.read_csv('hasil_evaluasi.csv')
                model_csv_key = "Naive Bayes" if model_display == "Naïve Bayes" else "SVM"
                
                df_view = df_eval_full[
                    (df_eval_full['Skenario K'] == k_selected) & 
                    (df_eval_full['Model'] == model_csv_key)
                ].copy()
                
                if df_view.empty:
                    st.warning(f"Data evaluasi untuk {model_display} dengan K={k_selected} tidak ditemukan.")
                else:
                    st.markdown(f"### Performa {model_display} (K={k_selected})")
                    
                    cols_show = ['Accuracy', 'Precision (Positif)', 'Recall (Positif)', 'F1-Score (Positif)', 'Precision (Negatif)', 'Recall (Negatif)', 'F1-Score (Negatif)']
                    st.dataframe(df_view[cols_show], use_container_width=True)
                    
                    st.write("#### Confusion Matrix")
                    row = df_view.iloc[0]
                    tp, tn = row['TP'], row['TN']
                    fp, fn = row['FP'], row['FN']
                    cm_array = np.array([[tp, fn], [fp, tn]]) 
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_array, annot=True, fmt='g', cmap='Blues', 
                                xticklabels=['Positif (0)', 'Negatif (1)'], 
                                yticklabels=['Positif (0)', 'Negatif (1)'], ax=ax)
                    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(f"Confusion Matrix {model_display}")
                    st.pyplot(fig)

            except FileNotFoundError:
                st.error("File 'hasil_evaluasi.csv' tidak ditemukan.")

# --- TAB 3: WORD CLOUD ---
with tab3:
    st.subheader("Visualisasi Word Cloud")
    
    if not st.session_state.is_processed:
         st.warning("⚠️ Mohon jalankan Pre-processing terlebih dahulu.")
    else:
        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            st.success("Sentimen Positif")
            try: st.image("wordcloudpositif.png", use_container_width=True)
            except: st.error("Gambar wordcloudpositif.png tidak ditemukan")
        with wc_col2:
            st.error("Sentimen Negatif")
            try: st.image("wordcloudnegatif.png", use_container_width=True)
            except: st.error("Gambar wordcloudnegatif.png tidak ditemukan")

# --- TAB 4: LIVE PREDICTION ---
with tab4:
    st.subheader("Uji Coba Model (Live)")
    
    # Tampilkan Error jika file PKL tidak ditemukan
    if pkl_error:
        st.error(f"❌ Gagal memuat file Model: {pkl_error}")
        st.warning("Pastikan file 'vectorizer.pkl' dan 'all_models.pkl' ada di folder yang sama dengan app.py")
    else:
        col_input, col_pred = st.columns([2, 1])
        with col_input:
            input_text = st.text_area("Masukkan Ulasan Baru:", height=150, placeholder="Ketik ulasan di sini...")
            
        with col_pred:
            model_opt = st.selectbox("Model Prediksi:", ["Naïve Bayes", "SVM"])
            st.write("") 
            predict_btn = st.button("Analisis Sekarang 🔍", type="primary", use_container_width=True)
            
        if predict_btn:
            # 1. Preprocessing Input User (Gunakan fungsi cleaning yang sama)
            cf_txt, norm_txt, tok_txt, stem_txt = cleaning_process_detailed(input_text, norm_dict)
            
            st.markdown("### 📝 Tahapan Pre-processing")
            
            st.text_input("1. Case Folding", value=cf_txt, disabled=True)
            st.text_input("2. Normalization", value=norm_txt, disabled=True)
            st.text_area("3. Tokenizing", value=tok_txt, disabled=True) 
            st.text_input("4. Stemming (Input Model)", value=stem_txt, disabled=True)
            
            st.divider()
            
            if stem_txt:
                # 2. Vectorize menggunakan VEC dari PKL
                pred_vec = vec_pkl.transform([stem_txt])
                
                # 3. Pilih Model dari PKL (yang sudah di-unpack di awal)
                if model_opt == "SVM":
                    model = svm_pkl
                else:
                    model = nb_pkl
                
                # 4. Prediksi
                pred = model.predict(pred_vec)[0]
                proba = model.predict_proba(pred_vec).max()
                
                st.markdown("### 🎯 Hasil Analisis Sentimen")
                if pred == 0:
                    st.success(f"**SENTIMEN POSITIF 😊**\nConfidence Score: {proba:.2%}")
                else:
                    st.error(f"**SENTIMEN NEGATIF 😡**\nConfidence Score: {proba:.2%}")
            else:
                st.warning("⚠️ Teks input kosong atau hanya berisi simbol.")
