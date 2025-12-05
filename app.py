import streamlit as st
import os
import pandas as pd

st.title("🕵️ MODE DETEKTIF: Mencari File Hilang")

# 1. Tampilkan di mana posisi program saat ini berjalan
current_folder = os.getcwd()
st.write(f"📂 **Posisi Program:** `{current_folder}`")

# 2. Tampilkan SEMUA file yang dilihat server di folder ini
files = os.listdir('.')
st.write("📜 **Daftar File yang Dilihat Server:**")
st.code(files)

# 3. Cek Spesifik File CSV
target_file = 'ulasan_mytelu.csv'

if target_file in files:
    st.success(f"✅ KETEMU! File '{target_file}' sebenarnya ADA.")
    st.write("Coba baca isinya:")
    try:
        df = pd.read_csv(target_file)
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"File ada tapi tidak bisa dibaca: {e}")
else:
    st.error(f"❌ TIDAK KETEMU! Server tidak melihat file bernama '{target_file}'")
    st.warning("Coba lihat daftar file di atas (kotak abu-abu). Apakah ada file yang namanya mirip tapi beda huruf besar/kecil? Atau beda spasi?")
