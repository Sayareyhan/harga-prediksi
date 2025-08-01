import joblib

# Muat file daftar kolom
try:
    kolom_model = joblib.load('model_harga_mobil_final/model_columns_final.joblib')
    print("--- DAFTAR LENGKAP KOLOM MODEL ---")
    # Cetak setiap kolom dalam baris baru agar mudah dibaca
    for kolom in kolom_model:
        print(kolom)
    print("-----------------------------------")
except Exception as e:
    print(f"Gagal memuat file kolom: {e}")