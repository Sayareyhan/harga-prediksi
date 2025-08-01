import streamlit as st
import pandas as pd
import joblib
import os

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="AutoPrice - Prediksi Harga Mobil",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Memuat Model dan Kolom ---
# Pastikan path ke model dan kolom benar
MODEL_PATH = 'model_harga_mobil_final/car_price_model_final.joblib'
COLUMNS_PATH = 'model_harga_mobil_final/model_columns_final.joblib'

model = None
model_columns = []

if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        st.success("✅ Model dan kolom berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau file kolom: {e}")
        st.warning("Pastikan file model dan kolom berada di direktori yang benar.")
else:
    st.error("❌ File model atau kolom tidak ditemukan.")
    st.warning(f"Mencari di: {MODEL_PATH} dan {COLUMNS_PATH}")

# --- Fungsi Prediksi ---
def predict_car_price(brand, model_name, year, mileage, transmission, fuel, engine_size, condition):
    if not model or not model_columns:
        st.error("Model tidak siap untuk prediksi. Harap periksa log.")
        return None

    try:
        # Siapkan kerangka data (dictionary) yang bersih
        feature_dict = {col: 0 for col in model_columns}

        # Isi nilai-nilai numerik
        feature_dict['Year'] = float(year)
        feature_dict["KM's driven"] = float(mileage)
        # Anda mungkin perlu menambahkan 'engine_size' dan 'condition' jika model Anda menggunakannya
        # feature_dict['Engine_Size'] = float(engine_size) # Sesuaikan nama kolom
        # feature_dict[f'Condition_{condition.lower()}'] = 1 # Sesuaikan nama kolom

        # Proses dan cocokkan nilai-nilai kategorikal
        # Pastikan nama kolom di model_columns cocok dengan format ini
        # Contoh: 'make_toyota', 'model_avanza', 'fuel_petrol', 'transmission_automatic'

        # Brand
        possible_brand_col = f'make_{brand.lower()}'
        if possible_brand_col in model_columns:
            feature_dict[possible_brand_col] = 1
        else:
            st.warning(f"Merek '{brand}' tidak ditemukan dalam kolom model. Prediksi mungkin kurang akurat.")

        # Model Name (jika ada di model_columns)
        possible_model_col = f'model_{model_name.lower()}'
        if possible_model_col in model_columns:
            feature_dict[possible_model_col] = 1
        else:
            st.warning(f"Model '{model_name}' tidak ditemukan dalam kolom model. Prediksi mungkin kurang akurat.")

        # Fuel Type
        possible_fuel_col = f'fuel_{fuel.lower()}'
        if possible_fuel_col in model_columns:
            feature_dict[possible_fuel_col] = 1
        else:
            st.warning(f"Tipe bahan bakar '{fuel}' tidak ditemukan dalam kolom model. Prediksi mungkin kurang akurat.")

        # Transmission
        possible_transmission_col = f'transmission_{transmission.lower()}'
        if possible_transmission_col in model_columns:
            feature_dict[possible_transmission_col] = 1
        else:
            st.warning(f"Tipe transmisi '{transmission}' tidak ditemukan dalam kolom model. Prediksi mungkin kurang akurat.")

        # Buat DataFrame final dari dictionary
        final_df = pd.DataFrame([feature_dict])

        # Pastikan urutan kolom 100% sama dengan saat model dilatih
        final_df = final_df[model_columns]

        # Lakukan prediksi
        prediction = model.predict(final_df)
        return prediction[0]

    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")
        return None

# --- Antarmuka Pengguna Streamlit ---
st.title("🚗 AutoPrice - Prediksi Harga & Pemilihan Mobil")
st.markdown("""
    Gunakan teknologi machine learning untuk memperkirakan harga mobil berdasarkan berbagai parameter.
    Dapatkan rekomendasi mobil terbaik sesuai kebutuhan Anda.
""")

# Tabs for Prediction and Colab Integration
tab1, tab2 = st.tabs(["Prediksi Harga", "Integrasi Colab"])

with tab1:
    st.header("Prediksi Harga Mobil")
    st.write("Masukkan detail mobil Anda untuk mendapatkan perkiraan harga yang akurat.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            brand = st.selectbox("Merek Mobil", ["", "Toyota", "Honda", "Suzuki", "Daihatsu", "Mitsubishi", "Nissan", "BMW", "Mercedes"])
            year = st.number_input("Tahun Produksi", min_value=1990, max_value=2023, value=2015, step=1)
            transmission = st.selectbox("Transmisi", ["Automatic", "Manual"])
            engine_size = st.number_input("Kapasitas Mesin (cc)", min_value=500, max_value=8000, value=1500, step=100)

        with col2:
            model_name = st.text_input("Model (Contoh: Avanza, Civic)")
            mileage = st.number_input("Kilometer (KM)", min_value=0, max_value=500000, value=50000, step=1000)
            fuel = st.selectbox("Bahan Bakar", ["Petrol", "Diesel", "Hybrid", "Electric"])
            condition = st.selectbox("Kondisi", ["Excellent", "Good", "Fair", "Poor"])

        submitted = st.form_submit_button("Prediksi Harga")

        if submitted:
            if not brand or not model_name:
                st.warning("Harap isi Merek Mobil dan Model.")
            else:
                with st.spinner("Memproses prediksi Anda..."):
                    predicted_price = predict_car_price(brand, model_name, year, mileage, transmission, fuel, engine_size, condition)
                    if predicted_price is not None:
                        st.success("Prediksi Harga Selesai!")
                        st.markdown(f"""
                            <div style="background-color:#2563eb; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                                <p style="font-size: 18px; margin-bottom: 5px;">HARGA PREDIKSI</p>
                                <p style="font-size: 36px; font-weight: bold;">Rp {predicted_price:,.0f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Gagal mendapatkan prediksi. Periksa input Anda atau log server.")

with tab2:
    st.header("Integrasi dengan Google Colab")
    st.write("Untuk menggunakan model machine learning dari Google Colab, salin kode berikut:")
    st.code("""
# Kode untuk integrasi dengan Google Colab
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load model (ganti dengan path model Anda)
# Pastikan file model_harga_mobil_final/car_price_model_final.joblib dan
# model_harga_mobil_final/model_columns_final.joblib sudah diupload ke Colab Anda
with open('model_harga_mobil_final/car_price_model_final.joblib', 'rb') as file:
    model = pickle.load(file)
with open('model_harga_mobil_final/model_columns_final.joblib', 'rb') as file:
    model_columns = pickle.load(file)

# Fungsi untuk prediksi harga
def predict_car_price_colab(brand, model_name, year, mileage, transmission, fuel):
    feature_dict = {col: 0 for col in model_columns}
    feature_dict['Year'] = float(year)
    feature_dict["KM's driven"] = float(mileage)

    # Set categorical features
    if f'make_{brand.lower()}' in model_columns:
        feature_dict[f'make_{brand.lower()}'] = 1
    if f'model_{model_name.lower()}' in model_columns:
        feature_dict[f'model_{model_name.lower()}'] = 1
    if f'fuel_{fuel.lower()}' in model_columns:
        feature_dict[f'fuel_{fuel.lower()}'] = 1
    if f'transmission_{transmission.lower()}' in model_columns:
        feature_dict[f'transmission_{transmission.lower()}'] = 1

    final_df = pd.DataFrame([feature_dict])
    final_df = final_df[model_columns] # Ensure column order

    prediction = model.predict(final_df)
    return prediction[0]

# Contoh penggunaan
# predicted_price = predict_car_price_colab('toyota', 'avanza', 2018, 50000, 'automatic', 'petrol')
# print(f"Predicted price: Rp {predicted_price:,.2f}")
    """, language="python")
    st.info("Catatan: Fungsi `predict_car_price_colab` di atas mungkin perlu disesuaikan jika model Anda menggunakan fitur `engine_size` atau `condition`.")

# --- Bagian "Pilih Mobil Terbaik" dan "Koleksi Mobil Kami" (Opsional) ---
# Bagian ini tidak akan berfungsi secara dinamis tanpa database atau data statis.
# Anda bisa menambahkan data mobil statis di sini atau menghubungkannya ke database.
st.markdown("---")
st.header("Pilih Mobil Terbaik")
st.write("Temukan mobil yang sesuai dengan kebutuhan dan budget Anda dari koleksi kami.")

# Contoh data mobil statis (Anda bisa menggantinya dengan data dari database)
sample_cars = [
    {"brand": "Toyota", "model": "Supra", "type": "Sports Car", "price": "Rp 1.200.000.000", "image": "https://placehold.co/400x300"},
    {"brand": "Honda", "model": "CR-V", "type": "SUV", "price": "Rp 450.000.000", "image": "https://placehold.co/400x300"},
    {"brand": "Mercedes", "model": "S-Class", "type": "Luxury Sedan", "price": "Rp 1.800.000.000", "image": "https://placehold.co/400x300"},
    {"brand": "Suzuki", "model": "Ertiga", "type": "MPV", "price": "Rp 220.000.000", "image": "https://placehold.co/400x300"},
    {"brand": "BMW", "model": "X5", "type": "Luxury SUV", "price": "Rp 1.500.000.000", "image": "https://placehold.co/400x300"},
    {"brand": "Daihatsu", "model": "Ayla", "type": "City Car", "price": "Rp 130.000.000", "image": "https://placehold.co/400x300"},
]

# Filter Section (Simplified for Streamlit)
st.subheader("Filter Mobil")
col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
with col_filter1:
    filter_brand = st.selectbox("Merek", ["Semua Merek"] + list(set([car["brand"] for car in sample_cars])))
with col_filter2:
    filter_price = st.selectbox("Harga", ["Semua Harga", "Di bawah 300 juta", "300 - 600 juta", "Di atas 600 juta"])
with col_filter3:
    filter_type = st.selectbox("Tipe", ["Semua Tipe"] + list(set([car["type"] for car in sample_cars])))
with col_filter4:
    filter_year = st.selectbox("Tahun", ["Semua Tahun", "2020-2023", "2015-2019", "2010-2014"]) # Static for now

filtered_cars = []
for car in sample_cars:
    match = True
    if filter_brand != "Semua Merek" and car["brand"] != filter_brand:
        match = False
    # Simplified price filter logic
    if filter_price != "Semua Harga":
        car_price_val = float(car["price"].replace("Rp ", "").replace(".", ""))
        if filter_price == "Di bawah 300 juta" and car_price_val >= 300000000:
            match = False
        elif filter_price == "300 - 600 juta" and not (300000000 <= car_price_val < 600000000):
            match = False
        elif filter_price == "Di atas 600 juta" and car_price_val < 600000000:
            match = False
    if filter_type != "Semua Tipe" and car["type"] != filter_type:
        match = False
    # Year filter is static, needs more complex logic if dynamic
    if match:
        filtered_cars.append(car)

st.subheader("Daftar Mobil")
if filtered_cars:
    cols_car = st.columns(3) # Display 3 cars per row
    for i, car in enumerate(filtered_cars):
        with cols_car[i % 3]:
            st.image(car["image"], caption=car["model"], use_column_width=True)
            st.markdown(f"**{car['brand']} {car['model']}**")
            st.write(f"Tipe: {car['type']}")
            st.write(f"Harga: {car['price']}")
            st.button(f"Lihat Detail {car['model']}", key=f"detail_{car['model']}_{i}")
else:
    st.info("Tidak ada mobil yang cocok dengan filter Anda.")

st.markdown("---")
st.header("Koleksi Mobil Kami")
st.write("Berbagai jenis mobil dari berbagai merek terkenal di dunia.")

# Carousel-like display (simplified)
car_collection_cols = st.columns(6) # Adjust number of columns as needed
for i, car in enumerate(sample_cars):
    with car_collection_cols[i % 6]: # Cycle through columns
        st.image(car["image"], use_column_width=True)
        st.markdown(f"**{car['brand']} {car['model']}**")
        st.write(car["type"])

st.markdown("---")
st.header("Tentang AutoPrice")
st.write("Platform prediksi harga mobil berbasis teknologi machine learning terdepan.")

col_about1, col_about2, col_about3 = st.columns(3)
with col_about1:
    st.subheader("Teknologi Canggih")
    st.write("Menggunakan model machine learning terbaru yang terus diperbarui untuk memberikan prediksi harga yang akurat berdasarkan data pasar terkini.")
with col_about2:
    st.subheader("Data Ekstensif")
    st.write("Database kami mencakup ribuan transaksi mobil dari berbagai merek dan model untuk memastikan prediksi yang representatif.")
with col_about3:
    st.subheader("Transparan & Akurat")
    st.write("Kami memberikan breakdown faktor-faktor yang mempengaruhi prediksi harga sehingga Anda bisa memahami bagaimana harga tersebut ditentukan.")

st.markdown("---")
st.markdown("© 2023 AutoPrice. All rights reserved.")
