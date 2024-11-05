import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

# Load the vocabulary
vocabulary = pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))
loader_vec = TfidfVectorizer(
    decode_error="replace", vocabulary=set(vocabulary))

# Fit the vectorizer on dummy data to initialize it
loader_vec.fit(["dummy data"])

# CSS for centering and justifying text
st.markdown("""
    <style>
        .center-content {
            text-align: center;
        }
        .justify-text {
            text-align: justify;
        }
        .center-table {
            display: flex;
            justify-content: center;
        }
        .center-table table {
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Judul Halaman WEB (Centered)
st.markdown("""
<h2 class="center-content">MENDETEKSI PESAN PENIPUAN PADA SMS</h2>
<h4 class="center-content">Oleh: Muhammad Gilang Dwi Saputra 21.11.4233</h4>
""", unsafe_allow_html=True)

# Menambahkan gambar
st.image('tipu.jpg')

# Dropdown untuk Tentang Aplikasi
section = st.selectbox(
    'Pilih Kategori',
    (
        'Klik di sini untuk memilih kategori',
        '1. Tentang Aplikasi',
        '2. Cara Penggunaan',
        '3. Keterangan Input Data',
        '4. Tentang Model',
        '5. Contoh Pesan SMS'
    )
)

# Konten berdasarkan pilihan dropdown
if section == '1. Tentang Aplikasi':
    st.markdown("""
    <div class="justify-text">
     Tentang Aplikasi

    Aplikasi web ini bertujuan untuk membantu pengguna dalam mengidentifikasi pesan SMS yang berpotensi merupakan penipuan.
    Dengan memanfaatkan model Naïve Bayes yang telah dilatih menggunakan dataset berisi berbagai contoh pesan SMS, aplikasi 
    ini dapat menganalisis teks dari setiap pesan yang diterima. Proses analisis ini memungkinkan aplikasi untuk menghitung 
    probabilitas bahwa suatu pesan termasuk dalam kategori penipuan berdasarkan kata-kata dan frasa yang terdapat di dalamnya. 
    Setelah analisis, aplikasi memberikan prediksi yang jelas kepada pengguna, yaitu apakah pesan tersebut kemungkinan besar 
    adalah penipuan atau tidak. Dengan demikian, aplikasi ini berfungsi sebagai alat yang efektif untuk melindungi pengguna 
    dari potensi penipuan yang sering kali dilakukan melalui SMS, sehingga para pengguna dapat membuat keputusan yang lebih baik dan 
    mengurangi risiko jatuh ke dalam penipuan. Hal ini diharapkan dapat meningkatkan kewaspadaan dan keamanan pengguna terhadap 
    ancaman penipuan melalui pesan SMS.
    </div>
    """, unsafe_allow_html=True)

elif section == '2. Cara Penggunaan':
    st.markdown("""
     Cara Penggunaan

    1. Masukkan teks SMS ke dalam kotak input.
    2. Klik tombol 'Hasil Deteksi' untuk melihat prediksi.
    3. Prediksi akan muncul di bawah tombol.
    """)

elif section == '3. Keterangan Input Data':
    st.markdown("""## Keterangan Input Data""")
    data = {
        'Label': [0, 1, 2],
        'Keterangan': ['SMS Normal', 'SMS Penipuan', 'SMS Promo']
    }
    df = pd.DataFrame(data)

    # Center the table
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.table(df)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == '4. Tentang Model':
    st.markdown("""
    <div class="justify-text">
     Tentang Model

    Model yang digunakan dalam aplikasi ini adalah model Naïve Bayes yang telah dilatih menggunakan dataset yang berisi seputar SMS. 
    Model ini digunakan untuk mengklasifikasikan pesan SMS menjadi kategori Normal, Penipuan, Dan Promo. Dengan asumsi bahwa setiap fitur atau 
    kata dalam pesan SMS bersifat independen, Naïve Bayes menghitung probabilitas sebuah pesan termasuk ke dalam salah satu kategori. 
    Model ini bekerja dengan baik untuk data teks seperti SMS karena kemampuannya dalam menangani data berukuran besar dan hasilnya 
    cukup akurat, meskipun asumsi independensinya sederhana. Hasil klasifikasi ini sangat membantu dalam menyaring pesan yang tidak 
    diinginkan secara otomatis.
    </div>
    """, unsafe_allow_html=True)

elif section == '5. Contoh Pesan SMS':
    st.markdown("""
     Contoh Pesan SMS 

    0. Dilantai 2 ya tepat diruang skripsi
    1. INFO RESMI kepada bpk/ibu yang terhormat SELAMAT No ANDA Pemenang ke 2dari pesta isi ulang 3care No PIN:25e477r di undi tadi malam pkl 22:30 di SCTV U/info hadiah anda klik Www.3care16.blogspot.com,1
    2. Akhir bulan harus tetap eksis loh! Internetan pake volume ultima 900MB/30hr. Hrga mulai Rp 35rb di *100*471#. Tarif&lokasi cek di tsel.me/fl,2
    """)

# Input teks SMS
clean_teks = st.text_input('Masukan Teks SMS')

fraud_detection = ''

if st.button('Hasil Deteksi'):
    if clean_teks:  # Pastikan input tidak kosong
        transformed_text = loader_vec.transform([clean_teks])
        predik_fraud = model_fraud.predict(transformed_text)
        predik_proba = model_fraud.predict_proba(transformed_text)

        if predik_fraud == 0:
            fraud_detection = f'SMS Normal dengan akurasi {
                predik_proba[0][0]*100:.2f}%'
        elif predik_fraud == 1:
            fraud_detection = f'SMS Penipuan dengan akurasi {
                predik_proba[0][1]*100:.2f}%'
        else:
            fraud_detection = f'SMS Promo dengan akurasi {
                predik_proba[0][2]*100:.2f}%'

        st.success(fraud_detection)
    else:
        st.warning("Silakan masukkan teks SMS sebelum melakukan prediksi.")
