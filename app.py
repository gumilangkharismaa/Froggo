import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import ModifiedDenseNet169  # Pastikan file model.py ada
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import gdown

# URL dan nama file model
MODEL_URL = "https://drive.google.com/file/d/1QIN6-yl5aOF33EjmCssCDTcHhoz2iYrt/view?usp=sharing"
MODEL_FILE = "model_densenet169_frog_classifier_9_classes.pth"

# Fungsi download model
def download_model():
    """Download model file jika belum tersedia."""
    if not os.path.exists(MODEL_FILE):
        with st.spinner(f"Mengunduh file model: {MODEL_FILE}..."):
            gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Fungsi memuat model
@st.cache_resource
def load_model():
    """Memuat model setelah memastikan file model tersedia."""
    download_model()
    model = ModifiedDenseNet169(num_classes=9)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    return model

# Fungsi untuk mengolah file audio menjadi MFCC
def process_audio_to_mfcc(mp3_path):
    try:
        # Muat file audio
        y, sr = librosa.load(mp3_path, sr=None)
        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_normalized = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc)) * 255  # Normalisasi
        mfcc_image = Image.fromarray(mfcc_normalized.astype(np.uint8)).convert("L")  # Konversi ke grayscale
        return mfcc_image
    except Exception as e:
        raise RuntimeError(f"Error processing audio to MFCC: {e}")

# Fungsi prediksi
def predict(image, model, transform, label_dict):
    """Melakukan prediksi menggunakan model."""
    image = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    label = [k for k, v in label_dict.items() if v == predicted.item()]
    return label[0] if label else "Unknown"

# Fungsi mendapatkan gambar katak berdasarkan label prediksi
def get_frog_image(label):
    image_path = f"frog_images/{label}.jpg"
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return None

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((224, 224)),               
    transforms.RandomAffine(degrees=10, shear=20),  
    transforms.ToTensor(),                      
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dictionary label
label_dict = {
    'Boana': 0,
    'Pool': 1,
    'PepperFrog': 2,
    'South': 3,
    'Dendropsophus': 4,
    'Leptodactylus': 5,
    'Rana': 6,
    'Rhinella': 7,
    'Scinax': 8
}

# Antarmuka Streamlit
st.title("Klasifikasi Suara Katak")
st.write("Unggah file audio MP3 untuk memprediksi jenis suara katak.")

uploaded_file = st.file_uploader("Unggah file .mp3", type=["mp3"])

if uploaded_file is not None:
    try:
        # Simpan file audio sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(uploaded_file.getbuffer())
            temp_audio_path = temp_audio_file.name

        # Proses audio menjadi MFCC
        mfcc_image = process_audio_to_mfcc(temp_audio_path)

        # Tampilkan MFCC
        st.image(mfcc_image, caption="MFCC dari Audio Input", use_column_width=True)

        # Muat model
        model = load_model()

        # Lakukan prediksi
        label = predict(mfcc_image, model, transform, label_dict)

        st.success(f"Prediksi: {label}")

        # Tampilkan gambar katak berdasarkan prediksi
        frog_image = get_frog_image(label)
        if frog_image:
            st.image(frog_image, caption=f"Gambar: {label}", use_column_width=True)
        else:
            st.warning(f"Tidak ada gambar yang cocok untuk label: {label}")

        # Hapus file sementara
        os.remove(temp_audio_path)
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
