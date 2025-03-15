import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from PIL import Image
import base64
import os
import requests
import io

# ---- Gemini API Key ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("🚨 Gemini API Key not found! Please set the environment variable `GEMINI_API_KEY`.")
    st.stop()

# ---- Gemini AI Analysis Function ----
def analyze_image_with_gemini(image_data):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}

    buffered = io.BytesIO()
    image_data.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Analyze this brain MRI image and provide an accurate medical explanation regarding the possibility of a brain tumor."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Analysis unavailable.")
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error retrieving analysis: {e}"

# ---- Base64 Image Encoding Function with Debug (No UI Notification) ----
def get_base64_of_bin_file(bin_file_path):
    try:
        if not os.path.exists(bin_file_path):
            st.warning(f"⚠️ File not found: {bin_file_path}. Full path: {os.path.abspath(bin_file_path)}")
            return None
        with open(bin_file_path, "rb") as f:
            data = f.read()
        base64_result = base64.b64encode(data).decode()
        print(f"Debug: Encoded {bin_file_path} to base64: {base64_result[:50]}...")  # Debug in terminal only
        return base64_result
    except Exception as e:
        st.error(f"⚠️ Error encoding {bin_file_path}: {e}")
        return None

# ---- Background & Custom CSS (Tanpa Animasi Gerak) ----
script_dir = os.path.dirname(os.path.abspath(__file__))
main_bg = os.path.join(script_dir, "mainimg1.jpg")
sidebar_bg = os.path.join(script_dir, "sb2.jpg")

main_bg_base64 = get_base64_of_bin_file(main_bg)
sidebar_bg_base64 = get_base64_of_bin_file(sidebar_bg)

# Fallback to online images if local files fail
main_bg_url = (f'data:image/jpeg;base64,{main_bg_base64}' if main_bg_base64 else "https://via.placeholder.com/1920x1080?text=AI+Brain+Detector+BG")
sidebar_bg_url = (f'data:image/jpeg;base64,{sidebar_bg_base64}' if sidebar_bg_base64 else "https://via.placeholder.com/300x1080?text=Sidebar+BG")

st.markdown(f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Poppins:wght@400;700&display=swap');

/* General Styling */
[data-testid="stAppViewContainer"] {{
    background: url("{main_bg_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: overlay;
    background-color: rgba(20, 20, 20, 0.9);
    color: #ffffff;
    min-height: 100vh;
    /* Hapus animasi gradientMove */
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background: url("{sidebar_bg_url}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: overlay;
    background-color: rgba(20, 20, 20, 0.75);
    color: #ffffff;
    border-right: 3px solid #00ffcc;
    box-shadow: 0 0 20px rgba(0, 255, 204, 0.5);
    animation: neonBorder 2s infinite alternate;
}}
@keyframes neonBorder {{
    0% {{ border-right-color: #00ffcc; box-shadow: 0 0 10px rgba(0, 255, 204, 0.5); }}
    100% {{ border-right-color: #00ccff; box-shadow: 0 0 20px rgba(0, 204, 255, 0.8); }}
}}

/* Title Styling */
.title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8em;
    text-align: center;
    color: #00ffcc;
    text-shadow: 2px 2px 10px rgba(0, 255, 204, 0.8), 0 0 20px rgba(0, 255, 204, 0.5);
    animation: glow 2s ease-in-out infinite alternate, fadeIn 1.5s ease-in-out;
}}
@keyframes glow {{
    from {{ text-shadow: 2px 2px 10px rgba(0, 255, 204, 0.8), 0 0 20px rgba(0, 255, 204, 0.5); }}
    to {{ text-shadow: 2px 2px 20px rgba(0, 255, 204, 1), 0 0 40px rgba(0, 255, 204, 0.8); }}
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

/* Subtitle Styling */
.subtitle {{
    font-family: 'Poppins', sans-serif;
    font-size: 1.2em;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Intro Text Styling */
.intro-text {{
    font-family: 'Poppins', sans-serif;
    font-size: 1em;
    text-align: center;
    color: #e0e0e0;
    margin-top: 10px;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    animation: fadeIn 2.5s ease-in-out;
}}

/* Button Styling */
.stButton>button {{
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    font-size: 18px;
    padding: 10px 20px;
    box-shadow: 0 0 15px rgba(255, 65, 108, 0.5), 0 0 30px rgba(255, 75, 43, 0.3);
    transition: all 0.3s ease;
    animation: fadeIn 2.5s ease-in-out;
}}
.stButton>button:hover {{
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    box-shadow: 0 0 25px rgba(255, 65, 108, 0.8), 0 0 50px rgba(255, 75, 43, 0.5);
}}

/* Disclaimer Styling */
.disclaimer {{
    text-align: center;
    background: rgba(255, 165, 0, 0.3);
    padding: 15px;
    border-radius: 10px;
    border: 2px solid #ffcc00;
    color: #ffffff;
    font-weight: bold;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Sidebar Links Styling */
.sidebar-link {{
    display: inline-block;
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    color: #1a1a1a;
    padding: 8px 15px;
    border-radius: 20px;
    text-decoration: none;
    font-family: 'Poppins', sans-serif;
    font-weight: bold;
    margin: 5px 0;
    transition: all 0.3s ease;
    animation: fadeIn 2.5s ease-in-out;
}}
.sidebar-link:hover {{
    transform: scale(1.05);
    background: linear-gradient(90deg, #00ccff, #00ffcc);
    box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);
}}

/* Sidebar Text Styling */
.sidebar-text {{
    font-family: 'Poppins', sans-serif;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    animation: fadeIn 2s ease-in-out;
}}

/* Result Card Styling */
.result-card {{
    background: rgba(20, 20, 20, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    border: 2px solid #00ffcc;
    box-shadow: 0 0 20px rgba(0, 255, 204, 0.5);
    color: #ffffff;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}}
.result-card h3 {{
    font-family: 'Orbitron', sans-serif;
    color: #00ffcc;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    margin-bottom: 10px;
}}
.result-card p {{
    font-family: 'Poppins', sans-serif;
    font-size: 1.1em;
    color: #ffffff;
    margin: 5px 0;
}}
.normal {{
    color: #00ff00;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
}}
.tumor {{
    color: #ff4444;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
}}

/* Expander Styling */
.stExpander {{
    background: rgba(20, 20, 20, 0.9);
    border-radius: 10px;
    border: 1px solid #00ffcc;
    color: #ffffff;
    animation: fadeIn 1.5s ease-in-out;
}}
.stExpander summary {{
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    color: #00ffcc;
    text-shadow: 0 0 5px rgba(0, 255, 204, 0.5);
}}
.stExpander p {{
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
}}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<p class="title">🧠 AI Brain Tumor Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🚀 Powered by Deep Learning & Google Gemini</p>', unsafe_allow_html=True)
st.markdown('<p class="intro-text">Advanced AI tool to detect brain tumors from MRI scans with high accuracy.</p>', unsafe_allow_html=True)

# ---- Load or Train Model ----
model_path = "brain_tumor_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.warning("⚠️ Model not found. Training a new model...")
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(4, activation='softmax')(x)  # 4 classes: no_tumor, glioma, meningioma, pituitary
    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        'brain-tumor-classification-mri/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        'brain-tumor-classification-mri/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    with st.spinner("⏳ Training model (this may take a while)..."):
        model.fit(train_generator, validation_data=validation_generator, epochs=10)
        model.save(model_path)
        st.sidebar.success("✅ Model trained and saved successfully!")

# ---- Sidebar Disclaimer ----
st.sidebar.markdown("""
<div class="disclaimer">
    ⚠️ <b>Disclaimer:</b><br>
    This tool is not a substitute for professional medical diagnosis.<br>
    Consult a doctor for accurate results.
</div>
""", unsafe_allow_html=True)

# ---- Sidebar Developer Info ----
st.sidebar.markdown("""
---
<p class="sidebar-text">👨‍💻 <b>Developed by Farrel0xx</b></p>
<div style="text-align: center;">
    <a href="https://github.com/Farrel0xx" target="_blank" class="sidebar-link">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align: middle; margin-right: 5px;"> GitHub
    </a><br>
    <a href="https://youtube.com/@Farrel0xx" target="_blank" class="sidebar-link">
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" width="30" style="vertical-align: middle; margin-right: 5px;"> YouTube
    </a>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar How It Works ----
st.sidebar.markdown("""
---
<p class="sidebar-text">ℹ️ <b>How It Works</b></p>
<p style="color: #e0e0e0;">
    1. Upload a brain MRI image in JPG, JPEG, or PNG format.<br>
    2. The AI, powered by VGG19, classifies the image into one of four categories: No Tumor, Glioma, Meningioma, or Pituitary Tumor.<br>
    3. Google Gemini provides a detailed medical explanation of the MRI scan.<br>
    4. View the prediction results, confidence score, and detailed probabilities.
</p>
""", unsafe_allow_html=True)

# ---- Upload Image ----
uploaded_file = st.sidebar.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

# ---- Prediction ----
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🧠 Uploaded MRI Scan", use_container_width=True)
    st.success("✅ Image uploaded successfully!")

    if st.button("🔎 Analyze"):
        with st.spinner("⏳ Processing..."):
            img = img.convert("RGB")
            img_resized = cv2.resize(np.array(img), (224, 224))
            img_resized = img_resized / 255.0
            img_preprocessed = np.expand_dims(img_resized, axis=0)

            prediction = model.predict(img_preprocessed)
            prediction_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][prediction_class] * 100
            labels = ["✅ No Tumor", "⚠️ Glioma Tumor", "⚠️ Meningioma Tumor", "⚠️ Pituitary Tumor"]
            result = labels[prediction_class]
            result_class = "normal" if prediction_class == 0 else "tumor"

            # ---- Prediction Result ----
            st.markdown(f"""
            <div class="result-card">
                <h3>📊 Prediction Result</h3>
                <p>Classification: <span class="{result_class}">{result}</span></p>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # ---- Gemini AI Explanation ----
            st.markdown('<div class="result-card"><h3>📝 AI Explanation</h3>', unsafe_allow_html=True)
            uploaded_file.seek(0)
            desc = analyze_image_with_gemini(img)
            st.markdown(f"<p>{desc}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Detailed Probabilities ----
            with st.expander("📌 View Detailed Probabilities"):
                st.write(f"✅ No Tumor: {prediction[0][0]:.4f}")
                st.write(f"⚠️ Glioma Tumor: {prediction[0][1]:.4f}")
                st.write(f"⚠️ Meningioma Tumor: {prediction[0][2]:.4f}")
                st.write(f"⚠️ Pituitary Tumor: {prediction[0][3]:.4f}")
