import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RoadGuard | AI Pothole Detector",
    page_icon="🛣️",
    layout="wide", # Changed back to wide for better control
    initial_sidebar_state="collapsed"
)

# --- INTERNAL CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Typography */
    h1 {
        font-family: 'Arial', sans-serif;
        color: #ffffff;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        font-size: 3rem;
    }
    h3 {
        text-align: center;
        color: #a0a0a0;
        font-weight: 400;
        margin-bottom: 30px;
    }
    
    /* Custom File Uploader */
    [data-testid='stFileUploader'] {
        margin: 0 auto;
    }
    .stFileUploader section {
        padding: 30px;
        background-color: #1E1E1E;
        border: 2px dashed #4CAF50;
        border-radius: 20px;
    }

    /* Result Cards */
    .result-container {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    
    .result-card {
        width: 100%;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        animation: slideUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
        color: white !important; /* Force white text */
    }
    
    .danger-card {
        background: linear-gradient(135deg, #d31027 0%, #ea384d 100%);
        border: 3px solid #ff6b6b;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #134E5E 0%, #71B280 100%);
        border: 3px solid #71B280;
    }
    
    /* Text Inside Cards */
    .big-icon { 
        font-size: 4rem; 
        margin-bottom: 15px; 
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .status-text { 
        font-size: 2.2rem; 
        font-weight: 900; 
        margin: 0; 
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
        color: #ffffff !important;
    }
    .conf-text { 
        font-size: 1.4rem; 
        font-weight: 500;
        margin-top: 10px; 
        opacity: 0.95;
        color: #f0f0f0 !important;
    }

    /* Hide Streamlit Footer & Menu */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- APP LOGIC ---

@st.cache_resource
def load_pothole_model():
    model_path = 'pothole_detector_final.h5'
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        return None

# Headers
st.markdown("<h1>RoadGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<h3>Real-Time Pothole Detection System</h3>", unsafe_allow_html=True)

model = load_pothole_model()

# Create a container to control width (approx 70% of screen)
_, main_col, _ = st.columns([1, 4, 1])

with main_col:
    if model is None:
        st.error("⚠️ Model file 'pothole_detector_final.h5' not found.")
    else:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a road image")

        if uploaded_file is not None:
            st.write("---")
            
            # Using columns for the analysis section
            col1, col2 = st.columns([1, 1], gap="large")
            
            # 1. DISPLAY IMAGE
            with col1:
                st.markdown("#### 📸 Input Feed")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            # 2. RUN ANALYSIS
            with col2:
                st.markdown("#### 🧠 AI Diagnosis")
                
                with st.spinner('Analyzing patterns...'):
                    time.sleep(1.0) # Demo delay
                    
                    try:
                        # Preprocess
                        img_resized = image.resize((224, 224))
                        x = img_to_array(img_resized)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)
                        
                        # Predict
                        prediction_prob = model.predict(x)[0][0]
                        
                        # Logic
                        THRESHOLD = 0.5
                        if prediction_prob > THRESHOLD:
                            confidence = prediction_prob * 100
                            status = "POTHOLE DETECTED"
                            card_style = "danger-card"
                            icon = "🚨"
                        else:
                            confidence = (1 - prediction_prob) * 100
                            status = "ROAD IS SAFE"
                            card_style = "safe-card"
                            icon = "✅"
                        
                        # Custom HTML Result Card
                        st.markdown(f"""
                        <div class="result-container">
                            <div class="result-card {card_style}">
                                <div class="big-icon">{icon}</div>
                                <p class="status-text">{status}</p>
                                <p class="conf-text">Confidence: {confidence:.2f}%</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence Bar
                        st.write("")
                        st.markdown("**System Confidence**")
                        st.progress(int(confidence) / 100)
                        
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")

# Footer Info
st.write("")
with st.expander("ℹ️  Technical Details"):
    st.markdown("""
    **Model:** MobileNetV2 (Transfer Learning)  
    **Input Resolution:** 224x224 RGB  
    **Training Accuracy:** 93.57%  
    **Latency:** <100ms
    """)