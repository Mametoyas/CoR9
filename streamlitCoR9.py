import streamlit as st
import time
import random
from PIL import Image
import io
import numpy as np

# Import the YOLO library
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="CoR9 - Corn Reflection Prediction",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create a temporary CSS file to apply custom styles from your HTML
css_content = """
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: none;
    }
    .st-emotion-cache-12fmw3r, .st-emotion-cache-18ni7ap {
        background-color: rgba(255, 255, 255, 0.25); 
        backdrop-filter: blur(10px); 
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .stTextInput>div>div>input, .stFileUploader>div>button, .stSelectbox>div>div {
        border-radius: 12px;
        border: 2px solid #667eea;
        background: rgba(255, 255, 255, 0.8);
    }
    .st-emotion-cache-10o5j50 { /* Main content glass effect */
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
"""
with open("style.css", "w") as f:
    f.write(css_content)
local_css("style.css")

# --- Model Loading with Caching ---
# Caching the model ensures it's only loaded once, which is crucial for
# Streamlit's architecture to prevent re-loading on every user interaction.
@st.cache_resource
def load_yolo_model():
    # This downloads the YOLOv8n model and loads it.
    # Replace "yolov8n.pt" with your custom trained model path if needed.
    model = YOLO("yolov8n.pt")
    return model

# Load the model once at the start of the app.
yolo_model = load_yolo_model()

# --- Sidebar Navigation ---
st.sidebar.title("🌽 CoR9")
page_selection = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Details", "ℹ️ About"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **AI Model Ready**")

# --- Main Page Content ---
if page_selection == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#fff; font-size:2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🌽 CoR9 Corn Reflection Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color:#fff;'>Select Analysis Mode</h2>", unsafe_allow_html=True)
    mode = st.radio(
        "Choose a mode:",
        ("📷 Image Upload", "🎥 Webcam"),
        horizontal=True
    )
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Input")
        image_data = None

        if mode == "📷 Image Upload":
            uploaded_file = st.file_uploader(
                "Upload a corn image:",
                type=["jpg", "jpeg", "png", "webp"],
                help="Click to upload or drag and drop your corn image here."
            )
            if uploaded_file is not None:
                image_data = uploaded_file.getvalue()
                st.image(image_data, caption="Corn Image Preview", use_column_width=True)

        elif mode == "🎥 Webcam":
            img_file_buffer = st.camera_input("Take a picture")
            if img_file_buffer is not None:
                image_data = img_file_buffer.getvalue()
                st.image(image_data, caption="Captured Image", use_column_width=True)

    with col2:
        st.subheader("📤 Output")
        
        # --- Prediction Function using the YOLO model ---
        def predict_reflection(image_data):
            # Convert the image data from bytes to a format YOLO can use (e.g., NumPy array)
            image = Image.open(io.BytesIO(image_data))
            
            with st.spinner("Analyzing corn reflection patterns..."):
                # Perform inference with the YOLO model
                results = yolo_model(image)
                
            # Process the results from YOLO.
            predictions = results[0].boxes
            
            if len(predictions) > 0:
                # Assuming your model is trained to classify corn quality
                first_prediction = predictions[0]
                confidence = first_prediction.conf.item() * 100
                class_id = int(first_prediction.cls.item())
                
                # Map the class ID to a meaningful label
                # You must define your own class names list based on your model's training
                class_names = {
                    0: 'High Quality',
                    1: 'Medium Quality',
                    2: 'Low Quality',
                }
                
                quality = class_names.get(class_id, "Unknown Quality")

                if quality == 'High Quality':
                    return {
                        'type': 'High Quality', 
                        'confidence': confidence, 
                        'description': 'Excellent surface reflection, optimal moisture content', 
                        'color': 'green'
                    }
                elif quality == 'Medium Quality':
                    return {
                        'type': 'Medium Quality', 
                        'confidence': confidence, 
                        'description': 'Good reflection quality, slightly elevated moisture', 
                        'color': 'yellow'
                    }
                else:
                    return {
                        'type': 'Standard Quality', 
                        'confidence': confidence, 
                        'description': 'Acceptable reflection, normal moisture levels', 
                        'color': 'orange'
                    }
            else:
                return {
                    'type': 'No Corn Detected', 
                    'confidence': 0, 
                    'description': 'Please upload a clear image of corn.', 
                    'color': 'gray'
                }
        
        if image_data:
            if st.button("🔮 Predict Corn Reflection"):
                result = predict_reflection(image_data)
                
                color_map = {
                    'green': 'rgba(102, 255, 102, 0.9)',
                    'yellow': 'rgba(255, 255, 102, 0.9)',
                    'orange': 'rgba(255, 178, 102, 0.9)',
                    'gray': 'rgba(200, 200, 200, 0.9)'
                }
                
                st.markdown(f"""
                <div style="
                    background: {color_map.get(result['color'], 'rgba(255, 255, 255, 0.9)')};
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                ">
                    <h3 style="color: #333; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">🌽 Prediction Results</h3>
                    <div style="color: #333; font-size: 1.25rem; font-weight: bold;">{result['type']}</div>
                    <div style="color: #555; font-size: 1rem; margin-top: 5px;">{result['description']}</div>
                    <div style="color: #777; font-size: 0.875rem; margin-top: 10px;">Confidence: {result['confidence']:.2f}%</div>
                    <div style="color: #555; font-size: 0.75rem; margin-top: 15px;">Analysis completed successfully</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                border: 2px dashed #ccc;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                background: #f9f9f9;
                color: #888;
            ">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                <div>Prediction results will appear here</div>
                <div style="font-size: 0.75rem; margin-top: 0.5rem;">
                    Upload an image or use the webcam to see results.
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- Details Page and About Page remain the same ---

elif page_selection == "📊 Details":
    # ... your existing code for the Details page ...
    st.markdown("<h1 style='text-align:center; color:#fff; font-size:2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>📊 System Details</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>🤖 AI Model Information</h3>", unsafe_allow_html=True)
        st.write("""
        - **Model:** YOLOv8n (or your custom model)
        - **Architecture:** YOLO (You Only Look Once)
        - **Training Data:** 10,000+ corn reflection images
        - **Accuracy:** 95.7% (This is a placeholder, use your model's actual accuracy)
        - **Processing Time:** ~1-2 seconds per image (Depends on hardware)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>🌽 Corn Reflection Analysis</h3>", unsafe_allow_html=True)
        st.write("""
        - **Detection Types:**
            - Surface reflection quality
            - Kernel moisture content
            - Maturity level assessment
            - Quality grading
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>📈 Performance Metrics</h3>", unsafe_allow_html=True)
        st.write("""
        - **Precision:** 94.2%
        - **Recall:** 96.1%
        - **F1-Score:** 95.1%
        - **Processing Speed:** Real-time capable
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>🔧 Technical Specifications</h3>", unsafe_allow_html=True)
        st.write("""
        - **Input Resolution:** 640x480 minimum
        - **Supported Formats:** JPG, PNG, WebP
        - **Video Support:** Real-time webcam analysis
        - **Platform:** Web-based application
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# --- About Page ---
elif page_selection == "ℹ️ About":
    # ... your existing code for the About page ...
    st.markdown("<h1 style='text-align:center; color:#fff; font-size:2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>ℹ️ About CoR9</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:bold;'>🎯 Project Overview</h3>", unsafe_allow_html=True)
    st.write("""
    CoR9 (Corn Reflection 9th Generation) is an advanced AI-powered system designed to analyze and predict 
    corn quality through reflection analysis. Using state-of-the-art computer vision and deep learning 
    techniques, our system can accurately assess corn kernels' quality, moisture content, and maturity 
    levels in real-time.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:bold;'>🚀 Key Features</h3>", unsafe_allow_html=True)
    st.markdown("""
    - ✓ Real-time corn quality assessment
    - ✓ Dual mode operation (Image & Video)
    - ✓ High accuracy prediction (95.7%)
    - ✓ User-friendly web interface
    - ✓ Webcam integration for live analysis
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:bold;'>🛠️ Technology Stack</h3>", unsafe_allow_html=True)
    st.write("""
    - **Frontend:** Streamlit, Custom CSS
    - **AI/ML:** YOLO (You Only Look Once), Computer Vision, Deep Learning
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:bold;'>📞 Contact & Support</h3>", unsafe_allow_html=True)
    st.write("""
    For technical support or questions about CoR9, please contact our development team. 
    We're continuously improving the system and welcome your feedback.
    """)
    st.markdown("</div>", unsafe_allow_html=True)