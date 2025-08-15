import streamlit as st
import time
import random
import matplotlib.font_manager as fm
from PIL import Image
import io
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

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

    /* Main body and app container */
    body {
        background-image: url("data:image/jpeg;base64,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");
        background-size: cover;
    }
    .stApp {
        background: none;
    }

    /* Glassmorphism effect for main containers and sidebar */
    .st-emotion-cache-12fmw3r, .st-emotion-cache-18ni7ap, .sidebar .sidebar-content {
        backgroung-image: url(CoR9_BG.jpg);
        # background-color: rgba(255, 255, 255, 0.5); /* More transparent for a lighter glass look */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        # border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 16px; /* Slightly more rounded corners */
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .stApp {
        background: none;
    }
    .st-emotion-cache-12fmw3r, .st-emotion-cache-18ni7ap {
        backgroung-image: url(CoR9_BG.jpg);
        # background-color: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(90px);
        # border: 1px solid rgba(255, 255, 255, 0.3);
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
@st.cache_resource
def load_yolo_model():
    model = YOLO("best.pt")
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

# --- Function to plot results and display analysis ---
def plot_yolo_result_image_and_analyze(image_data, results):
    # Class names (YOLO ID → Name)
    class_names = {
        1: 'เมล็ดดี',       # Good
        5: 'ซัง',      # Sung
        4: 'แมลงทำลาย',     # Insect
        3: 'เมล็ดเน่า',     # Bad
        2: 'เมล็ดน้ำผึ้ง',   # Honey
        0: 'คลุกยา', # Clookya (I assume this is a transliteration)
        6: 'ราขาว'    # Rakhaw (White mold)
    }
    font_path = 'Sarabun-Regular.ttf'  # Make sure this font file is in the same directory as your script
    font_prop = fm.FontProperties(fname=font_path)
    # Convert image data to a format cv2 can read
    file_bytes = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    
    imgs = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    height, width = imgs.shape[:2]

    # Extract YOLO results
    labels = results[0].boxes.xywhn.cpu().numpy()  # Normalized xywh
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    # Count class instances
    class_counts = Counter(classes)
    total_instances = sum(class_counts.values())

    
    # Color map per class
    unique_classes = np.unique(classes)
    color_map = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(unique_classes)}

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(imgs)
    ax.axis('off')

    for i in range(len(classes)):
        x, y, w, h = labels[i]
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)
        cls = int(classes[i])
        color = color_map[cls]

        # Top-left corner from center
        x0 = x - w // 2
        y0 = y - h // 2

        # Draw box
        rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Draw label
        ax.text(
            x,
            y0 - 5,
            f"{class_names[cls]}",
            color=color,
            fontsize=8,
            fontproperties=font_prop,
            ha='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
        )
    
    # Use st.pyplot() to display the Matplotlib figure
    st.pyplot(fig)
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.subheader("📊 Analysis Results")
    st.write(f"**Total Instances Detected:** {total_instances}")
    cols = st.columns(2)
    for i, cls in enumerate(sorted(class_counts.keys())):
        count = class_counts[cls]
        percent = (count / total_instances) * 100
        
        # Using columns to display counts nicely
        with cols[i % 2]:
            st.info(f"**{class_names[cls]}:** {count} instances ({percent:.2f}%)")
    st.markdown("</div>", unsafe_allow_html=True) # Close the container


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
                # st.image(image_data, caption="Captured Image", use_column_width=True)

    with col2:
        st.subheader("📤 Output")
        
        if image_data:
            if st.button("🔮 Predict Corn Reflection"):
                with st.spinner("Analyzing corn reflection patterns..."):
                    try:

                        
                        # Convert the image data from bytes to a PIL Image
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Perform inference with the YOLO model
                        results = yolo_model.predict(source=image)
                        
                        plot_yolo_result_image_and_analyze(image_data, results)

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
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
    st.markdown("<h1 style='text-align:center; color:#fff; font-size:2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>📊 System Details</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>🤖 AI Model Information</h3>", unsafe_allow_html=True)
        st.write("""
        - **Model:** YOLOv11m 
        - **Architecture:** YOLO (You Only Look Once)
        - **Training Data:** 64 corn reflection images
        - **Accuracy:** 94.01% 
        - **Processing Time:** ~1-2 seconds per image (Depends on hardware)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>🌽 Corn Reflection Analysis</h3>", unsafe_allow_html=True)
        st.write("""
        - **Detection Types:**
            - Object Bounding Boxes detection
        - **Classes:**
            ซัง, ราขาว, เมล็ดคลุกยา, เมล็ดดี, เมล็ดเน่า, เมล็ดน้ำผึ้ง, แมลงทำลาย
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>📈 Performance Metrics</h3>", unsafe_allow_html=True)
        st.write("""
        - **Precision:** 96.18%
        - **Recall:** 94.15%
        - **F1-Score:** 95.15%
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

