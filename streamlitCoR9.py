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
        background: url("https://images.pexels.com/photos/716656/pexels-photo-716656.jpeg");
        background-size: cover;
    }
    body::before {
        content: "";
        position: fixed;
        inset: 0;
        backdrop-filter: blur(100px);
        background-color: rgba(0, 0, 0, 0.0);
        z-index: 0;
    }
    .stApp {
        background: none;
    }
    
    /* Global text color for all elements */
    body, .st-emotion-cache-j7qwjs, .st-emotion-cache-1kyx5z6, .st-emotion-cache-10a4v9k > div > label > div > p, h1, h2, h3, h4, h5, h6, p, li, div, span, a {
    color: white !important;
    }
    
    /* Specifically for the 'System Status' info box and its content */
    .st-emotion-cache-14j989e {
        color: white !important; /* Force text to be white for this info box */
        border-left-color: white !important; /* Change the border color too */
    }
    /* Specifically for the 'System Status' info box and its content */
    .st-emotion-cache-14j989e {
        color: white !important;
        border-left-color: white !important;
    }
    
    /* NEW: This targets the specific container you want to make black */
    .st-emotion-cache-vgzhz4 {
        background-color: #000000;
    }
    
    /* NEW: This targets the sidebar and its content to set a dark color */
    .st-emotion-cache-1lqf7hx {
        background-color: #282828;
        color: #ffffff;
    }
    .st-emotion-cache-1lqf7hx h1,
    .st-emotion-cache-1lqf7hx .st-emotion-cache-1kyx5z6,
    .st-emotion-cache-1lqf7hx .st-emotion-cache-10a4v9k > div > label > div > p {
        color: #ffffff;
    }

    /* Consolidated Sidebar Styling: Black background with white text and a clean glassmorphism effect */
    .st-emotion-cache-12fmw3r, .st-emotion-cache-18ni7ap {
        background-color: #000000;
        color: #ffffff;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #000000;
        color: #ffffff;
    }

    /* Consolidated Sidebar Styling: Black background with white text and a clean glassmorphism effect */
    .st-emotion-cache-12fmw3r, .st-emotion-cache-18ni7ap, .sidebar .sidebar-content {
        background-color: #000000;
        color: #ffffff; /* Sets all text inside the sidebar to white */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
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
    
    /* Input fields */
    .stTextInput>div>div>input, .stFileUploader>div>button, .stSelectbox>div>div {
        border-radius: 12px;
        border: 2px solid #667eea;
        background: rgba(255, 255, 255, 0.8);
    }
    
    /* Main content glass effect */
    .st-emotion-cache-10o5j50 {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .space1 {
        margin-top: 112px;
    }
    .space2 {
        margin-top: 26px;
    }
    .st-emotion-cache-vgzhz4 {
    background-color: #282828;
}
"""
with open("style.css", "w") as f:
    f.write(css_content)
local_css("style.css")

# --- Model Loading with Caching ---
@st.cache_resource
def load_yolo_model():
    # model = YOLO("best.pt")
    model = YOLO("best2.pt")
    
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
    if mode != "🎥 Webcam":
        st.markdown('<div class="space1"></div>', unsafe_allow_html=True)
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
    mode = st.radio(
        "Choose a mode:",
        ("📷 Image Upload", "🎥 Webcam"),
        horizontal=True
    )
    
    col1, col2= st.columns(2)

    with col1:
        st.subheader("📤 Input")
        image_data = None

        if mode == "📷 Image Upload":
            # with col3:
                uploaded_file = st.file_uploader(
                    "Upload a corn image:",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Click to upload or drag and drop your corn image here."
                )
                if uploaded_file is not None:
                    image_data = uploaded_file.getvalue()
                    st.image(image_data, caption="Corn Image Preview", use_container_width=True)

        elif mode == "🎥 Webcam":
            st.markdown('<div class="space2"></div>', unsafe_allow_html=True)
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
        # else:
        #     st.markdown("""
        #     <div style="
        #         border: 2px dashed #ccc;
        #         border-radius: 12px;
        #         padding: 40px;
        #         text-align: center;
        #         background: #f9f9f9;
        #         color: #888;
        #     ">
        #         <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
        #         <div>Prediction results will appear here</div>
        #         <div style="font-size: 0.75rem; margin-top: 0.5rem;">
        #             Upload an image or use the webcam to see results.
        #         </div>
        #     </div>
        #     """, unsafe_allow_html=True)

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
        - **Detection Types:** Object Bounding Boxes detection
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
                  CoR9 มีวัตถุประสงค์หลักของการพัฒนาโมเดล คือการสร้างระบบตรวจจับและจำแนกประเภทเมล็ดข้าวโพดอย่างแม่นยำและมีประสิทธิภาพ 
    โดยโมเดลจะสามารถแยกแยะเมล็ดข้าวโพดที่ถูกตรวจจับออกเป็นประเภทต่าง ๆ ได้อย่างอัตโนมัติ ซึ่งเป็นกระบวนการสำคัญในการติดตามคุณภาพ 
    คัดแยกความหลากหลาย และประเมินมาตรฐานของผลผลิตทางการเกษตร การนำเทคโนโลยี AI มาใช้ในกระบวนการนี้ 
    จะช่วยลดเวลาและขั้นตอนการทำงานเมื่อเทียบกับการตรวจสอบด้วยวิธีการแบบดั้งเดิม 
    อีกทั้งยังสามารถลดความผิดพลาดที่มักเกิดจากการประเมินด้วยสายตาหรือการทำงานของบุคลากร 
    ซึ่งเป็นปัญหาที่พบได้บ่อยในภาคการผลิต และการวิจัยด้านเกษตรกรรม 
    """)
    col1, col2 = st.columns(2)
    with col1:
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
    with col2:
        st.markdown("</div>", unsafe_allow_html=True)
    
        st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight:bold;'>🛠️ Technology Stack</h3>", unsafe_allow_html=True)
        st.write("""
        - **Frontend:** Streamlit, Custom CSS
        - **AI:** YOLO (You Only Look Once), Computer Vision, Deep Learning
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='mode-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:bold;'>📞 Contact & Support</h3>", unsafe_allow_html=True)
    st.write("""GitHub: https://github.com/Mametoyas/CoR9""")
    st.write("""Email: Chakkaphan.m@kkumail.com""")
    st.write("""PromptPay: 0986452734""")
    st.markdown("</div>", unsafe_allow_html=True)

