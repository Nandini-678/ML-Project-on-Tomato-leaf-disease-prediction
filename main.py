import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# Set Streamlit page configuration
st.set_page_config(
    page_title="Tomato Plant Leaf Disease Classification",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to download model from Google Drive using the File ID
def download_model(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Load EfficientNetB0 model with caching
@st.cache_resource
def load_efficientnet_model():
    model_path = "efficientnetb0_model.h5"
    file_id = "1mmbRK-CRPFKoCFEfDLSYtD6ulZ8B1h7u"  # EfficientNetB0 File ID
    try:
        with open(model_path, "rb"):
            pass  # Model file exists
    except FileNotFoundError:
        download_model(file_id, model_path)
    return tf.keras.models.load_model(model_path)

# Load MobileNetV2 model with caching
@st.cache_resource
def load_mobilenet_model():
    model_path = "mobilenetv2_model.h5"
    file_id = "1EUeV1TrDv7E0ExAW96d92st3MSuTpVff"  # MobileNetV2 File ID
    try:
        with open(model_path, "rb"):
            pass  # Model file exists
    except FileNotFoundError:
        download_model(file_id, model_path)
    return tf.keras.models.load_model(model_path)

efficientnet_model = load_efficientnet_model()
mobilenet_model = load_mobilenet_model()

# Class names
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match the model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict with a given model
def predict(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100
    return class_names[predicted_class_idx], confidence

# Add custom CSS for theme compatibility
st.markdown(
    """
    <style>
    /* App Background */
    .stApp {
        background: rgb(242,49,58);
        background: radial-gradient(circle, rgba(242,49,58,1) 0%, rgba(233,148,177,1) 100%);
        color: black;
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: rgb(18,2,29);
            background: linear-gradient(90deg, rgba(18,2,29,1) 0%, rgba(208,36,36,1) 50%, rgba(116,69,2,1) 100%);
            color: white;
        }
    }

    /* Header and Subheader Styling */
    .header-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader-title {
        font-size: 18px;
        text-align: center;
        margin-bottom: 1em;
    }

    /* Upload Section Styling */
    .upload-section {
        border: 2px dashed #ccc; /* Neutral border color */
        padding: 20px;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.9); /* Light transparent */
        border-radius: 10px;
    }
    @media (prefers-color-scheme: dark) {
        .upload-section {
            border: 2px dashed #555; /* Dark border color */
            background-color: rgba(0, 0, 0, 0.5); /* Dark transparent */
        }
    }

    /* Prediction Section Styling */
    .prediction-section {
        background-color: rgba(255, 255, 255, 0.9); /* Light transparent */
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    @media (prefers-color-scheme: dark) {
        .prediction-section {
            background-color: rgba(0, 0, 0, 0.5); /* Dark transparent */
        }
    }

    /* Sidebar Styling */
    .sidebar .css-1d391kg {
        background-color: rgba(240, 240, 240, 0.8); /* Light mode sidebar */
    }
    @media (prefers-color-scheme: dark) {
        .sidebar .css-1d391kg {
            background-color: rgba(30, 30, 30, 0.8); /* Dark mode sidebar */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.markdown('<div class="header-title">🍅 Tomato Plant Disease Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader-title">Compare predictions from EfficientNetB0 and MobileNetV2 models for enhanced insights!</div>',
    unsafe_allow_html=True,
)

# Sidebar content
with st.sidebar:
    st.image("https://img.freepik.com/premium-photo/tomato-with-water-droplets-it-leaf-stem_927923-682.jpg")
    st.write(
        """
        **How it works:**
        1. Upload an image of a tomato plant leaf.
        2. Choose a model to classify the disease.
        3. Compare predictions to gain insights.
        """
    )
    st.write("### Models Available:")
    st.write("- **EfficientNetB0**")
    st.write("- **MobileNetV2**")

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload an image of a tomato leaf:",
    type=["jpg", "jpeg", "png"],
    help="Only JPG, JPEG, or PNG images are supported.",
)

if uploaded_file is not None:
    try:
        # Display uploaded image at smaller size
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)  # Reduced size

        # Dropdown to select models
        model_option = st.selectbox(
            "Select a model to predict:",
            ["EfficientNetB0", "MobileNetV2", "Compare Both"],
        )

        # Prediction logic
        if model_option == "EfficientNetB0":
            with st.spinner("Classifying with EfficientNetB0..."):
                pred_class, confidence = predict(image, efficientnet_model)
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            st.success("Prediction Complete!")
            st.write(f"### **EfficientNetB0 Prediction:** {pred_class}")
            st.write(f"### **Confidence:** {confidence:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        elif model_option == "MobileNetV2":
            with st.spinner("Classifying with MobileNetV2..."):
                pred_class, confidence = predict(image, mobilenet_model)
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            st.success("Prediction Complete!")
            st.write(f"### **MobileNetV2 Prediction:** {pred_class}")
            st.write(f"### **Confidence:** {confidence:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        elif model_option == "Compare Both":
            with st.spinner("Classifying with both models..."):
                eff_pred_class, eff_confidence = predict(image, efficientnet_model)
                mob_pred_class, mob_confidence = predict(image, mobilenet_model)
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            st.success("Comparison Complete!")
            st.write(f"- **EfficientNetB0 Prediction:** {eff_pred_class} ({eff_confidence:.2f}%)")
            st.write(f"- **MobileNetV2 Prediction:** {mob_pred_class} ({mob_confidence:.2f}%)")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.info("📤 Please upload an image to get started.")
