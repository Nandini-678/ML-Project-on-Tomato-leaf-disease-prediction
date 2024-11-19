import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# Set Streamlit page configuration
st.set_page_config(
    page_title="Tomato Plant leaf Disease Classification",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load both models with caching
@st.cache_resource
def load_efficientnet_model():
    return tf.keras.models.load_model("efficientnetb0_model.h5")

@st.cache_resource
def load_mobilenet_model(): 
    return tf.keras.models.load_model("mobilenetv2_model.h5")

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
    "Tomato___healthy"
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

# Add custom CSS for gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: rgb(33,0,29);
        background: linear-gradient(90deg, rgba(33,0,29,1) 0%, rgba(142,38,40,1) 51%, rgba(89,54,4,1) 100%);
        color: white;
    }
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
    .upload-section {
        border: 2px dashed white;
        padding: 20px;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .prediction-section {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .sidebar .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
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
    st.write("""
        **How it works:**
        1. Upload an image of a tomato plant leaf.
        2. Choose a model to classify the disease.
        3. Compare predictions to gain insights.
    """)
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
        image = Image.open(uploaded_file).convert('RGB')
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
