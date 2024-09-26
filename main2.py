import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model_path = r"E:\Project_2024\All Python Projects\Deep Learning Project\Streamlit_demo\Models01\3\Models1"
model = tf.keras.models.load_model(model_path)

# Class names (make sure this matches your dataset class names)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize to match the model's expected input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize the image
    return img_array

# Streamlit app layout
st.title("Potato Disease Classification")
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
        This dataset consists of about 2152 images of healthy and diseased crop leaves which is categorized into 3 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
        #### Content
        1. train (1728 images)
        2. test (256 images)
        3. validation (192 images)
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=(256, 256))
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            # Preprocess and predict
            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)

            # Display the result
            st.success(f"Prediction: {predicted_class} with {confidence}% confidence.")

