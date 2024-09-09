import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.image("https://iili.io/dj5iBHB.jpg", width=700)
st.markdown(
    """
    <style>
    .caption {
        text-align: center;
        font-size: 12px;
        color: white;
        margin-top: 10px;
    }
    </style>
    <div class="caption">The Team Behind This Project with our inspiring guide</div>
    """,
    unsafe_allow_html=True
)

model = keras.models.load_model('./model6.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

class_names = ['0.Healthy', '1.Anthracnose', '2.Phytophthora Blight', '3.Brown Spot', '4.Black Spot', '5.Others']

# Streamlit UI
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .center-button {
        display: flex;
        justify-content: center;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 20px;
    }
    </style>
    <div class="title">Papaya Fruit Multiple Disease Classification using Deep Learning Techniques</div>
    """,
    unsafe_allow_html=True
)
st.write("This app can predict the disease present in the papaya fruit")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Convert the image to RGB
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    st.image(opencv_image, channels="RGB")

    # Preprocess the image
    img = cv2.resize(opencv_image, (150, 150))
    img = np.expand_dims(img, axis=0).astype('float32')  # Convert to float32
    img = test_datagen.standardize(img)
    
    # Make predictions
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display the prediction
    st.write(f"Predicted Disease: {predicted_class}")

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by <a href="https://www.linkedin.com/in/jhajibhaskar/" target="_blank">CS20B1060</a> , <a href="https://www.linkedin.com/in/abhishektirkey/" target="_blank">CS20B1002</a> , <a href="https://www.linkedin.com/in/vivek140902/" target="_blank">CS20B1065</a>, Empowering agricultural advancements with deep learning!
    </div>
    """,
    unsafe_allow_html=True
)
