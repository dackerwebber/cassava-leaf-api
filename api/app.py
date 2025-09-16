import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# from utils import predict_label
from PIL import Image
import numpy as np

st.title("Cassava Leaf Disease Classification")

st.write("Predict the leaf disease that is being represented in the image.")

model = load_model("model.h5")

l=['Cassava Bacterial Blight (CBB)','Cassava Brown Streak Disease (CBSD)','Cassava Green Mottle (CGM)','Cassava Mosaic Disease (CMD)','Healthy']


uploaded_file = st.file_uploader(
    "Upload an image of a cassava leaf:", type="jpg"
)
prediction=-1
if uploaded_file is not None:
    img_image = Image.open(uploaded_file)
    img_image = img_image.resize((224, 224))
    img_image = np.array(img_image) / 255.0
    img_image = np.expand_dims(img_image, axis=0)
    prediction = model.predict(img_image)
    predicted_label = np.argmax(prediction)
    label=l[predicted_label]

st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Predicted Diesease : {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")

st.write("If you would not like to upload an image, you can use the sample image instead:")

sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    img_image = Image.open("sample_img.jpg")
    img_image = img_image.resize((224, 224))
    img_image = np.array(img_image) / 255.0
    img_image = np.expand_dims(img_image, axis=0)
    prediction = model.predict(img_image)
    predicted_label = np.argmax(prediction)
    label=l[predicted_label]
    
    st.image(img_image, caption="Sample Image", use_column_width=True)
    st.markdown(
        f"<h2 style='text-align: center;'>Predicted Diesease :{label}</h2>",
        unsafe_allow_html=True,
    )
