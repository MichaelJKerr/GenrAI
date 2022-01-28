import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('model/GAI_Model.h5')
class_names = ["Blues", "Classical", "Country" ,"Death Metal","Doom Metal","DrumNBass","Electronic","Folk","Grime","Heavy Metal","HipHop","Jazz","LoFi","Pop","Psychedelic Rock","Punk","Reggae","Rock","Soul", "Techno"]

st.write("""
         # Welcome To GenrAI
         """
         )
file = st.file_uploader("Please upload an image file", type=["jpg"])

def import_and_predict(image_data, model):

    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    Accuracy = 100 * np.max(score)
    st.write("Image is  ", class_names[np.argmax(score)], " with ", "%.2f" % Accuracy, "% Accuracy")
    st.image(image, use_column_width=True)

if file is None:
    st.image("Assets/logo.jpg")
else:
    image = Image.open(file)
    prediction = import_and_predict(image, model)
