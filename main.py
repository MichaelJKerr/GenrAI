import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('model/GAI_Model2.h5')
class_names = ["Classical", "Country" ,"Death Metal","Doom Metal","Drum and Bass","Electronic","Grime","Heavy Metal","Hip-Hop","Lo-fi","Pop","Punk","Reggae","Soul"]

st.image("Assets/logo.jpg")
file = st.file_uploader("", type=["jpg"])

def makePrediction(image_data, model):

    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    accuracy = 100 * np.max(score)

    st.write("Image is  ", class_names[np.argmax(score)], " with ", "%.2f" % accuracy, "% Confidence")
    st.image(image, use_column_width=True)

if file is None:
    st.write("Welcome to GenrAI")
else:
    image = Image.open(file)
    prediction = makePrediction(image, model)
