import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(layout = "wide")
st.title("Facial Emotion Detector:")

st.write("""
## What Facial Emotion is depicted?
""")


# load pretrained model
# Source: https://docs.streamlit.io/library/advanced-features/caching
model = tf.keras.models.load_model('expression.model')


# Function to preprocess image and retrieve prediction from model
def get_pred(model, image):

    label_dict = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    open_image = Image.open(image)
    resized_image = open_image.resize((48,48)).convert('L')  #Resize and convert to grayscale
    np_image = img_to_array(resized_image)
    reshaped = np.expand_dims(np_image, axis=0)   #expands image shape (1,48,48)
    reshaped = reshaped * (1./255)  #Normalize image

    pred_prob = model.predict(reshaped)
    pred_prob = list(pred_prob[0])
    img_index = np.argmax(pred_prob)
    result = label_dict[img_index]
    confidence = pred_prob[img_index]
    return f"{result}, Confidence {confidence: .2f}"


# upload a file
# code source: https://stephenallwright.com/streamlit-upload-file/
uploaded_file = st.file_uploader('Upload your own image here:', type = ['jpg', 'jpeg'])
submitted = st.button('Submit')

# When image is submitted, display image and run inference
if uploaded_file is not None and submitted:        
        st.image(uploaded_file)
        st.text(get_pred(model, uploaded_file))

