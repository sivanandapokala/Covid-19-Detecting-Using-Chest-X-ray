from PIL import Image
import streamlit as st
import pandas as pd
from tensorflow import keras
import numpy as np
import cv2
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st

model = keras.models.load_model('model6.h5')
def preprocess(image):
    print((image))
    images=[]
    images.append(cv2.resize(image,(224,224))/255)
    imag=np.array(images)
    print(imag.shape)
    
    
    result = model.predict(imag)
    if np.argmax(result,axis=1)==0:
        output= 'Covid'
    elif np.argmax(result,axis=1)==1:
        output= 'Normal'
    else:
        output = 'Viral Pneumonia'
    s='predicted as {} with probability : {}'.format(output,np.max(result,axis=1))
    return s



html_temp = """ 
    <div id=1 style ="background-color:orange;padding:10px"> 
    <h1 style ="color:blue;text-align:left;">Covid Detection Using Chest X-ray</h1> 
    </div>
    
    <style>
    body {
    background-image: url("https://news.usc.edu/files/2018/03/Nanoparticle-cancer-detection-web.jpg");
    background-size: cover;
    }
    </style>
    """

html1 = """<h2 style ="color:blue;text-align:right;"> - OM AND SIVA</h2> """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.markdown(html1, unsafe_allow_html = True)
st.markdown("""<h2 style ="color:orange;text-align:left;">Please Choose the X-ray to Classify</h2> """, unsafe_allow_html = True)
      
    



img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    image=np.array(image)   
    pred=preprocess(image)
    if st.button("Predict"):
        st.success(pred)
    

   
