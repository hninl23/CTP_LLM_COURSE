import torch
import streamlit as st
from transformers import pipeline


image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


#music lyrics tranlator 
st.title("Extract text from Image!")

image = st.file_uploader(label= "Drop your Image!")
if image is not None:
    image_data = image.getvalue()
    with open(image.name, "wb") as file:
            file.write(image_data)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    result = image_to_text(image.name)              
txt = st.text_area(
    "Your Results will be here",
    result[0]['generated_text']
    )