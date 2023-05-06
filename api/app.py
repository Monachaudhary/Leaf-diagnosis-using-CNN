import streamlit as st
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np
st.set_page_config(
    page_title="Plant detection & prediction"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.set_option('deprecation.showfileUploaderEncoding', False)

model=tf.keras.models.load_model('C:/Users/manuc/plant-diseases-detection/models/my_model.h5')
    

st.write("""
         # Plant detection & prediction
         """
         )
CLASS_NAMES = ['APPLE_ ROT_LEAVES',
 'APPLE_HEALTHY_LEAVES',
 'APPLE_LEAF_ BLOTCH',
 'APPLE_SCAB_LEAVES',
 'Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

img_file_buffer = st.file_uploader('Upload a image', type= ["png", "jpg","jpeg"] )
def import_and_predict(image_data, model):
        img_batch = np.expand_dims(image_data, 0)
        predictions = model.predict(img_batch)
        return predictions
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    image = np.array(Image.open(BytesIO(bytes_data))) 
    prediction= import_and_predict(image, model)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    # string = "Prediction : " + CLASS_NAMES[np.argmax(prediction[0])]
    st.write(CLASS_NAMES[np.argmax(prediction[0])])

    # if CLASS_NAMES[np.argmax(prediction[0])] == 'Healthy':
    #     st.success(string)
        
    # else:
    #     st.warning(string)

else:
    st.text("please provide image")
