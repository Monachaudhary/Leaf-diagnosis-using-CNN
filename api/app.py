import streamlit as st
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import streamlit.components.v1 as html
import io

st.set_page_config(page_title="Plant detection & prediction")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_option("deprecation.showfileUploaderEncoding", False)

with st.sidebar:
    choose = option_menu(
        "App Gallery",
        ["Home", "About", "Detection & prediction", "Contact"],
        icons=["house", "kanban", "camera fill", "person lines fill"],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )

if choose == "About":
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    new_title = '<p style="font-family:sans-serif; color:Green;padding-left:67px; font-size: 32px;">Ritians</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    image = Image.open('Manu.png')
    image2 = Image.open('Somrupa.png')
    st.image(image, output_format="PNG")
    st.image(image2, output_format="PNG")
elif choose == "Home":
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
   
    st.header("Welcome to Leaf diagnosis app")
    st.write("It is crucial for crop quality and productivity to accurately identify disease outbreaks in the early stages in order to choose the best treatments.  By creating an early detection system of disease, an automated system for disease detecting in crop will play a significant role in agriculture.")
    st.write("You can identify your crops on this platform and receive the finest solution. User can forecast the outcome based on detection of any leaves. ")
    def load_lottieurl(url: str):
      r=requests.get(url)
      if r.status_code != 200:
       return None
      return r.json()
    lottie_animation_1= "https://assets3.lottiefiles.com/private_files/lf30_8exlgvzr.json"
    lottie_anim_json= load_lottieurl(lottie_animation_1)
    st_lottie(lottie_anim_json, key="hello", height=200, width=600, ) 
elif choose == "Contact":
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 32px;">Contact Us</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write("Please use this contact details if you have any questions or concern about our apps")
    st.write("Email id: manuchaudhary991@gmail.com, somrupas018@gmail.com")
    st.write("You can also contact us")
    st.write("[Linkedin](https://www.linkedin.com/in/manu-chaudhary-210924200/)")
    st.write("[Github](https://www.linkedin.com/in/manu-chaudhary-210924200/)")
elif choose == "Detection & prediction":
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    model = tf.keras.models.load_model(
        "C:/Users/manuc/plant-diseases-detection/models/my_model.h5"
    )
    st.write(
        """
         # Plant detection & prediction
         """
    )
    CLASS_NAMES = [
        "APPLE_ ROT_LEAVES",
        "APPLE_HEALTHY_LEAVES",
        "APPLE_LEAF_ BLOTCH",
        "APPLE_SCAB_LEAVES",
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy",
    ]
    Solution = [
        "One of the best ways to prevent black rot in the first place is to select and plant varieties that are resistant to it. Best gardening practices such as proper watering, sunlight, airflow, and fertilizer are still necessary for the optimal chances of preventing this disease.",
        "Leaf is healthy",
        "Treating trees with zinc-containing fungicides (e.g., Ziram) or foliar sprays containing zinc nutrients can decrease the severity of necrotic leaf blotch. Zinc oxide applied every 2 weeks from budbreak to harvest can diminish symptoms.",
        "Apply fungicides at 7 to 10 day intervals, starting at the time of bud break. Remove any fallen leaves or infected fruit from the ground",
        "Seed treatment with hot water, soaking seeds for 30 minutes in water pre-heated to 125 F/51 C, is effective in reducing bacterial populations on the surface and inside the seeds.",
        "Leaf is healthy",
        "Remove infected leaves and use copper fungicides. Apply mulch to keep the soil moisture consistent.",
        "Remove infected leaves and use copper fungicides. Apply mulch to keep the soil moisture consistent",
        "Leaf is healthy",
        "Remove the infected plant tissue and apply copper fungicides",
        "Remove infected leaves and use copper fungicides. Apply mulch to keep the soil moisture consistent.",
        "Remove infected leaves and use copper fungicides. Apply mulch to keep the soil moisture consistent",
        "Remove infected leaves and use copper fungicides. Water the plants from the base rather than from above",
        "Remove infected leaves and use copper fungicides. Avoid getting water on the leaves when watering the plants",
        "Prune infected areas and remove infected leaves, fruits and twigs. Use copper fungicides.",
        "Remove the infected plant tissue and apply copper fungicides",
        "Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. ",
        "Leaf is healthy",
    ]
    img_file_buffer = st.file_uploader("Upload a image", type=["png", "jpg", "jpeg"])

    @st.cache_data
    def load_image(image):
        img = Image.open(image)
        return img

    def import_and_predict(image_data, model):
        img_batch = np.expand_dims(image_data, 0)
        predictions = model.predict(img_batch)
        return predictions

    if img_file_buffer is not None:
        st.image(load_image(img_file_buffer))
        bytes_data = img_file_buffer.getvalue()
        image = np.array(Image.open(BytesIO(bytes_data)))
        prediction = import_and_predict(image, model)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        accuracy = confidence * 100
        if CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[0]:
            z = Solution[0]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[1]:
            z = Solution[1]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[2]:
            z = Solution[2]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[3]:
            z = Solution[3]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[4]:
            z = Solution[4]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[5]:
            z = Solution[5]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[6]:
            z = Solution[6]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[7]:
            z = Solution[7]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[8]:
            z = Solution[8]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[9]:
            z = Solution[9]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[10]:
            z = Solution[10]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[11]:
            z = Solution[11]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[12]:
            z = Solution[12]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[13]:
            z = Solution[13]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[14]:
            z = Solution[14]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[15]:
            z = Solution[15]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[16]:
            z = Solution[16]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[17]:
            z = Solution[17]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[18]:
            z = Solution[18]
        elif CLASS_NAMES[np.argmax(prediction[0])] == CLASS_NAMES[19]:
            z = Solution[19]
        X = ["Result after diagnosis", "Accuracy"]
        y = [CLASS_NAMES[np.argmax(prediction[0])], accuracy, z]
        interactive = st.container()

        # reading data in table
        with interactive:
            fig = go.Figure(
                data=go.Table(
                    header=dict(
                        values=[
                            ["<b>Result after diagnosis</b>"],
                            ["<b>Accuracy</b>"],
                            ["<b>Solution</b>"],
                        ],
                        fill_color="slategrey",
                        line_color="darkslategray",
                        align="center",
                        font=dict(color="white", size=20),
                        height=50,
                        
                    ),
                    cells=dict(
                        values=y[0:3],
                        line_color="darkslategray",
                        fill=dict(color=["white", "white"]),
                        align="center",
                        font_size=15,
                        height=70,
                    ),
                )
            )

            fig.update_layout(width=600, height=500)
            st.write(fig)

    else:
        st.text("please provide image")
