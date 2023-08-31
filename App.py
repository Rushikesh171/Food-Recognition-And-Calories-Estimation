import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os

model = load_model('FV.h5')

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class)
    res = labels[y]
    return res.capitalize()


def run():
    st.markdown('<link href="style.css" rel="stylesheet">', unsafe_allow_html=True)

    st.title("CalorieMart")
    st.write("Upload an image of a fruit or vegetable for classification.")

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        save_image_dir = './upload_images'
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)

        save_image_path = os.path.join(save_image_dir, img_file.name)

        try:
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())

            st.success("Image saved successfully.")

            if st.button("Predict"):
                st.text("Predicting...")
                result = prepare_image(save_image_path)
                display_prediction_result(result)

                cal = fetch_calories(result)
                if cal:
                    st.warning('**Calories (100 grams): ' + cal + '**')
                else:
                    st.warning('**Calories information not available**')

        except Exception as e:
            st.error("Error saving image:")
            st.error(e)


def display_prediction_result(prediction):
    if prediction in vegetables:
        st.info('**Category: Vegetables**')
    else:
        st.info('**Category: Fruit**')
    st.success("**Predicted: " + prediction + '**')


run()
