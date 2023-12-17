# Disabling GPU
# Currently my laptop has buggy version of CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# README
# Hello everyone, in here I (Kaenova | Bangkit Mentor ML-20) 
# will give you some headstart on createing ML API. 
# Please read every lines and comments carefully. 
# 
# I give you a headstart on text based input and image based input API. 
# To run this server, don't forget to install all the libraries in the
# requirements.txt simply by "pip install -r requirements.txt" 
# and then use "python main.py" to run it
# 
# For ML:
# Please prepare your model either in .h5 or saved model format.
# Put your model in the same folder as this main.py file.
# You will load your model down the line into this code. 
# There are 2 option I give you, either your model image based input 
# or text based input. You need to finish functions "def predict_text" or "def predict_image"
# 
# For CC:
# You can check the endpoint that ML being used, eiter it's /predict_text or 
# /predict_image. For /predict_text you need a JSON {"text": "your text"},
# and for /predict_image you need to send an multipart-form with a "uploaded_file" 
# field. you can see this api documentation when running this server and go into /docs
# I also prepared the Dockerfile so you can easily modify and create a container iamge
# The default port is 8080, but you can inject PORT environement variable.
# 
# If you want to have consultation with me
# just chat me through Discord (kaenova#2859) and arrange the consultation time
#
# Share your capstone application with me! 🥳
# Instagram @kaenovama
# Twitter @kaenovama
# LinkedIn /in/kaenova

## Start your code here! ##

import os
import uvicorn
import traceback
import tensorflow as tf

# This is what i add to implement the model
import tensorflow_text
import numpy as np

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array, resize_and_convert_to_bw

from PIL import Image

# Initialize Model
# If you already put yout model in the same folder as this main.py
# You can load .h5 model or any model below this line

# If you use h5 type uncomment line below
vision_model = tf.keras.models.load_model('./vision.h5')
# If you use saved model type uncomment line below
text_model = tf.saved_model.load("./text")

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# If your model need text input use this endpoint!
class RequestText(BaseModel):
    text:str

@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # In here you will get text sent by the user
        text = req.text
        print("Uploaded text:", text)
        
        model_text_out = text_model([text])
        model_text_out = model_text_out.numpy().flatten()
        model_text_out = np.round(model_text_out).tolist()
        
        return model_text_out[0]
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

# If your model need image input use this endpoint!
@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        # Checking if it's an image
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"
        
        # In here you will get a numpy array in "image" variable.
        # You can use this file, to load and do processing
        # later down the line
        image = load_image_into_numpy_array(uploaded_file.file.read())
        print("Image shape:", image.shape)
        
        # Step 1: (Optional, but you should have one) Do your image preprocessing
        image = resize_and_convert_to_bw(image, (28,28)) # this is from parameter we created
        
        print("processed shape:", image.shape)
        Image.fromarray(image.squeeze(-1)).save("test.png")
        
        # Step 2: Prepare your data to your model
        image = np.expand_dims(image, 0) # adding batch dimension
        
        # Step 3: Predict the data
        result = vision_model(image).numpy().flatten()
        
        # Step 4: Change the result your determined API output
        idx_res = np.argmax(result).tolist()
        
        return idx_res
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)