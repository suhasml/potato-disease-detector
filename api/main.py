from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import requests
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

endpoint = "http://localhost:8601/v1/models/model:predict"

MODEL = tf.keras.models.load_model("D:\\projects\\deep_learning\\potato-disease-identification\\models\\1")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "Hello World"

def read_file(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    image = np.array(image)
    return image
    # image = np.array(Image.open(BytesIO(data)))
    # return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint,json=json_data)
    prediction = np.array(response.json()['predictions'][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]
    return {
        "class": predicted_class,
        "confidence": confidence
    }
    



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)