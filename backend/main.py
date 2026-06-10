from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

model = tf.keras.models.load_model("../models/model.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


origins = [
    "http://localhost",
    "http://localhost:3000",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"ping": "pong"}


def read_file_as_image(image):
    return np.array(Image.open(BytesIO(image)))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    img = tf.expand_dims(image, 0)
    predictions = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)