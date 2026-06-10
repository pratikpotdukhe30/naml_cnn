from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np


model = None

CLASSES = ["Early Blight", "Late Blight", "Healthy"]


BUCKET_NAME = "naml_cnn"

def download_file(bucket_name, source_blob_file, destination_file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_file)

    blob.download_to_filename(destination_file_path)
    print(f"Blob {source_blob_file} downloaded to {destination_file_path}")


def predict(request):
    global model
    if model is None:
        download_file(
            BUCKET_NAME,
            "models/potatoesV1.h5",
            "/tmp/potatoes.h5"
        )

        model = tf.keras.models.load_model("/tmp/potatoes.h5")


    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    #image = image / 255  # normalize the image in 0 to 1 range
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    print(f"Predictions: {predictions}")

    return {"class": predicted_class, "confidence": confidence}

