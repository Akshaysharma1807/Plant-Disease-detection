from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import cv2

model=tf.keras.models.load_model("../models/potatoes")
class_name = ["Early Blight", "Late Blight", "Healthy"]
def file_image(bytes)->np.ndarray:

    return np.array(Image.open(BytesIO(bytes)))

app = FastAPI()

@app.get("/ping")
async def ping():
    return "hello!!"

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    img=file_image(await file.read())
    img=cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img=np.expand_dims(img, 0)
    prediction=model.predict(img)
    output=class_name[np.argmax(prediction[0])]
    return {
        'class':output,
        'confidence':float(np.max(prediction[0]))
    }

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)