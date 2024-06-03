from typing import Union
from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import boto3
import uvicorn
from face_detect import ImageBlur
import cv2
from fastapi.responses import JSONResponse
import numpy as np
import uuid
import urllib

class Image(BaseModel):
    query: str
    target_lang: str

@asynccontextmanager
async def lifespan(app:FastAPI):
    global llm
    
    yield

app = FastAPI(lifespan=lifespan)

filter = ImageBlur()

load_dotenv()

s3 = boto3.client("s3", aws_access_key_id=os.getenv('S3_ACCESS_KEY'), aws_secret_access_key=os.getenv('S3_SECRET_KEY'))

@app.get("/")
async def initiate():
    return

@app.post("/filter")
async def filter_image(file: UploadFile):

    content = await file.read()

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    filename = f'{uuid.uuid4()}.jpg'
    cv2.imwrite(filename, img)
    filtered_img = filter.blur(filename)
    os.remove(filename)
    filtered_filename = f'{uuid.uuid4()}.jpg'
    cv2.imwrite(filtered_filename, filtered_img)

    try:
        s3.upload_file(f'{filtered_filename}', os.getenv('S3_BUCKET_NAME'), filtered_filename, ExtraArgs={'ContentType': 'image/jpeg'})
    except:
        os.remove(filtered_filename)
        return JSONResponse(status_code=500, content={'msg': 'failed to upload image'})
    
    url = "https://%s/%s" % (
        os.getenv('CLOUDFRONT_DOMAIN'),
        urllib.parse.quote(filtered_filename, safe="~()*!.'")
    )


    os.remove(filtered_filename)
    return JSONResponse(status_code=200, content={'msg': 'success', 'url': url})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)