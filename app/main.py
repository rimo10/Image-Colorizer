import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app import get_prediction
from PIL import Image
from io import BytesIO
import io
import numpy as np
import matplotlib.pyplot as plt
import base64
from fastapi.responses import JSONResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static",StaticFiles(directory="static"),name="static")


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('index.html', context)


def read_image(img_encoded):
    pil_img = Image.open(BytesIO(img_encoded))
    return (pil_img)


@app.post('/upload', response_class=HTMLResponse)
async def pred_img(request: Request, image: UploadFile = File(...), colorization_type: str = Form(...)):
    img_dec = (read_image(image.file.read()))
    out = get_prediction(img_dec, colorization_type)
    img_bytes = io.BytesIO()
    plt.imsave(img_bytes, out, format='png')
    img_bytes.seek(0)
    # encode image bytes as base64
    encoded_image = base64.b64encode(img_bytes.read()).decode('utf-8')
    context = {'image': encoded_image}
    return JSONResponse(content=context)


if __name__ == "__main__":
    uvicorn.run(app)
