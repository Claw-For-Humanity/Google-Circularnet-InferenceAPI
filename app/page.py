import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import aiohttp
from PIL import Image
import numpy as np
from fastapi import FastAPI, Response,File
# from service import inference_modified_debugging
import os

from fastapi.staticfiles import StaticFiles

app = FastAPI()



app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})



@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Remove background
        image = await file.read()
        image = Image.open(io.BytesIO(image))
        image_array = np.array(image)  # Transform into np array

        if len(image_array.shape) == 2:  # Grayscale image
            image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
        if len(image_array.shape) == 3:  # 3D but no batch dimension
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # final_result, image_np_cp = inference_modified_debugging.main.inference(np_img=image_array, is_display=True)

        print('\nfinal result is \n')
        # print(final_result)

        image_np_cp = np.squeeze(image_np_cp)
        image_pil = Image.fromarray(image_np_cp)


        # Save the final image
        # final_image_path = "static/final_image.png"
        # image_pil.save(final_image_path)

        return {"filename": final_image_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

