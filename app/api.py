import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
from fastapi import FastAPI, Response,File
from service import inference_modified_debugging
from archer import archer
import os
import datetime
import random

from fastapi.staticfiles import StaticFiles

if not os.path.exists('./static'):
    print(
        '\n\nstatic does not exist. creating one\n\n'
    )
    os.mkdir('static')

if not os.path.exists('./static_ac'):
    print(
        '\n\nstatic does not exist. creating one\n\n'
    )
    os.mkdir('static_ac')

app = FastAPI()
templates = Jinja2Templates(directory="templates") 

holder = None



def generate_name():
    # Get the current date in YYYYMMDD format
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Generate a random number (e.g., between 1000 and 9999)
    random_number = random.randint(1000, 9999)
    
    # Combine date and random number
    generated_name = f"{date_str}_{random_number}"
    
    return generated_name


# static
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static_ac", StaticFiles(directory="static_ac"), name="static_ac")

# circularnet | js css
app.mount("/circularnet/css", StaticFiles(directory="templates/circularnet/css"), name="cn-css")
app.mount("/circularnet/js", StaticFiles(directory="templates/circularnet/js"), name="cn-js")

# home | js css
app.mount("/home/css", StaticFiles(directory="templates/home/css"), name="home-css")
app.mount("/home/js", StaticFiles(directory="templates/home/js"), name="home-js")

app.mount("/archer/css", StaticFiles(directory="templates/archer/css"), name="archer-css")
app.mount("/archer/js", StaticFiles(directory="templates/archer/js"), name="archer-js")


app.mount("/sources", StaticFiles(directory="sources"), name="sources")

@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("/home/home.html", {"request": {}})

@app.get("/circularnet/", response_class=HTMLResponse)
async def circularnet():
    return templates.TemplateResponse("/circularnet/circularnet.html", {"request": {}})

@app.get("/archer/", response_class=HTMLResponse)
async def circularnet():
    return templates.TemplateResponse("/archer/archer.html", {"request": {}})


# circularnet
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg','.PNG')):
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

        # inference
        cn_final_result, image_np_cp = inference_modified_debugging.main.inference(np_img=image_array, is_display=True)

        if type(cn_final_result) == type(None):
            image_np_cp = image_array

        holder = generate_name()
        print('\nfinal result is \n')
        print(cn_final_result)

        image_np_cp = np.squeeze(image_np_cp)
        image_pil = Image.fromarray(image_np_cp)


        # Save the final image
        final_image_path = f"static/{holder}.png"
        image_pil.save(final_image_path)

        return {"filename": final_image_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")



# archer
@app.post("/upload_ac/")
async def upload_img(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg','.PNG', '.JPG', '.JPEG')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Remove background
        image = await file.read()
        image = Image.open(io.BytesIO(image))
        image_array = np.array(image)  # Transform into np array
        print('image opened\n')

        # inference
        image_np, ac_final_result = archer.inference(image_array)

        holder = generate_name()
        print('\nfinal result is \n')
        print(ac_final_result)

        image_np_cp = np.squeeze(image_np)
        image_pil = Image.fromarray(image_np_cp)


        # Save the final image
        final_image_path = f"static_ac/{holder}.png"
        image_pil.save(final_image_path)

        return {"filename": final_image_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


