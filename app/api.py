import numpy as np 
from PIL import Image
import io 
from typing import Dict, Any
from pydantic import BaseModel, RootModel
from fastapi import FastAPI,UploadFile,Response,HTTPException
from fastapi.middleware.cors import CORSMiddleware # NOTE: fastapi middleware
from service import inference_modified_debugging

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

stored_image_np_cp = None

class PredictionResult(BaseModel):
    score: float
    box: list
    class_name: str

class PredictionsResponse(BaseModel):
    predictions: dict[int, PredictionResult]

@app.get("/", tags = ["root"])
async def read_root() -> dict:
    return {"message":"hello world"}


@app.post('/get_predictions')
async def get_predictions(file: UploadFile):
    global stored_image_np_cp  # Declare as global to modify the variable
    
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    image_array = np.array(image)  # Transform into np array
    
    if len(image_array.shape) == 2:  # Grayscale image
        image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    if len(image_array.shape) == 3:  # 3D but no batch dimension
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    final_result, image_np_cp = inference_modified_debugging.main.inference(np_img=image_array, is_display=True)
    
    # Store the image globally for later retrieval
    stored_image_np_cp = image_np_cp
    
    print(final_result)

    if final_result is not None:
        try:
            # Convert final_result to integer keys and validate values
            formatted_result = {}
            for k, v in final_result.items():
                # Ensure the key is an integer
                int_key = int(k)
                # Ensure the value is a valid dictionary
                if isinstance(v, dict) and all(key in v for key in ['score', 'box', 'class_name']):
                    formatted_result[int_key] = PredictionResult(
                        score=v['score'],
                        box=v['box'],
                        class_name=v['class_name']
                    )
                else:
                    raise ValueError(f"Invalid value format for key {k}: {v}")

            response = PredictionsResponse(predictions=formatted_result)
        except Exception as e:
            print("Error processing final_result:", e)
            response = {'output': f'Error processing final_result: {str(e)}'}
    else:
        response = {'output': 'None'}

    return response

@app.post('/output_image')
async def output_image():
    global stored_image_np_cp  

    if stored_image_np_cp is None:
        raise HTTPException(status_code=404, detail="No image found. Please run '/get_predictions' first.")

    # Remove the batch dimension and convert the NumPy array back to an image
    image_np_cp = np.squeeze(stored_image_np_cp)
    image_pil = Image.fromarray(image_np_cp)

    # Convert the image to bytes
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Return the image in the response
    return Response(content=img_byte_arr, media_type="image/png")


# for testing 
# uvicorn api:app --reload