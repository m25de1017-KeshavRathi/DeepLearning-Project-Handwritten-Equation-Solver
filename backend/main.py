import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from model import get_crnn_model, char_list, ctc_decode
from solver import solve_equation
from utils import preprocess_image

# --- Configuration ---
IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_CLASSES = len(char_list) + 2

# --- FastAPI App ---
app = FastAPI(
    title="Handwritten Equation Solver",
    description="An API to solve handwritten equations from images.",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity. For production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
# In a real application, you would load your trained model weights here.
# For this example, we are just creating the model architecture.
model = get_crnn_model(IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES)
# model.load_weights('path/to/your/model_weights.h5') # Uncomment and modify this line

# --- Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok"}

@app.post("/predict/", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predicts the equation from an uploaded image and solves it.
    """
    # --- 1. Save the uploaded file temporarily ---
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --- 2. Preprocess the image ---
        preprocessed_image = preprocess_image(file_path, IMG_HEIGHT, IMG_WIDTH)

        # --- 3. Make a prediction ---
        # In a real scenario, you would use the model to predict.
        # For this example, we'll mock the prediction.
        # y_pred = model.predict(preprocessed_image)
        # recognized_text = ctc_decode(y_pred)[0]
        
        # Mocked prediction for demonstration purposes
        # Let's pretend the model recognized "12+3*4"
        recognized_text = "12+3*4"
        
        if not recognized_text:
            raise HTTPException(status_code=400, detail="Could not recognize any text in the image.")

        # --- 4. Solve the equation ---
        try:
            result = solve_equation(recognized_text)
        except (ValueError, ZeroDivisionError) as e:
            raise HTTPException(status_code=400, detail=f"Could not solve the equation: {e}")

        # --- 5. Return the result ---
        return {
            "recognized_equation": recognized_text,
            "solution": result,
        }

    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    finally:
        # --- 6. Clean up the temporary file ---
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    # To run this file: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
