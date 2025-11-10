import cv2
import numpy as np
from skimage.transform import resize

def preprocess_image(image_path: str, img_height: int, img_width: int) -> np.ndarray:
    """
    Preprocesses an image for the CRNN model.
    - Reads the image
    - Converts to grayscale
    - Resizes to model's expected input dimensions
    - Normalizes pixel values
    - Transposes for the CRNN model (width becomes the time dimension)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize the image
    img = resize(img, (img_height, img_width), anti_aliasing=True)

    # Normalize the image
    img = (img / 255.0).astype(np.float32)

    # Add a channel dimension and transpose
    img = np.expand_dims(img, axis=-1)
    img = img.transpose((1, 0, 2))

    # Add a batch dimension
    img = np.expand_dims(img, axis=0)

    return img

if __name__ == '__main__':
    # Create a dummy image for testing
    dummy_image = np.ones((100, 200), dtype=np.uint8) * 128
    cv2.imwrite("dummy.png", dummy_image)

    # Test the preprocessing function
    try:
        processed_image = preprocess_image("dummy.png", 64, 256)
        print(f"Processed image shape: {processed_image.shape}")
        assert processed_image.shape == (1, 256, 64, 1)
        print("Preprocessing test passed.")
    except Exception as e:
        print(f"Preprocessing test failed: {e}")

    # Clean up the dummy image
    import os
    os.remove("dummy.png")
