import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
NUM_SAMPLES = 1000
IMG_HEIGHT = 64
IMG_WIDTH = 256
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.txt")

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf"
FONT_PATH = "IndieFlower-Regular.ttf"

# --- Helper Functions ---
def generate_random_equation():
    """Generates a simple random equation string."""
    num1 = random.randint(0, 99)
    num2 = random.randint(0, 99)
    operator = random.choice(["+", "-", "*", "/"])
    return f"{num1}{operator}{num2}"

def download_font():
    """Downloads the handwriting font if it doesn't exist."""
    if not os.path.exists(FONT_PATH):
        print("Downloading handwriting font...")
        try:
            response = requests.get(FONT_URL)
            response.raise_for_status()
            with open(FONT_PATH, "wb") as f:
                f.write(response.content)
            print("Font downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading font: {e}")
            print("Please download the font manually and place it as 'IndieFlower-Regular.ttf'")
            return False
    return True

# --- Main Data Generation ---
def generate_dataset():
    """Creates the synthetic dataset."""
    if not download_font():
        return

    print("Generating synthetic dataset...")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    font = ImageFont.truetype(FONT_PATH, 40)
    labels = []

    for i in range(NUM_SAMPLES):
        equation = generate_random_equation()
        image = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)  # White background
        draw = ImageDraw.Draw(image)

        # Get text size
        try:
            text_bbox = draw.textbbox((0, 0), equation, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # For older Pillow versions
            text_width, text_height = draw.textsize(equation, font=font)


        # Position the text with some randomness
        x = (IMG_WIDTH - text_width) / 2 + random.randint(-10, 10)
        y = (IMG_HEIGHT - text_height) / 2 + random.randint(-5, 5)

        draw.text((x, y), equation, font=font, fill=0)  # Black text

        # Save the image and label
        img_path = os.path.join(IMAGES_DIR, f"img_{i}.png")
        image.save(img_path)
        labels.append(f"images/img_{i}.png,{equation}")

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} samples...")

    # Save the labels file
    with open(LABELS_FILE, "w") as f:
        f.write("\n".join(labels))

    print(f"\nDataset generation complete!")
    print(f"Images saved in: {IMAGES_DIR}")
    print(f"Labels saved in: {LABELS_FILE}")

if __name__ == "__main__":
    generate_dataset()
