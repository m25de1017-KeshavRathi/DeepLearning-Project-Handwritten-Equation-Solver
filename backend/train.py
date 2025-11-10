import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import get_crnn_model, char_list, char_to_num
from utils import preprocess_image

# --- Configuration ---
DATASET_DIR = "dataset"
LABELS_FILE = os.path.join(DATASET_DIR, "labels.txt")
IMG_HEIGHT = 64
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# --- 1. Load Data ---
def load_dataset():
    """Loads image paths and labels from the labels file."""
    with open(LABELS_FILE, "r") as f:
        lines = f.read().splitlines()
    
    image_paths, labels = [], []
    for line in lines:
        path, label = line.split(",", 1)
        image_paths.append(os.path.join(DATASET_DIR, path))
        labels.append(label)
        
    return image_paths, labels

# --- 2. Create tf.data.Dataset ---
def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # 5. Transpose the image because we want the time dimension to be the width
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in the label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a tuple as our model expects (input, target)
    return {"image": img}, label

def create_data_pipeline(image_paths, labels):
    """Creates a tf.data pipeline for training."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = (
        dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(
            BATCH_SIZE,
            padded_shapes=({"image": [IMG_WIDTH, IMG_HEIGHT, 1]}, [None]),
            padding_values=({"image": 0.0}, tf.constant(0, dtype=tf.int64)),
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset

# --- 3. CTC Loss ---
class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, name="ctc_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        
        # Calculate the actual length of each label before padding
        label_length = tf.math.count_nonzero(y_true, axis=1, dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        loss = tf.keras.backend.ctc_batch_cost(
            tf.cast(y_true, tf.int32), y_pred, input_length, tf.expand_dims(label_length, axis=-1)
        )
        return loss

# --- 4. Training ---
def train():
    """Main training function."""
    print("1. Loading data...")
    image_paths, labels = load_dataset()
    
    # Add a check for max label length
    max_label_len = max([len(label) for label in labels])
    print(f"Max label length: {max_label_len}")

    print("\n2. Splitting data...")
    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42
    )
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

    print("\n3. Creating data pipelines...")
    train_dataset = create_data_pipeline(x_train, y_train)
    validation_dataset = create_data_pipeline(x_val, y_val)

    print("\n4. Building and compiling model...")
    model = get_crnn_model(IMG_WIDTH, IMG_HEIGHT, len(char_list) + 2) # Note the swapped dimensions
    
    # The model's output shape is (batch, width, classes), which is what CTC loss expects.
    # The image is transposed in the data pipeline to (width, height, 1).
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=CTCLoss(),
    )
    model.summary()

    print("\n5. Setting up callbacks...")
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/model_checkpoint.h5"
    
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )

    print("\n6. Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
    )

    print("\nTraining complete!")
    print(f"Best model saved at: {checkpoint_path}")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print("Dataset not found. Please run 'python data_generator.py' first.")
    else:
        train()
