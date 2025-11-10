import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape,
    Bidirectional, LSTM, Dense
)
from tensorflow.keras.models import Model

# Define the character set
# The extra character is for the CTC blank token
char_list = "0123456789+-*/"

def get_crnn_model(img_height: int, img_width: int, num_classes: int) -> Model:
    """
    Builds the CRNN model.
    """
    input_img = Input(shape=(img_height, img_width, 1), name="image", dtype="float32")

    # CNN part
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = MaxPooling2D((1, 2))(x)
    x = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = MaxPooling2D((1, 2))(x)

    # Reshape for RNN
    # The output of the CNN is (batch_size, height, width, channels)
    # We want to treat the width dimension as the time-step dimension for the RNN
    shape = x.shape
    x = Reshape((shape[1] * shape[2], shape[3]))(x) # (batch_size, width * height, channels)
    x = Dense(64, activation="relu")(x)


    # RNN part
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)

    # Output layer
    output = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=input_img, outputs=output)

    return model

def ctc_decode(y_pred):
    """
    Decodes the output of the model.
    """
    input_len = tf.ones(tf.shape(y_pred)[0]) * tf.shape(y_pred)[1]
    # Use greedy search. For complex tasks, beam search would be better.
    results = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0]
    
    # Convert back to string
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Helper functions for character mapping
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=list(char_list), mask_token=None
)
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


if __name__ == '__main__':
    # Get the model
    img_height = 64
    img_width = 256
    num_classes = len(char_list) + 1 # +1 for the CTC blank character
    model = get_crnn_model(img_height, img_width, num_classes)
    model.summary()

    # You would typically compile the model with a CTC loss function like this:
    # model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred) # Loss is handled by the training process
