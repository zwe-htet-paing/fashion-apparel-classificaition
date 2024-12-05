import os
import json
import io
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model

import gradio as gr

import warnings
warnings.filterwarnings('ignore')

class Predictor:
    """
    Class to load a trained model and make predictions on new data.
    """
    def __init__(self, model_path, class_mapping_path, input_size):
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.input_size = input_size
        self.model = self._load_model()
        self.classes = self._load_classes()
        

    def _load_model(self):
        """
        Load the trained model from the given path.
        """
        return load_model(self.model_path)

    def _load_classes(self):
        """
        Load the class mapping from the given path.
        """
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            class_mapping = {int(k): v for k, v in json.load(f).items()}

        return class_mapping

    def preprocess_image(self, image):
        """
        Preprocess an image for prediction. Handles both file paths and byte images.

        Args:
            image: Image input (file path or byte stream).
        Returns:
            Preprocessed image array.
        """
        if isinstance(image, str):  # File path
            pil_image = load_img(image, target_size=(self.input_size, self.input_size))
        elif isinstance(image, bytes):  # Byte stream
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
            pil_image = pil_image.resize((self.input_size, self.input_size))
        else:
            raise ValueError("Invalid image input type. Must be a file path or bytes.")

        image_array = img_to_array(pil_image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return preprocess_input(image_array)

    def predict(self, image):
        """
        Make a prediction for an image. Handles both file paths and byte images.

        Args:
            image: Image input (file path or byte stream).
        Returns:
            Predicted category.
        """
        preprocessed_image = self.preprocess_image(image)
        predictions = self.model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_category = self.classes[predicted_class_index]
        return predicted_category


def launch_gradio_interface(predictor):
    """
    Launch a Gradio interface for the given predictor.

    Args:
        predictor (Predictor): An instance of the Predictor class.
    """
    def gradio_inference(image):
        """
        Wrapper function to predict the category of an uploaded image.
        Args:
            image: The image uploaded via Gradio.
        Returns:
            str: Predicted category as a string.
        """
        prediction = predictor.predict(image)
        return {
            "prediction": prediction
            }  # Return the prediction

    # Define Gradio Interface
    interface = gr.Interface(
        fn=gradio_inference,
        inputs=gr.Image(type="filepath", label="Upload an Image"),
        outputs="json",
        title="Fashion Apparel Images Classifier",
        description="Upload an image to classify it using the trained model.",
        allow_flagging="never"  # Disable the flag button
    )

    # Launch the Gradio Interface
    interface.launch(share=True)

    
if __name__ == "__main__":
    # Parameters
    model_path = "./model/xception_v1_best.keras"
    class_mapping_path = "./model/class_mapping.json"
    input_size = 299

    # Create a predictor
    predictor = Predictor(model_path, class_mapping_path, input_size)

#    # Test with file path
#     image_path = "./images/test_image.jpg"
#     print("Prediction (file path):", predictor.predict(image_path))

#     # Test with byte stream
#     with open(image_path, "rb") as img_file:
#         image_bytes = img_file.read()
#     print("Prediction (byte stream):", predictor.predict(image_bytes))
    
    # Launch the Gradio interface
    launch_gradio_interface(predictor)