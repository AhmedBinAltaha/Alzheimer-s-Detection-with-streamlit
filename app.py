import io
import streamlit as st
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
MODEL_PATH = 'tflite_model_another.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Flask app
app = Flask(__name__)

# Function for processing the input image and prediction
def model_predict(img_data, interpreter):
    # Preprocessing the image
    img = img_data.resize((224, 224))  # Resize the image directly
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], x)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_class_index = np.argmax(output)
    classes = ['AD', 'CN', 'EMCI', 'LMCI']
    pred_class = classes[pred_class_index]
    prob = round(float(output[0, pred_class_index]), 2)

    # Format predictions
    results = [{"class": pred_class, "probability": prob}]
    return results

# Main Streamlit app
def main():
    st.title("Alzheimer's Disease Diagnosis App")
    st.write("The Site For Alzheimer's Disease Diagnosis For 4 Class ADNI Data Categories")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False, width=224)

        st.write("")
        st.write("Classifying...")

        # Open the uploaded image
        img = Image.open(uploaded_file)

        # Make prediction
        result = model_predict(img, interpreter)

        # Process your result for human
        if result:
            st.success(f"The predicted class is: {result[0]['class']} with probability: {result[0]['probability']}")

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)
    result = model_predict(img, interpreter)
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    main()
    app.run(port=5002)
