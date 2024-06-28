import streamlit as st
import numpy as np
import cv2
import pickle as pkl
from PIL import Image

# Load the combined model
with open('app.pkl', 'rb') as f:
    model = pkl.load(f)

# Define image size
image_size = 224

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make a prediction
def make_prediction(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Flatten the image for the model
    input_array = processed_image.reshape(1, -1)
    
    # Predict the class using the combined model
    prediction = model.predict(input_array)
    
    # Map the prediction to the corresponding class
    labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
    predicted_class = labels[prediction[0]]
    
    return predicted_class

def main():
    st.title("Brain Tumor Classification")

    # File uploader for image upload
    uploaded_file = st.file_uploader("Upload a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Make Prediction'):
            # Make a prediction on the uploaded image
            result = make_prediction(image)
            st.success(f'The prediction is: {result}')

if __name__ == '__main__':
    main()
