import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    return load_model(r"C:\Users\Zarrar\Desktop\personality_traits\personality_model.h5")

# Preprocess image for prediction
def load_and_preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

# Predict personality traits
def predict_personality(model, image):
    processed_image = load_and_preprocess_image(image)
    image_batch = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(image_batch)
    return prediction

# Main Streamlit app
def main():
    st.title("Personality Trait Inference from Image")
    
    # Load the model
    model = load_trained_model()
    
    # Trait names
    trait_names = [
        "Agreeableness (AO)", 
        "Conscientiousness (VS)",
        "Emotional Stability (SC)", 
        "Extraversion (IE)", 
        "Openness (OP)"
    ]
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict personality traits
        prediction = predict_personality(model, image)
        
        # Display prediction results
        st.subheader("Personality Trait Scores")
        
        # Create a bar plot of trait scores
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(trait_names))
        ax.barh(y_pos, prediction[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(trait_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Score')
        ax.set_title('Personality Trait Scores')
        
        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Display detailed scores and plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detailed Scores")
            for trait, score in zip(trait_names, prediction[0]):
                st.metric(trait, f"{score:.2f}")
        
        with col2:
            st.subheader("Trait Score Visualization")
            st.image(buf)
        
        # Identify dominant trait
        dominant_trait_index = np.argmax(prediction[0])
        dominant_trait = trait_names[dominant_trait_index]
        st.success(f"Dominant Personality Trait: {dominant_trait}")

if __name__ == "__main__":
    main()