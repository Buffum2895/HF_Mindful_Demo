import gradio as gr
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import torch
from PIL import Image
import numpy as np

# Load the pre-trained DPT model and feature extractor
model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
feature_extractor = DPTFeatureExtractor.from_pretrained('Intel/dpt-large')

# Function to perform depth estimation
def estimate_depth(image):
    # Convert the image to the required format
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Predict depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Resize and normalize the depth map for visualization
    depth = predicted_depth.squeeze().cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    
    return Image.fromarray((normalized_depth * 255).astype("uint8"))

# Create a Gradio interface for image input and output
interface = gr.Interface(
    fn=estimate_depth,
    inputs=gr.Image(type="pil"),  # Expect a PIL image as input
    outputs="image",              # Output a depth image
    title="Depth Estimation Demo",
    description="Upload an image and the model will estimate its depth.",
)

# Launch the Gradio app
interface.launch()
