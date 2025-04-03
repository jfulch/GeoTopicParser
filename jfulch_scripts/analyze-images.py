import requests
import pandas as pd
import os
import json
from PIL import Image

# Local paths
LOCAL_IMAGES_DIR = "../generated_images"
CSV_PATH = "/processed_image_data.csv"

# Tika API Endpoints (with verified working methods)
TIKA_IMAGE_CAPTION_API = "http://localhost:8764/inception/v3/caption/image"
TIKA_OBJECT_DETECTION_API = "http://localhost:9998/tika"

# Load CSV (if it exists, otherwise create a new one)
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV with {len(df)} rows")
except:
    print("Creating new DataFrame")
    # Create a DataFrame from the images in the directory
    image_files = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.endswith('.png')]
    df = pd.DataFrame({
        'image_filename': image_files,
        'image_path': [os.path.join(LOCAL_IMAGES_DIR, f) for f in image_files]
    })

# Function to analyze an image with Tika
def analyze_image(image_path):
    # Get image caption - using the successful method from testing
    with open(image_path, 'rb') as img:
        caption_response = requests.post(
            TIKA_IMAGE_CAPTION_API, 
            data=img.read()
        )
    
    if caption_response.status_code == 200:
        try:
            caption_data = caption_response.json()
            # Extract top caption from the response
            if "captions" in caption_data and len(caption_data["captions"]) > 0:
                caption = caption_data["captions"][0]["sentence"]
            else:
                caption = "No caption found in response"
            print(f"Caption: {caption}")
        except Exception as e:
            print(f"Error parsing caption: {e}")
            caption = "Error parsing caption"
    else:
        print(f"Error getting caption: {caption_response.status_code}")
        caption = "Error generating caption"

    # Get image metadata using Tika
    with open(image_path, 'rb') as img:
        headers = {"Accept": "application/json"}
        metadata_response = requests.put(
            TIKA_OBJECT_DETECTION_API, 
            headers=headers,
            data=img.read()
        )
    
    if metadata_response.status_code == 200:
        try:
            metadata = metadata_response.json()
            # Extract all available metadata
            # Using metadata as "detected objects"
            relevant_fields = {k: v for k, v in metadata.items() 
                              if k not in ["X-TIKA:Parsed-By", "X-TIKA:Parsed-By-Full-Set"]}
            metadata_str = json.dumps(relevant_fields)
            print(f"Metadata: {metadata_str[:100]}...")
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            metadata_str = "{}"
    else:
        print(f"Error getting metadata: {metadata_response.status_code}")
        metadata_str = "{}"

    return caption, metadata_str

# Process all images
captions = []
objects_detected = []

for index, row in df.iterrows():
    if 'image_path' in row:
        image_path = row["image_path"]
    else:
        # If image_path not in row, construct from image_filename
        image_path = os.path.join(LOCAL_IMAGES_DIR, row.get("image_filename", f"haunted_{index}.png"))

    if os.path.exists(image_path):
        print(f"\nProcessing: {image_path}")
        caption, metadata = analyze_image(image_path)
        captions.append(caption)
        objects_detected.append(metadata)
    else:
        print(f"Skipping missing file: {image_path}")
        captions.append("No image found")
        objects_detected.append("{}")

# Add results to DF
df["image_caption"] = captions
df["image_metadata"] = objects_detected  

# Save final dataset
final_csv_path = "../haunted_places_description.csv"
df.to_csv(final_csv_path, index=False)

print(f"\nFinal dataset saved: {final_csv_path}")