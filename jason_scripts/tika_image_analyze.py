import requests
import pandas as pd
import os
import json
import time
import concurrent.futures
from tqdm import tqdm
import logging
from requests.adapters import HTTPAdapter
#from requests.packages.urllib3.util.retry import Retry
from urllib3.util.retry import Retry  # Updated import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='image_processing.log'
)

# Local paths
LOCAL_IMAGES_DIR = "C:/Users/jason/OneDrive/Desktop/GeoParser/generated_images"
CSV_PATH = "C:/Users/jason/OneDrive/Desktop/GeoParser/processed_image_data.csv"
FINAL_CSV_PATH = "C:/Users/jason/OneDrive/Desktop/GeoParser/haunted_places_v2.csv"

# Tika API Endpoints - UNCHANGED from your original script
TIKA_IMAGE_CAPTION_API = "http://localhost:8764/inception/v3/caption/image"
TIKA_OBJECT_DETECTION_API = "http://localhost:9998/tika"

# Configure session with retries
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

# Load or create DataFrame
def load_or_create_df(csv_path, images_dir):
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded CSV with {len(df)} rows")
        print(f"Loaded CSV with {len(df)} rows")
        
        # Check if any images need processing
        if 'image_caption' in df.columns and 'image_metadata' in df.columns:
            missing_captions = df['image_caption'].isna().sum()
            if missing_captions == 0:
                logging.info("All images already have captions and metadata")
                return df, []
        
        # Get list of images that need processing
        to_process = []
        for index, row in df.iterrows():
            if 'image_path' in row:
                image_path = row["image_path"]
            else:
                image_path = os.path.join(images_dir, row.get("image_filename", f"image_{index}.png"))
            
            # Skip if already processed and data exists
            if 'image_caption' in df.columns and 'image_metadata' in df.columns:
                if pd.notna(row.get('image_caption')) and pd.notna(row.get('image_metadata')):
                    continue
            
            if os.path.exists(image_path):
                to_process.append((index, image_path))
        
        return df, to_process
        
    except Exception as e:
        logging.info(f"Creating new DataFrame: {e}")
        print("Creating new DataFrame")
        
        # Create a DataFrame from the images in the directory
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        df = pd.DataFrame({
            'image_filename': image_files,
            'image_path': [os.path.join(images_dir, f) for f in image_files]
        })
        
        to_process = [(index, row['image_path']) for index, row in df.iterrows()]
        return df, to_process

# Function to analyze an image with Tika - CORE LOGIC PRESERVED from your original script
def analyze_image(args):
    index, image_path = args
    session = create_session()
    results = {'index': index, 'caption': None, 'metadata': None}
    
    try:
        # Get image caption - SAME API ENDPOINT AND PARAMETERS as your original
        with open(image_path, 'rb') as img:
            caption_response = session.post(
                TIKA_IMAGE_CAPTION_API, 
                data=img.read(),
                timeout=30  # Added timeout
            )
        
        if caption_response.status_code == 200:
            try:
                caption_data = caption_response.json()
                # Extract top caption from the response - IDENTICAL LOGIC to your original
                if "captions" in caption_data and len(caption_data["captions"]) > 0:
                    results['caption'] = caption_data["captions"][0]["sentence"]
                else:
                    results['caption'] = "No caption found in response"
            except Exception as e:
                logging.error(f"Error parsing caption for {image_path}: {e}")
                results['caption'] = "Error parsing caption"
        else:
            logging.error(f"Error getting caption for {image_path}: {caption_response.status_code}")
            results['caption'] = f"Error: HTTP {caption_response.status_code}"

        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
        
        # Get image metadata using Tika - SAME API ENDPOINT AND PARAMETERS as your original
        with open(image_path, 'rb') as img:
            headers = {"Accept": "application/json"}
            metadata_response = session.put(
                TIKA_OBJECT_DETECTION_API, 
                headers=headers,
                data=img.read(),
                timeout=30  # Added timeout
            )
        
        if metadata_response.status_code == 200:
            try:
                metadata = metadata_response.json()
                # Extract all available metadata - IDENTICAL LOGIC to your original
                relevant_fields = {k: v for k, v in metadata.items() 
                                if k not in ["X-TIKA:Parsed-By", "X-TIKA:Parsed-By-Full-Set"]}
                results['metadata'] = json.dumps(relevant_fields)
            except Exception as e:
                logging.error(f"Error parsing metadata for {image_path}: {e}")
                results['metadata'] = "{}"
        else:
            logging.error(f"Error getting metadata for {image_path}: {metadata_response.status_code}")
            results['metadata'] = "{}"
            
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        results['caption'] = f"Error: {str(e)}"
        results['metadata'] = "{}"
        
    return results

# Main processing function
def main():
    start_time = time.time()
    
    # Load or create the DataFrame
    df, to_process = load_or_create_df(CSV_PATH, LOCAL_IMAGES_DIR)
    
    if not to_process:
        logging.info("No images need processing")
        print("No images need processing")
        df.to_csv(FINAL_CSV_PATH, index=False)
        print(f"Final dataset saved: {FINAL_CSV_PATH}")
        return
    
    # Initialize results storage if needed
    if 'image_caption' not in df.columns:
        df['image_caption'] = None
    if 'image_metadata' not in df.columns:
        df['image_metadata'] = None
    
    # Calculate chunk size for batch processing to avoid memory issues
    total_images = len(to_process)
    chunk_size = min(500, total_images)  # Process in chunks of 500 images
    
    logging.info(f"Processing {total_images} images in chunks of {chunk_size}")
    print(f"Processing {total_images} images in chunks of {chunk_size}")
    
    # Process in chunks
    for i in range(0, total_images, chunk_size):
        chunk = to_process[i:i+chunk_size]
        logging.info(f"Processing chunk {i//chunk_size + 1}/{(total_images+chunk_size-1)//chunk_size}")
        print(f"Processing chunk {i//chunk_size + 1}/{(total_images+chunk_size-1)//chunk_size}")
        
        # Process images in parallel - limit workers to avoid overwhelming Tika servers
        max_workers = min(10, len(chunk))  # Reduced from 20 to 10 for stability
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(analyze_image, chunk),
                total=len(chunk),
                desc="Processing images"
            ))
        
        # Update DataFrame with results
        for result in results:
            index = result['index']
            df.at[index, 'image_caption'] = result['caption']
            df.at[index, 'image_metadata'] = result['metadata']
        
        # Save intermediate results
        intermediate_path = f"{FINAL_CSV_PATH}.part{i//chunk_size + 1}"
        df.to_csv(intermediate_path, index=False)
        logging.info(f"Saved intermediate results to {intermediate_path}")
        
    # Save final dataset
    df.to_csv(FINAL_CSV_PATH, index=False)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Final dataset saved: {FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()