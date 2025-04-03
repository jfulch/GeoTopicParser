import pandas as pd
import os

# Local paths
PROJECT_DIR = "C:/Users/jason/OneDrive/Desktop/GeoParser"
ORIGINAL_TSV_PATH = os.path.join(PROJECT_DIR, "haunted_places_extracted_entities_spacy.tsv")
GENERATED_IMAGES_CSV = os.path.join(PROJECT_DIR, "haunted_places_v2.csv")
OUTPUT_PATH = GENERATED_IMAGES_CSV

def merge_datasets():
    # Load original dataset
    print(f"Loading original dataset from: {ORIGINAL_TSV_PATH}")
    try:
        original_df = pd.read_csv(ORIGINAL_TSV_PATH, sep='\t')
        print(f"Original dataset loaded: {len(original_df)} rows")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        return
    
    # Load generated images dataset with captions and objects
    print(f"Loading generated images dataset from: {GENERATED_IMAGES_CSV}")
    try:
        images_df = pd.read_csv(GENERATED_IMAGES_CSV)
        print(f"Generated images dataset loaded: {len(images_df)} rows")
    except Exception as e:
        print(f"Error loading generated images dataset: {e}")
        return
    
    # Check if images_df has the necessary columns
    required_columns = ['index', 'image_caption', 'detected_objects']
    missing_columns = [col for col in required_columns if col not in images_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in generated images dataset: {missing_columns}")
        return
    
    # Create a mapping from index to caption and objects
    print("Creating mapping from indices to captions and objects...")
    caption_map = {}
    objects_map = {}
    
    for _, row in images_df.iterrows():
        idx = row['index']
        caption = row.get('image_caption', 'No caption')
        objects = row.get('detected_objects', 'No objects')
        
        caption_map[idx] = caption
        objects_map[idx] = objects
    
    # Add new columns to original dataset
    print("Adding image caption and detected objects columns to original dataset...")
    original_df['image_caption'] = original_df.index.map(lambda x: caption_map.get(x, 'No caption available'))
    original_df['detected_objects'] = original_df.index.map(lambda x: objects_map.get(x, 'No objects detected'))
    
    # Count successful matches
    caption_matches = sum(1 for caption in original_df['image_caption'] if caption != 'No caption available')
    object_matches = sum(1 for objects in original_df['detected_objects'] if objects != 'No objects detected')
    
    print(f"Successfully matched {caption_matches} captions and {object_matches} object sets")
    
    # Save the merged dataset
    print(f"Saving merged dataset to: {OUTPUT_PATH}")
    original_df.to_csv(OUTPUT_PATH, sep='\t', index=False)
    print("Done!")
    
    # Show a sample of the results
    print("\nSample of merged dataset:")
    sample = original_df.sample(min(5, len(original_df)))
    for idx, row in sample.iterrows():
        print(f"Index: {idx}")
        print(f"Location: {row.get('location', 'Unknown')}")
        print(f"Image Caption: {row.get('image_caption', 'No caption')}")
        print(f"Detected Objects: {row.get('detected_objects', 'No objects')}")
        print("-" * 50)

if __name__ == "__main__":
    merge_datasets()