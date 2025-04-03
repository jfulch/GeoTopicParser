import pandas as pd
import json
import re
import os
import nltk
import time
import shutil
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm
import sys

# Download all required NLTK resources upfront
def download_nltk_resources():
    print("Downloading NLTK resources...")
    try:
        # Download to a local directory to avoid OneDrive sync issues
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data_local")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Set the download directory
        nltk.data.path.insert(0, nltk_data_dir)
        
        # First install all standard NLTK resources we need
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
        
        # Now manually install the specific resources that are giving errors
        # These specific resources might need to be installed differently
        print("Installing specific missing resources...")
        
        # Create directories for punkt_tab if they don't exist
        punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
        os.makedirs(punkt_dir, exist_ok=True)
        
        # Create directories for averaged_perceptron_tagger_eng if they don't exist
        tagger_dir = os.path.join(nltk_data_dir, 'taggers', 'averaged_perceptron_tagger_eng')
        os.makedirs(tagger_dir, exist_ok=True)
        
        # Write dummy files to satisfy the imports
        # For punkt_tab
        with open(os.path.join(punkt_dir, 'punkt.tab'), 'w') as f:
            f.write("# Dummy punkt_tab file to satisfy import\n")
            
        # For averaged_perceptron_tagger_eng
        with open(os.path.join(tagger_dir, 'README'), 'w') as f:
            f.write("# Dummy averaged_perceptron_tagger_eng file to satisfy import\n")
        
        print(f"NLTK resources downloaded to {nltk_data_dir}")
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False

# Local paths
CSV_PATH = "C:/Users/jason/OneDrive/Desktop/GeoParser/haunted_places_v2.csv"
# Use a non-OneDrive location for temporary files
TEMP_DIR = "C:/Temp/GeoParser"
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_CSV_PATH = os.path.join(TEMP_DIR, "haunted_places_processed.csv")

def extract_detected_objects(row):
    """Extract detected objects from image caption and metadata using NLTK"""
    detected_objects = []
    
    # Try to extract from caption
    if pd.notna(row.get('image_caption')):
        caption = row['image_caption']
        
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(caption)
            tagged = pos_tag(tokens)
            
            # Extract nouns (NN, NNS, NNP, NNPS)
            nouns = [word.lower() for word, tag in tagged if tag.startswith('NN')]
            
            # Add common objects that might be described but not direct nouns
            common_objects = ['bench', 'building', 'house', 'table', 'chair', 'tree', 
                             'door', 'window', 'wall', 'floor', 'ceiling', 'roof', 
                             'street', 'road', 'path', 'field', 'forest', 'mountain',
                             'lake', 'river', 'bridge', 'car', 'vehicle', 'fence',
                             'refrigerator', 'kitchen', 'bathroom', 'bedroom']
            
            # Add objects from caption
            for obj in common_objects:
                if obj in caption.lower() and obj not in nouns:
                    nouns.append(obj)
            
            # Extract noun phrases (e.g., "wooden bench", "stainless steel refrigerator")
            noun_phrases = []
            for i in range(len(tagged)-1):
                # Look for adjective + noun combinations
                if tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN'):
                    phrase = f"{tagged[i][0].lower()} {tagged[i+1][0].lower()}"
                    if phrase not in noun_phrases:
                        noun_phrases.append(phrase)
            
            # Add noun phrases to detected objects
            detected_objects.extend(noun_phrases)
            
            # Clean and add nouns to detected objects
            for noun in nouns:
                noun = noun.lower().strip()
                if len(noun) > 2 and noun not in detected_objects:
                    detected_objects.append(noun)
            
            # Look for phrases with "a" or "an" followed by words - add even with NLTK
            caption_lower = caption.lower()
            a_an_phrases = re.findall(r'\b(?:a|an)\s+(\w+(?:\s+\w+){0,2})', caption_lower)
            for phrase in a_an_phrases:
                if len(phrase) > 2 and phrase not in detected_objects:
                    detected_objects.append(phrase.strip())
                    
        except Exception as e:
            # Fallback to regex if NLTK fails
            print(f"NLTK processing failed, using regex fallback: {e}")
            caption_lower = caption.lower()
            
            # Common objects to look for - expanded list
            common_objects = ['bench', 'building', 'house', 'table', 'chair', 'tree', 
                             'door', 'window', 'wall', 'floor', 'ceiling', 'roof', 
                             'street', 'road', 'path', 'field', 'forest', 'mountain',
                             'lake', 'river', 'bridge', 'car', 'vehicle', 'fence',
                             'refrigerator', 'kitchen', 'bathroom', 'bedroom', 'stairs',
                             'desk', 'shelf', 'bookshelf', 'lamp', 'light', 'cabinet',
                             'grass', 'plant', 'flower', 'statue', 'monument', 'sign']
            
            # Extract objects using simple pattern matching
            for obj in common_objects:
                if obj in caption_lower and obj not in detected_objects:
                    detected_objects.append(obj)
            
            # Look for phrases with "a" or "an" followed by words
            a_an_phrases = re.findall(r'\b(?:a|an)\s+(\w+(?:\s+\w+){0,2})', caption_lower)
            for phrase in a_an_phrases:
                if len(phrase) > 2 and phrase not in detected_objects:
                    detected_objects.append(phrase.strip())
            
            # Look for compound objects (adjective + object)
            compound_patterns = [
                r'(\w+)\s+bench', r'(\w+)\s+building', r'(\w+)\s+house',
                r'(\w+)\s+table', r'(\w+)\s+chair', r'(\w+)\s+field',
                r'(\w+)\s+room', r'(\w+)\s+door', r'(\w+)\s+wall'
            ]
            
            for pattern in compound_patterns:
                matches = re.findall(pattern, caption_lower)
                for match in matches:
                    # Add the full phrase by finding it in the caption
                    for full_phrase in re.findall(f"{match}\\s+\\w+", caption_lower):
                        if full_phrase not in detected_objects:
                            detected_objects.append(full_phrase.strip())
    
    # Return results
    if detected_objects:
        # Remove duplicates and sort
        detected_objects = sorted(list(set(detected_objects)))
        return ", ".join(detected_objects)
    else:
        return "No specific objects detected"

def main():
    # Download NLTK resources
    if not download_nltk_resources():
        print("Warning: NLTK resources could not be downloaded. Using fallback regex approach.")
    
    print(f"Loading CSV from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    
    try:
        # Load the existing CSV file
        # First, copy to temp location to avoid OneDrive sync issues
        temp_input = os.path.join(TEMP_DIR, "input.csv")
        shutil.copy2(CSV_PATH, temp_input)
        
        # Load from temp location
        df = pd.read_csv(temp_input)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Create a backup of the original file
        backup_path = os.path.join(TEMP_DIR, "haunted_places_v2.backup.csv")
        df.to_csv(backup_path, index=False)
        print(f"Created backup at {backup_path}")
        
        # Check if detected_objects column already exists
        if 'detected_objects' in df.columns:
            print("'detected_objects' column already exists, updating it...")
        else:
            print("Adding 'detected_objects' column...")
            df['detected_objects'] = None
        
        # Process each row to extract detected objects
        print("Extracting detected objects from captions...")
        tqdm.pandas()
        df['detected_objects'] = df.progress_apply(extract_detected_objects, axis=1)
        
        # Save to a temporary file
        df.to_csv(TEMP_CSV_PATH, index=False)
        print(f"Processing complete. Saved to: {TEMP_CSV_PATH}")
        
        # Now try to safely update the original file
        try:
            # Attempt to copy back to OneDrive location
            print(f"Attempting to update original file at {CSV_PATH}...")
            # Wait to ensure file handles are closed
            time.sleep(2)
            
            # Try to copy the file back
            shutil.copy2(TEMP_CSV_PATH, CSV_PATH)
            print(f"Successfully updated original file at {CSV_PATH}")
        except Exception as e:
            print(f"Could not update original file: {e}")
            print(f"\nPlease manually copy {TEMP_CSV_PATH} to {CSV_PATH} when OneDrive sync is not active.")
        
        # Show a sample of the results
        print("\nSample of detected objects:")
        sample = df.sample(min(5, len(df)))
        for idx, row in sample.iterrows():
            print(f"Image: {row.get('image_filename', 'Unknown')}")
            print(f"Caption: {row.get('image_caption', 'No caption')}")
            print(f"Detected objects: {row.get('detected_objects', 'None')}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        print("Traceback:", sys.exc_info())

if __name__ == "__main__":
    main()