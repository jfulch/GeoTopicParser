"""
Stable Diffusion Image Generation Script (GPU-accelerated mode for NVIDIA RTX 3080)
Optimized for fast image generation on dedicated GPU hardware
"""
import os
import time
import json
import pandas as pd
from PIL import Image
import traceback
import gc

# Local paths (update these to match your desktop paths)
PROJECT_DIR = r"C:\Users\jason\OneDrive\Desktop\GeoParser"  # Using raw string with r prefix
ENTITIES_PATH = os.path.join(PROJECT_DIR, "haunted_places_extracted_entities_spacy.tsv")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "generated_images")
METADATA_OUTPUT_PATH = os.path.join(PROJECT_DIR, "processed_image_data.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "generation_checkpoint.json")

# Ensure output directory exists
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Configuration - Optimized for RTX 3080 with 10GB VRAM (prioritizing quality)
BATCH_SIZE = 20        # Process more images per batch while maintaining memory for quality
IMAGE_SIZE = 512      # High quality images
NUM_INFERENCE_STEPS = 50  # Increased steps for better quality and detail
GUIDANCE_SCALE = 8.0  # Higher guidance scale for better prompt adherence
START_INDEX = 0       # Start from this index (useful for resuming)
END_INDEX = 10992     # Process up to this index

def load_model():
    """Load the Stable Diffusion model with GPU acceleration"""
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        
        # Model ID from Hugging Face
        model_id = "stabilityai/stable-diffusion-2-1"
        
        print("Checking CUDA availability...")
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
            print("CUDA not available, falling back to CPU (not recommended)")
        
        # Load pipeline with appropriate settings for GPU
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use half precision for GPU
            safety_checker=None,  # Disable safety checker for speed
        ).to(device)
        
        # Set scheduler for faster inference
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Enable memory optimization for GPU
        if device == "cuda":
            pipe.enable_attention_slicing()
            # Optional: Enable xformers for memory efficiency if installed
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except:
                print("xformers not available, using standard attention")
        
        print(f"Model loaded successfully on {device}")
        return pipe, device
    
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        exit(1)

def generate_image(pipe, prompt, output_file, device, seed=None):
    """Generate an image using the pipeline with robust error handling"""
    try:
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Create generator with seed if provided
        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Start generation
        start_time = time.time()
        
        # Generate with GPU acceleration
        try:
            result = pipe(
                prompt=prompt, 
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                generator=generator
            )
            
            image = result.images[0]
            end_time = time.time()
            
            # Save image
            image.save(output_file)
            duration = end_time - start_time
            print(f"Image generated in {duration:.2f} seconds and saved to: {os.path.basename(output_file)}")
            
            # Force garbage collection to prevent memory leaks
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return True
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress...")
            # Need to return False to avoid marking this image as processed
            return False
        except torch.cuda.OutOfMemoryError:
            print(f"GPU out of memory error! Clearing cache and skipping this complex prompt.")
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return False
    
    except Exception as e:
        print(f"Error generating image: {e}")
        traceback.print_exc()
        return False

def save_checkpoint(processed_indices, processed_data):
    """Save checkpoint to resume later"""
    checkpoint = {
        "last_processed_index": max(processed_indices) if processed_indices else START_INDEX - 1,
        "processed_indices": processed_indices,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save checkpoint file with error handling
    try:
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f)
    except Exception as e:
        print(f"Warning: Could not save checkpoint file: {e}")
    
    # Save processed data with error handling
    try:
        processed_df = pd.DataFrame(processed_data)
        # Try multiple attempts with different filenames if needed
        try:
            processed_df.to_csv(METADATA_OUTPUT_PATH, index=False)
        except PermissionError:
            # Try alternative filename if original is locked
            alt_path = os.path.join(PROJECT_DIR, f"processed_image_data_backup_{int(time.time())}.csv")
            print(f"Could not access original file, saving to: {alt_path}")
            processed_df.to_csv(alt_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save metadata: {e}")
        # Continue processing even if saving fails
    
    print(f"Checkpoint saved. Processed {len(processed_indices)} images so far.")

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                checkpoint = json.load(f)
            
            print(f"Found checkpoint from {checkpoint['timestamp']}")
            print(f"Resuming from index {checkpoint['last_processed_index'] + 1}")
            
            return checkpoint['last_processed_index'] + 1, checkpoint['processed_indices']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return START_INDEX, []

def create_prompt(row):
    """Create a more practical prompt with location prominently featured"""
    # Extract location info
    location_name = ""
    for field in ['extracted_location_name', 'location', 'city']:
        if field in row and not pd.isna(row[field]):
            location_name = row[field]
            break
    
    if not location_name:
        location_name = "Haunted place"  # Generic fallback if no location name is found
    
    # Extract entities (focus on GPE and LOC entities)
    entities = []
    try:
        for col in ['spacy_entities', 'entities']:
            if col in row and not pd.isna(row[col]):
                entities = json.loads(row[col])
                break
    except Exception as e:
        print(f"Couldn't parse entities: {e}")
    
    # Filter relevant entities
    entity_text = ""
    try:
        relevant_entities = [e['text'] for e in entities if isinstance(e, dict) and 'label' in e and e['label'] in ['GPE', 'LOC']]
        entity_text = ", ".join(relevant_entities)
    except Exception as e:
        print(f"Error processing entities: {e}")
    
    # Start with the location to ensure it's included
    prompt = f"Haunted scene at {location_name}: "
    
    # Add description (truncated intelligently if needed)
    if 'description' in row and not pd.isna(row['description']):
        desc = str(row['description'])
        # Extract the first sentence or part to prioritize description
        first_part = desc.split('.')[0]
        if len(first_part) > 150:
            first_part = first_part[:150]
        prompt += f"{first_part}"
    
    # Add key entities if they're not already in the location name
    if entity_text and not any(entity.lower() in location_name.lower() for entity in entity_text.split(", ")):
        entity_parts = entity_text.split(", ")[:2]  # Limit to 2 entities to save tokens
        short_entity_text = ", ".join(entity_parts)
        prompt += f" ({short_entity_text})"
    
    return prompt, location_name

def main():
    """Main function to process dataset and generate images"""
    global START_INDEX
    
    # Import torch at the beginning of main
    import torch
    
    # Load checkpoint if exists
    START_INDEX, processed_indices = load_checkpoint()
    processed_indices = [int(idx) for idx in processed_indices]  # Ensure integers
    
    # Load model
    pipe, device = load_model()
    
    # Load dataset
    print(f"Loading dataset from: {ENTITIES_PATH}")
    try:
        df_entities = pd.read_csv(ENTITIES_PATH, sep='\t')
        print(f"Loaded dataset with {df_entities.shape[0]} rows and {df_entities.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure the file exists at: {ENTITIES_PATH}")
        return
    
    # Load existing processed data if available
    processed_data = []
    if os.path.exists(METADATA_OUTPUT_PATH):
        try:
            processed_df = pd.read_csv(METADATA_OUTPUT_PATH)
            processed_data = processed_df.to_dict('records')
            print(f"Loaded {len(processed_data)} existing records from metadata file")
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
    
    # Process in batches
    end_index = min(END_INDEX, len(df_entities))
    
    # Calculate number of batches
    total_to_process = end_index - START_INDEX
    num_batches = (total_to_process + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\nProcessing {total_to_process} samples in {num_batches} batches")
    
    batch_start = START_INDEX
    for batch in range(num_batches):
        batch_end = min(batch_start + BATCH_SIZE, end_index)
        
        print(f"\nProcessing batch {batch+1}/{num_batches} (rows {batch_start} to {batch_end-1})")
        
        for index in range(batch_start, batch_end):
            # Skip if already processed
            if index in processed_indices:
                print(f"Skipping row {index}: Already processed")
                continue
                
            row = df_entities.iloc[index]
            
            if 'description' not in row or pd.isna(row['description']):
                print(f"Skipping row {index}: No description available")
                continue
                
            print(f"\nProcessing row {index}")
            
            # Create prompt using the standardized function
            prompt, location_name = create_prompt(row)
                
            # Generate and save the image
            image_filename = f"haunted_{index}.png"
            output_file = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
            seed = 1000 + index  # Fixed seed for consistency
            
            success = generate_image(pipe, prompt, output_file, device, seed=seed)
            if success:
                processed_data.append({
                    'index': index,
                    'location_name': location_name,
                    'image_filename': image_filename,
                    'image_path': output_file,
                    'prompt': prompt
                })
                processed_indices.append(index)
            
            # Save checkpoint after each image
            save_checkpoint(processed_indices, processed_data)
        
        # Move to next batch
        batch_start = batch_end
        
        # Force memory cleanup between batches
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Completed batch {batch+1}/{num_batches}")
    
    # Final save
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(METADATA_OUTPUT_PATH, index=False)
    
    # List generated images
    image_files = [f for f in os.listdir(IMAGE_OUTPUT_DIR) if f.endswith(".png")]
    print(f"\nTotal processed: {len(processed_indices)} images")
    print(f"Generated {len(image_files)} images in {IMAGE_OUTPUT_DIR}")
    print("Images ready for analysis.")

if __name__ == "__main__":
    main()