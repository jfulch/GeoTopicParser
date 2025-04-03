import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pandas as pd
import json
import os
import time
from PIL import Image

# Define dataset path local
entities_path = "../split_files/batch3-1501-2000.tsv"

# Ensure the output directory exists
image_output_dir = "../generated_images"
os.makedirs(image_output_dir, exist_ok=True)

# Load haunted places extracted entities dataset
print("Loading dataset...")
df_entities = pd.read_csv(entities_path, sep='\t')
print(f"Loaded dataset with {df_entities.shape[0]} rows and {df_entities.shape[1]} columns")

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    num_gpus = torch.cuda.device_count()
    print(f"⚡ Using {num_gpus} GPU(s) for image generation")
else:
    device = "cpu"
    num_gpus = 0
    print("⚠️ No GPU detected, running on CPU (this will be very slow)")

# Load Stable Diffusion 2.1 Model with optimizations - LOAD ONLY ONCE
model_id = "stabilityai/stable-diffusion-2-1"

print(f"Loading model from {model_id}...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # To speed up loading
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Apply optimizations based on available hardware
    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()  # Basic optimization for all GPU setups
        
        if num_gpus > 1:
            print("Configuring for multi-GPU setup...")
            # Multi-GPU setup
            pipe.unet = torch.nn.DataParallel(pipe.unet)
        
        # Reduce inference steps to speed up when on CPU
        inference_steps = 50
    else:
        # CPU optimizations - use fewer steps to save time
        inference_steps = 20
        print(f"⚠️ Using reduced quality settings for CPU (inference_steps={inference_steps})")
        pipe.enable_attention_slicing()
        pipe = pipe.to(device)
    
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Function to generate and save images
def generate_image(prompt, output_file, seed=None, steps=inference_steps):
    print(f"Generating image for: {prompt[:50]}...")
    try:
        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        image = pipe(
            prompt=prompt, 
            guidance_scale=7.5, 
            num_inference_steps=steps,
            generator=generator
        ).images[0]
        
        image.save(output_file)
        print(f"✅ Saved image: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return False

# Process dataset and generate images
num_samples = 10  # Adjust as needed
processed_data = []

print(f"\n🚀 Starting to process {num_samples} samples...")
for index, row in df_entities.head(num_samples).iterrows():
    if pd.isna(row.get('description')):
        print(f"Skipping row {index} - missing description")
        continue

    print(f"\n📝 Processing row {index+1}/{num_samples}")

    # Extract location info
    location_name = next(
        (row.get(field) for field in ['extracted_location_name', 'location', 'city']
         if field in row and not pd.isna(row.get(field))), "Unknown location"
    )

    # Extract entities (focus on GPE and LOC for geographic context)
    try:
        entities = []
        for col in ['spacy_entities', 'entities']:
            if col in row and not pd.isna(row.get(col)):
                entities = json.loads(row.get(col))
                break
    except Exception as e:
        print(f"Couldn't parse entities: {e}")
        entities = []

    # Filter relevant entities
    try:
        relevant_entities = [e['text'] for e in entities if isinstance(e, dict) and 'label' in e and e['label'] in ['GPE', 'LOC']]
        entity_text = ", ".join(relevant_entities)
    except Exception as e:
        print(f"Error processing entities: {e}")
        entity_text = ""

    # Create prompt
    prompt = f"A haunted location at {location_name}. "
    if entity_text:
        prompt += f"Includes elements: {entity_text}. "
    prompt += "Eerie atmosphere, dark shadows, mist, old architecture, abandoned feeling."

    print(f"🔮 Prompt: {prompt}")

    # Generate and save the image
    image_filename = f"haunted_{index}.png"
    output_file = os.path.join(image_output_dir, image_filename)
    seed = 1000 + index  # Fixed seed for consistency
    success = generate_image(prompt, output_file, seed=seed)

    if success:
        processed_data.append({
            'index': index,
            'location_name': location_name,
            'image_filename': image_filename,
            'image_path': output_file,
            'prompt': prompt
        })

    # Prevent overloading resources
    if index < num_samples - 1:  # No need to wait after the last image
        if device == "cpu":
            print("💤 Pausing briefly between CPU operations...")
            time.sleep(2)  # Longer pause for CPU to cool down
        else:
            time.sleep(1)  # Regular pause for GPU

print(f"\n✨ Processed {len(processed_data)} images successfully!")

# Optionally save metadata about the processed images
try:
    with open(os.path.join(image_output_dir, "metadata.json"), "w") as f:
        json.dump(processed_data, f, indent=2)
    print(f"📄 Saved metadata to {os.path.join(image_output_dir, 'metadata.json')}")
except Exception as e:
    print(f"❌ Error saving metadata: {e}")