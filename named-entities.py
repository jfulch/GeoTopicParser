import spacy
import pandas as pd
import sys

# Load spaCy English language model
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model not found. Installing spaCy English model...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    """Extract named entities from text using spaCy"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    doc = nlp(text)
    # Extract entities with their types
    entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    return "; ".join(entities) if entities else ""

def main():

    input_file = "haunted_places_augmented.tsv"  # Default input file
    output_file = "haunted_places_augmented_with_entities.tsv"
    
    print(f"Processing file: {input_file}")
    
    # Read input TSV file
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Verify 'description' column exists
    if 'description' not in df.columns:
        print("Error: 'description' column not found in the input file")
        sys.exit(1)
    
    # Process descriptions and extract named entities
    print(f"Extracting named entities from {len(df)} descriptions...")
    df['named_entities'] = df['description'].apply(extract_named_entities)
    
    # Save results to output file
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, sep='\t', index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()