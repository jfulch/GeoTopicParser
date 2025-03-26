import pandas as pd
import spacy
import json

def extract_entities_with_spacy(text):
    """Extract entities from text using spaCy with confidence scores."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        token_scores = [token.prob for token in ent]
        confidence = sum(token_scores) / len(token_scores) if token_scores else 0
        
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': confidence
        })
    
    return entities

def main():
    df = pd.read_csv('haunted_places_augmented.tsv', sep='\t')
    spacy_df = df.copy()
    spacy_df['spacy_entities'] = None
    
    for index, row in df.iterrows():
        if 'description' in row:
            print(f"Processing row {index+1}/{len(df)}")
            description = row['description']
            
            if pd.isna(description) or description.strip() == '':
                continue
            
            entities = extract_entities_with_spacy(description)
            spacy_df.at[index, 'spacy_entities'] = json.dumps(entities)
    
    spacy_df.to_csv('haunted_places_extracted_entities.tsv', sep='\t', index=False)
    print("Data with spaCy entities saved to haunted_places_extracted_entities.tsv")

if __name__ == "__main__":
    main()