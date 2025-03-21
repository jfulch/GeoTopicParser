#!/usr/bin/env python3
import pandas as pd
import requests
import tempfile
import os

def extract_locations(text):
    """Send text to Tika server and extract location information"""
    # Create a temporary file containing the text
    with tempfile.NamedTemporaryFile(suffix='.geot', delete=False) as temp:
        temp.write(text.encode('utf-8'))
        temp_path = temp.name
    
    try:
        # Send the file to the Tika server
        headers = {
            "Content-Disposition": f"attachment; filename={os.path.basename(temp_path)}"
        }
        
        with open(temp_path, 'rb') as f:
            response = requests.put(
                "http://localhost:9998/rmeta",
                headers=headers,
                data=f.read()
            )
        
        # Parse the response
        if response.status_code == 200:
            data = response.json()
            
            # Extract location information
            locations = []
            if data and isinstance(data, list) and len(data) > 0:
                # Check for primary geographic location
                if 'Geographic_NAME' in data[0]:
                    locations.append({
                        'name': data[0].get('Geographic_NAME', 'Unknown'),
                        'latitude': data[0].get('Geographic_LATITUDE'),
                        'longitude': data[0].get('Geographic_LONGITUDE')
                    })
                
                # Check for optional/secondary locations
                i = 1
                while f'Optional_NAME{i}' in data[0]:
                    locations.append({
                        'name': data[0].get(f'Optional_NAME{i}', 'Unknown'),
                        'latitude': data[0].get(f'Optional_LATITUDE{i}'),
                        'longitude': data[0].get(f'Optional_LONGITUDE{i}')
                    })
                    i += 1
                    
            return locations
        else:
            print(f"Error: Received status code {response.status_code}")
            return []
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    # Read the TSV file
    df = pd.read_csv('haunted_places.tsv', sep='\t')
    
    # Limit to first 500 rows
    df = df.head(500)
    
    # Create a copy of the dataframe to preserve the original
    augmented_df = df.copy()
    
    # Add new columns for the extracted location data
    augmented_df['extracted_location_name'] = None
    augmented_df['extracted_latitude'] = None
    augmented_df['extracted_longitude'] = None
    
    # Process each description
    for index, row in df.iterrows():
        if 'description' in row:
            print(f"Processing row {index+1}/{len(df)}")
            description = row['description']
            
            # Skip empty descriptions
            if pd.isna(description) or description.strip() == '':
                continue
                
            # Enhanced description with context from the row
            enhanced_description = description
            if not pd.isna(row.get('location')):
                enhanced_description = f"{description} This happened at {row.get('location')} in {row.get('city')}, {row.get('state')}."
                
            # Extract locations from this description
            locations = extract_locations(enhanced_description)
            
            selected_location = None
            
            if locations:
                # 1. First, look for location matches with the location field
                original_location = row.get('location', '').lower()
                for loc in locations:
                    if loc['name'].lower() in original_location or original_location in loc['name'].lower():
                        selected_location = loc
                        print(f"  Found match: {loc['name']} matches {original_location}")
                        break
                
                # 2. If no match found, look for location in city
                if not selected_location and not pd.isna(row.get('city')):
                    city = row.get('city', '').lower()
                    for loc in locations:
                        if loc['name'].lower() in city or city in loc['name'].lower():
                            selected_location = loc
                            print(f"  Found city match: {loc['name']} matches {city}")
                            break
                
                # 3. If still no match, use the first location with coordinates
                if not selected_location:
                    for loc in locations:
                        if loc['latitude'] and loc['longitude']:
                            selected_location = loc
                            print(f"  Using location with coordinates: {loc['name']}")
                            break
                
                # 4. If still no match, just use the first location
                if not selected_location and locations:
                    selected_location = locations[0]
                    print(f"  Using first location: {selected_location['name']}")
            
            # Add the selected location to the DataFrame
            if selected_location:
                augmented_df.at[index, 'extracted_location_name'] = selected_location['name']
                augmented_df.at[index, 'extracted_latitude'] = selected_location['latitude']
                augmented_df.at[index, 'extracted_longitude'] = selected_location['longitude']
            else:
                # Fallback: Use the location from the original data if no locations were extracted
                if not pd.isna(row.get('location')) and not pd.isna(row.get('latitude')) and not pd.isna(row.get('longitude')):
                    print(f"  Using fallback location: {row.get('location')}")
                    augmented_df.at[index, 'extracted_location_name'] = row.get('location')
                    augmented_df.at[index, 'extracted_latitude'] = row.get('latitude')
                    augmented_df.at[index, 'extracted_longitude'] = row.get('longitude')
    
    # Save the augmented dataframe to a new TSV file
    augmented_df.to_csv('haunted_places_augmented.tsv', sep='\t', index=False)
    print(f"Original data plus extracted locations saved to haunted_places_augmented.tsv")
    
    # Calculate how many locations were successfully extracted
    extracted_count = augmented_df['extracted_location_name'].notna().sum()
    print(f"Successfully extracted locations for {extracted_count}/{len(df)} rows")
    
    # Print a sample of the results
    print("\nSample of augmented data:")
    columns_to_show = ['city', 'state', 'location', 'extracted_location_name', 
                      'extracted_latitude', 'extracted_longitude']
    print(augmented_df[columns_to_show].head())

if __name__ == "__main__":
    main()