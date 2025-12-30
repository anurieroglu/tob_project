import ee
import requests
import os
import time

# --- ACTION REQUIRED ---
# This script downloads Sentinel-2 images from Google Earth Engine
# using a projected Coordinate Reference System (CRS) for accurate
# area measurements.

def download_single_image(image, region, filepath):
    """Downloads a single ee.Image object using the correct projection."""
    
    # Select RGB bands and scale to surface reflectance
    rgb = image.select(['B4', 'B3', 'B2']).multiply(0.0001)
    
    # Generate the download URL
    url = rgb.getDownloadURL({
        'scale': 10,                 # Resolution in meters
        'crs': 'EPSG:32636',         # UTM Zone 36N for accurate measurements
        'region': region.bounds(),   # Use bounds() for safety with projections
        'format': 'GEO_TIFF'
    })
    
    # Download and save the file
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filepath}: {e}")

def main():
    import ee

    SERVICE_ACCOUNT = 'sapanca@water-466914.iam.gserviceaccount.com'
    KEY_FILE = '/home/b3lab/projects/TOB/water-466914-a4465859bde1.json'

    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials)
    
    # --- Parameters to Configure ---
    
    # 1. Define the Area of Interest (AOI) with longitude/latitude coordinates
    sapanca_region = ee.Geometry.Rectangle([
        30.14,  # Western boundary
        40.69,  # Southern boundary
        30.34,  # Eastern boundary
        40.745  # Northern boundary
    ])
    
    # 2. Set the maximum cloud cover percentage
    CLOUD_FILTER_PERCENTAGE = 1

    # 3. Define the output folder for downloaded images
    folder_suffix = str(CLOUD_FILTER_PERCENTAGE).replace('.', '_')
    output_folder = f'/home/b3lab/projects/TOB/sapanca_collection_UTM_{folder_suffix}'
    
    # 4. Set the start and end dates for the image search
    START_DATE = '2022-01-01'
    END_DATE = '2025-07-24'
    


    # --- Script Execution ---
    
    os.makedirs(output_folder, exist_ok=True)

    # Get the image collection from GEE
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(sapanca_region) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER_PERCENTAGE))

    # Get the list of images to process from the GEE server
    image_list = collection.toList(collection.size())
    num_images = image_list.size().getInfo()
    print(f"Found {num_images} images to download with less than {CLOUD_FILTER_PERCENTAGE}% cloud cover.")

    # Loop through the list and download each image
    for i in range(num_images):
        # Get the image object from the list
        image = ee.Image(image_list.get(i))
        
        # Get the unique image ID to use for the filename
        image_id = image.id().getInfo()
        
        # Create a unique, clean filename
        filename = f"{image_id.replace('/', '_')}.tif"
        filepath = os.path.join(output_folder, filename)
        
        print(f"\n--- Downloading image {i+1} of {num_images} ---")
        
        # Check if the file already exists to avoid re-downloading
        if not os.path.exists(filepath):
            download_single_image(image, sapanca_region, filepath)
            # A small delay to avoid overwhelming the server
            time.sleep(2) 
        else:
            print(f"File already exists, skipping: {filepath}")

    print("\nAll downloads complete.")

# Main execution block
if __name__ == "__main__":
    main()