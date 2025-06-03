Youtube video link: 

# Sentinel 3-Finland/Russia-Cloud Classification
Using k-means classification to analyse SENTINEL-3 satellite data for cloud classification across Finland.

## A description of the problem to be tackled
 Optical satellite imagery is an indispensable tool for global Earth observation, however the presence of clouds and their associated shadows can create atmospheric obstructions. This obscuring of significant portions of the Earth's surface can lead to data gaps needed for consistent land cover and atmosphere montioring. Therefore, to effectively make use of high resolution optical data obtained by Sentinel 3, it is crucial to develop accurate automated methods for compensating for these interferences. 

Traditional cloud detection often relies on complex physical models such as Fmask or extensive labeled training datasets, which can be time consuming. To address these limtations, this project utilises k-means clsutering, an unsupervised classification method, to classify and mask thick clouds, thin clouds, and cloud shadows within Sentinel-3 OLCI radiance imagery. Unsupervised methods are particularly valuable in remote sensing as they can derive patterns directly from the data without prior labeling, offering a flexible solution for automated processing and a flexible solution for diverse environmental conditions.

To provide a an assessment of the k means clustering algorithms performance in distinguishing distinct environmental features, the K-Means derived algorithm of clouds is compared against a reference mask generated using the Normalized Difference Water Index (NDWI), a widely recognized spectral index for water detection. This comparison offers a quantitative insight into the clustering's ability to accurately differentiate between water and other land cover types, complementing its primary goal of cloud and shadow masking.

## A figure illustrating the remote sensing technique: How does Sentinel-3 work?
![image alt](https://github.com/diahara07/Sentinel3-Finland-CloudClassification/blob/02ae71d6e95e7a3b5f8a3e826aace469a789d232/images/SENTINEL%203%20FIGURE%20AI.png)

References:

EUMETSAT. Sentinel-3 OLCI Level 1 Data Guide. Accessed June 3, 2025. https://user.eumetsat.int/resources/user-guides/sentinel-3-olci-level-1-data-guide.

Copernicus. Sentinel-3 Mission. SentiWiki. Accessed June 3, 2025. https://sentiwiki.copernicus.eu/web/s3-mission.

Sentinel-3 User Handbook, GMES-S3OP-EOPG-TN-13-0001, 2nd September 2013

Sentinel-3, Sentinel Online (esa.int)

University of Calgary (ucalgary.ca) https://www.ucalgary.ca/
Welcome to the International Coastal Altimetry Community 
www.coastalt.eu

## A diagram of the K-means clustering algorithm and its implementation
![image alt](https://github.com/diahara07/Sentinel3-Finland-CloudClassification/blob/82c1404d8fc2fac7a06bdcacdc6523e82947234e/images/K%20means%20image.png)
K-Means clustering is an unsupervised learning algorithm that organizes a dataset into a pre-defined number of k distinct groups. In this project, K=4. The fundamental principle involves strategically placing k centroids (central points) and then assigning each data point to the closest nearby centroid. This process ensures that points within the same cluster are as tightly grouped together as possible. It is useful when  exploring data with unknown categories since it doesn't require prior labeling of different classes. As such, it is a powerful tool for initial data exploration and uncovering natural groupings. This makes it suitable for this project because clouds are a great example of natural groupings. (Unsupervised Learning — GEOL0069 Guide Book, n.d.) (Zhang, Y., & Wu, L. (2012))

## K means clustering can be broken down into the following steps: ##
1. Choosing K: Choose the number of k clusters you want.
2. Centroid Initialization: The initial placement of these k centroids can influence the final clustering outcome.
3. Assignment: Every data point in the dataset is assigned to the nearest centroid. This "nearest" is determined by calculating the squared Euclidean distance between the assigned data point and each centroid.
4. Update: After all points are assigned, the centroids are recalculated. Each centroid moves to the average position (the mean) of all the data points currently assigned to its cluster.
5. Iterate: The assignment and update steps are repeated in a loop. This iterative process continues until the centroids no longer shift significantly, indicating that the clusters have stabilized and the algorithm has converged on a solution.


## Data Overview
![image alt](https://github.com/diahara07/Sentinel3-Finland-CloudClassification/blob/cb02589bfbff763e3311bc67df267abf8deff47e/images/Screenshot%202025-06-03%20000350.png)

As illustrated in the figure above, the Sentinel 3 data encompasses Finland as well as some parts of Russia. The polygon used for data selection can also be seen.

`S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3`

## Data Preparation
## Mount Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```
## Define the base directory for the Sentinel-3 data
```
nc_dir = '/content/drive/MyDrive/FINLAND CLOUD DATA/dataset'
```
### Convert nc datasets to numpy .npy file and stack bands

```
!pip install netCDF4
!pip install basemap

# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import os

def nc2npy(base_dir, file_pattern, subset):
    full_path = os.path.join(base_dir, f"{file_pattern}_{subset}.nc")

    try:
        print(f"Attempting to open: {full_path}") 
        nc = Dataset(full_path)
        print(f"Successfully opened: {full_path}") 
        var_name = f"{file_pattern}_{subset}"
        if var_name not in nc.variables:
             print(f"Variable '{var_name}' not found in {full_path}. Available variables: {list(nc.variables.keys())}")
             raise KeyError(f"Variable '{var_name}' not found in {full_path}")

        nc_var = nc.variables[var_name]
        nc_array = np.array(nc_var)

        # Handle fill values
        if 'unc' in subset:
            nc_array[np.where(nc_array == 255)] = 0
        else:
            if hasattr(nc_var, 'valid_range'):
                valid_min, valid_max = nc_var.valid_range
                nc_array[(nc_array < valid_min) | (nc_array > valid_max)] = 0
            elif hasattr(nc_var, '_FillValue'):
                 fill_value = nc_var._FillValue
                 nc_array[nc_array == fill_value] = 0
            else:
                nc_array[np.where(nc_array == 65535)] = 0

        print(f"Successfully processed data for {var_name}") # Added print statement
        print(f"Stats for {var_name}: Max={nc_array.max()}, Min={nc_array.min()}, Mean={nc_array.mean()}")
        return nc_array

    except FileNotFoundError:
        print(f"Error in nc2npy: File not found at {full_path}")
        raise
    except KeyError as e:
        print(f"Error in nc2npy accessing variable: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in nc2npy: {e}")
        raise

def main(base_dir):
    print(f"Starting main function with base directory: {base_dir}") # Added print statement
    for subset in ["radiance"]:
        res = []
        for i in range(1,22):
            file_pattern = "Oa%02d"%i
            try:
                print(f"Processing band {i} for subset '{subset}'") # Added print statement
                arr = nc2npy(base_dir, file_pattern, subset)
                if arr is not None: # Check if nc2npy returned a valid array
                    res.append(arr)
                    
                    # plt.imsave(f"{file_pattern}_{subset}.jpg", arr, cmap='gray')
                    print(f"Successfully processed and added band {i} to results.") # Added print statement
                else:
                    print(f"nc2npy returned None for band {i}, skipping append.") # Added print statement

            except (FileNotFoundError, KeyError) as e:
                print(f"Caught expected error for {file_pattern}_{subset}: {e}. Skipping.")
                continue
            except Exception as e:
                print(f"Caught unexpected error for {file_pattern}_{subset}: {e}. Skipping this band.")
                continue # Continue to the next iteration even for unexpected errors

        print(f"Finished processing all bands for subset '{subset}'. Number of successful bands: {len(res)}") # Added print statement
        if res:
            try:
                # Ensure all arrays in res have the same shape before stacking
                
                # You can check shapes here: [a.shape for a in res]
                res_stacked = np.stack(res, axis=-1)
                print(f"Successfully stacked data. Shape: {res_stacked.shape}")
                save_path = os.path.join(base_dir, f"{subset}.npy") # Save to the specified directory
                np.save(save_path, res_stacked)
                print(f"Successfully saved {subset}.npy to {save_path}") # Added print statement
            except ValueError as e:
                 print(f"Error stacking arrays: {e}. Ensure all arrays have the same dimensions before stacking.")
            except Exception as e:
                 print(f"An unexpected error occurred during stacking or saving: {e}")

        else:
            print(f"No data processed for subset '{subset}'. Cannot stack or save.")

# Call main with the base directory
main(nc_dir)
```
## Optional (split data in chunks for manual IRIS classification of clouds) 
In this notebook we will be using the k means classification method. Alternatively, you can split the data into 5 chunks at this stage and create a mask in IRIS. Here is the code for that:
```
import os
import netCDF4
import numpy as np
import re



def split_npy(data, num_splits, save_dir, prefix='chunk'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    chunks = np.array_split(data, num_splits, axis=0)


    for i, chunk in enumerate(chunks):
        save_path = os.path.join(save_dir, f"{prefix}_{i+1}.npy")
        np.save(save_path, chunk)
        print(f"Saved chunk {i+1} to {save_path}")

data = np.load('radiance.npy')


split_npy(data, num_splits=5, save_dir='/content/drive/MyDrive/FINLAND CLOUD DATA/FINLAND SENTINEL 3 DATA/S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3', prefix='chunk')
import os
import netCDF4
import numpy as np
import re



def split_npy(data, num_splits, save_dir, prefix='chunk'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    chunks = np.array_split(data, num_splits, axis=0)


    for i, chunk in enumerate(chunks):
        save_path = os.path.join(save_dir, f"{prefix}_{i+1}.npy")
        np.save(save_path, chunk)
        print(f"Saved chunk {i+1} to {save_path}")

data = np.load('radiance.npy')


split_npy(data, num_splits=5, save_dir='/content/drive/MyDrive/dataset', prefix='chunk')
```
Use the following code to print and see the data chunks:
```
import numpy as np
import matplotlib.pyplot as plt

file_dir = '/content/drive/MyDrive/FINLAND CLOUD DATA/FINLAND SENTINEL 3 DATA/S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3'
os.chdir(file_dir)

# load data
data = np.load('radiance.npy')

print(f"Data shape: {data.shape}")

# split data into 5 chunks
chunks = np.array_split(data, 5, axis=0)

for i, chunk in enumerate(chunks):
    print(f'Chunk {i+1} shape: {chunk.shape}')
    plt.imshow(chunk[:, :, 0], cmap='gray')  #
    plt.title(f'Chunk {i+1}')
    plt.show()
```
## Load in the .npy file containing the stacked bands
```
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture # If you want to try GMM
# from skimage.morphology import binary_opening, binary_closing, disk # For post-processing

nc_dir = '/content/drive/MyDrive/FINLAND CLOUD DATA/FINLAND SENTINEL 3 DATA/S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3'
radiance_npy_path = os.path.join(nc_dir, 'radiance.npy')

try:
    stacked_olci_data = np.load(radiance_npy_path)
    print(f"Successfully loaded radiance.npy with shape: {stacked_olci_data.shape}")
except FileNotFoundError:
    print(f"Error: radiance.npy not found at {radiance_npy_path}. Please ensure it was created correctly.")
    exit() # Stop if the file isn't found

height, width, num_bands = stacked_olci_data.shape
```
## Define bands for k-means clustering
```
import numpy as np

selected_band_indices = [
    1,  # Oa02 (412.5 nm)
    2,  # Oa03 (442.5 nm)
    7,  # Oa08 (665 nm)
    11, # Oa12 (753.75 nm)
    14, # Oa15 (767.5 nm)
    16, # Oa17 (865 nm)
    20  # Oa21 (1020 nm)
]

# Extract only these selected bands from full stacked data
features_for_clustering = stacked_olci_data[:, :, selected_band_indices]

# Get the new number of selected bands
num_selected_bands = features_for_clustering.shape[-1] # This will be 7 in this example

# Reshape for clustering (pixels x features)
height, width, _ = features_for_clustering.shape # Update height, width from features_for_clustering
reshaped_data = features_for_clustering.reshape(-1, num_selected_bands)
```
## Normalization 
```
data_min = np.min(reshaped_data)
data_max = np.max(reshaped_data)
if data_max - data_min == 0:
    normalized_data = reshaped_data
else:
    normalized_data = (reshaped_data - data_min) / (data_max - data_min)
print(f"Data normalized to range [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")
```
## Set the number of clusters for K-mean algorithm 
```
n_clusters = 4 # Start with 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robustness
cluster_labels = kmeans.fit_predict(normalized_data)
cluster_map = cluster_labels.reshape(height, width)
print(f"Clustering complete. Cluster map shape: {cluster_map.shape}")
```

## Create RGB composite and visualize K-means Clustering
```
# Input exact band indices from radiance data
red_band_idx = 7  # Example: Oa08
green_band_idx = 4 # Example: Oa05
blue_band_idx = 2 # Example: Oa03

# Create an RGB composite for visualization (normalize for display)
rgb_display = np.stack([
    stacked_olci_data[:, :, red_band_idx],
    stacked_olci_data[:, :, green_band_idx],
    stacked_olci_data[:, :, blue_band_idx]
], axis=-1)

# Simple display normalization (clipping to a reasonable range for visual clarity)
rgb_display = np.clip(rgb_display / np.max(rgb_display) * 1.5, 0, 1) # Adjust multiplier as needed

plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(rgb_display)
plt.title("Original RGB Composite (Oa08, Oa05, Oa03)")
plt.axis('on')

plt.subplot(1, 2, 2)
plt.imshow(cluster_map, cmap='tab10')
plt.title(f"K-Means Clusters (K={n_clusters})")
plt.colorbar(ticks=range(n_clusters), label="Cluster ID")
plt.axis('off')
plt.show()

# After visual inspection, define your cluster IDs:
# cluster_id_thick_cloud = X
# cluster_id_thin_cloud = Y
# cluster_id_cloud_shadow = Z

# Generate masks
```
## Calculate NDWI index 
```
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

# Load your stacked radiance data (from previous steps)
nc_dir = '/content/drive/MyDrive/FINLAND CLOUD DATA/FINLAND SENTINEL 3 DATA/S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3'
radiance_npy_path = os.path.join(nc_dir, 'radiance.npy')

try:
    stacked_olci_data = np.load(radiance_npy_path)
    print(f"Successfully loaded radiance.npy with shape: {stacked_olci_data.shape}")
except FileNotFoundError:
    print(f"Error: radiance.npy not found at {radiance_npy_path}. Please ensure it was created correctly.")
    exit()

# Get dimensions
height, width, num_bands = stacked_olci_data.shape

# --- Calculate NDWI ---
# Ensure these indices match your actual stacked data's band order
# Oa06 (560nm) is typically index 5 if Oa01 is 0.
# Oa17 (865nm) is typically index 16 if Oa01 is 0.
green_band = stacked_olci_data[:, :, 5] # Oa06_radiance
nir_band = stacked_olci_data[:, :, 16] # Oa17_radiance

# Handle potential division by zero or very small denominators
# Add a small epsilon to avoid NaN issues where Green + NIR is zero
epsilon = 1e-8
ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)

# Handle invalid values (e.g., if you have 65535 or similar fill values that weren't handled)
# If your nc2npy conversion set invalid values to 0, NDWI might become NaN.
# Let's assume you've already handled invalid radiance values, but if not,
# you might want to mask them out here before calculating NDWI.
# A common approach is to set NDWI to NaN where input bands are very low (e.g., in fill areas)
# ndwi[stacked_olci_data[:, :, 0] <= 0] = np.nan # Example if 0 is your fill value in a reference band

print(f"NDWI calculated. Min: {np.nanmin(ndwi):.2f}, Max: {np.nanmax(ndwi):.2f}")
```

## Create a water mask from NDWI
```
# --- Create a water mask from NDWI ---
# You'll need to choose this threshold based on visual inspection of your NDWI image
# Common thresholds are 0.0, 0.2, or 0.3. Experiment!
ndwi_threshold = 0.2 # Example threshold

water_mask_ndwi = (ndwi > ndwi_threshold).astype(np.uint8) * 255 # Convert to 255 for visualization

# Optional: Further refinement of the NDWI mask (e.g., morphological operations)
# from skimage.morphology import binary_opening, disk
# kernel = disk(2) # Small kernel
# water_mask_ndwi_cleaned = binary_opening(water_mask_ndwi, kernel).astype(np.uint8) * 255

print(f"NDWI water mask created. Water pixels: {np.sum(water_mask_ndwi > 0)} out of {height * width}")
```
## Visualize NDWI water mask
```
import matplotlib.pyplot as plt
import numpy as np # Make sure numpy is imported if not already

# Assuming 'water_mask_ndwi', 'height', and 'width' are already defined from the previous cell

# --- Visualize the NDWI water mask ---
plt.figure(figsize=(10, 10)) # Adjust figure size as needed
plt.imshow(water_mask_ndwi, cmap='gray') # Use 'gray' colormap for binary mask
plt.title(f"NDWI Water Mask (Threshold > {ndwi_threshold})")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.colorbar(label='Pixel Value (255 for Water, 0 for Non-Water)') # Add a colorbar to show values
plt.axis('on') # Show axes with pixel indices
plt.show()
```
## Compare and plot NDWI, Original Image and K-Means Masks
```
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # New imports for confusion matrix

# --- Re-load your stacked radiance data 
nc_dir = '/content/drive/MyDrive/FINLAND CLOUD DATA/FINLAND SENTINEL 3 DATA/S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3'
radiance_npy_path = os.path.join(nc_dir, 'radiance.npy')

try:
    stacked_olci_data = np.load(radiance_npy_path)
    print(f"Successfully loaded radiance.npy with shape: {stacked_olci_data.shape}")
except FileNotFoundError:
    print(f"Error: radiance.npy not found at {radiance_npy_path}. Please ensure it was created correctly.")
    exit()

height, width, num_bands = stacked_olci_data.shape

# --- 1. Calculate NDWI and create NDWI water mask ---
green_band_idx = 5  # Oa06_radiance (560nm)
nir_band_idx = 16   # Oa17_radiance (865nm)

green_band = stacked_olci_data[:, :, green_band_idx]
nir_band = stacked_olci_data[:, :, nir_band_idx]

epsilon = 1e-8
ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)

ndwi_threshold = 0.2 # Adjust this threshold based on your data and visual inspection
water_mask_ndwi = (ndwi > ndwi_threshold).astype(np.uint8) # Binary mask (0 or 1)
water_mask_ndwi_vis = water_mask_ndwi * 255 # Mask for visualization


# --- 2. Perform K-Means and identify the cloud clusters ---
selected_band_indices = [1, 2, 7, 11, 14, 16, 20] 
features_for_clustering = stacked_olci_data[:, :, selected_band_indices]
reshaped_data = features_for_clustering.reshape(-1, len(selected_band_indices))

data_min = np.min(reshaped_data)
data_max = np.max(reshaped_data)
if data_max - data_min == 0:
    normalized_data = reshaped_data
else:
    normalized_data = (reshaped_data - data_min) / (data_max - data_min)


n_clusters = 4 # Use the same K you used for your main classification
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(normalized_data)
cluster_map = cluster_labels.reshape(height, width)

# Manually identify your K-Means CLOUD cluster ID(s) and replace value

kmeans_cloud_cluster_ids = [1, 2] 

# Create a combined K-Means cloud mask based on the identified cluster IDs
kmeans_cloud_mask = np.zeros_like(cluster_map, dtype=np.uint8)
for cloud_id in kmeans_cloud_cluster_ids:
    kmeans_cloud_mask[cluster_map == cloud_id] = 1 

kmeans_cloud_mask_vis = kmeans_cloud_mask * 255 # Mask for visualization

print(f"K-Means cloud mask created. Cloud pixels: {np.sum(kmeans_cloud_mask > 0)} out of {height * width}")


# --- Visual Comparison ---
plt.figure(figsize=(18, 6))

# Original image for context
red_band_idx = 7
green_band_idx = 4
blue_band_idx = 2
rgb_display = np.stack([
    stacked_olci_data[:, :, red_band_idx],
    stacked_olci_data[:, :, green_band_idx],
    stacked_olci_data[:, :, blue_band_idx]
], axis=-1)
rgb_display = np.clip(rgb_display / np.max(rgb_display) * 1.5, 0, 1)

plt.subplot(1, 3, 1)
plt.imshow(rgb_display)
plt.title("Original Image (RGB)")
plt.axis('on')

plt.subplot(1, 3, 2)
plt.imshow(ndwi, cmap='viridis') # Display NDWI values
plt.colorbar(label='NDWI Value')
plt.title("NDWI Image")
plt.axis('on') # Ensure axis is on for consistent plots

plt.subplot(1, 3, 3)
# Overlay the NDWI water mask (e.g., in blue) and the K-Means cloud mask (e.g., in red)

plt.imshow(water_mask_ndwi_vis, cmap='Blues', alpha=0.7) # Show NDWI water mask
plt.imshow(kmeans_cloud_mask_vis, cmap='Reds', alpha=0.7) # Overlay K-Means cloud mask
plt.title(f"Masks Overlay\n(NDWI Water Blue, K-Means Cloud Red)")
plt.axis('on')

# --- Add Save Functionality for the 3-Panel Plot ---
# Assuming 'save_dir' is defined from a previous cell
comparison_filename = 'ndwi_kmeans_cloud_comparison_plot.png'
comparison_save_path = os.path.join(save_dir, comparison_filename)

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# Save the figure 
plt.savefig(comparison_save_path, bbox_inches='tight')
print(f"Saved comparison plot to: {comparison_save_path}")

# Display the figure 
plt.show()


# --- Side-by-Side Visualization of the two masks ---
# This helps to directly see the spatial distribution of each mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(water_mask_ndwi_vis, cmap='Blues')
plt.title(f"Water Mask (from NDWI > {ndwi_threshold})")
plt.axis('on')

plt.subplot(1, 2, 2)
plt.imshow(kmeans_cloud_mask_vis, cmap='Greens') # Using Green for K-Means Cloud mask
cloud_ids_str = ', '.join(map(str, kmeans_cloud_cluster_ids))
plt.title(f"Cloud Mask (from K-Means Clusters {cloud_ids_str})")
plt.axis('on')

# --- Add Save Functionality for Side-by-Side Plot ---
side_by_side_filename = 'ndwi_water_kmeans_cloud_side_by_side_plot.png'
side_by_side_save_path = os.path.join(save_dir, side_by_side_filename)

# Ensure the save directory exists 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save figure 
plt.savefig(side_by_side_save_path, bbox_inches='tight')
print(f"Saved side-by-side plot to: {side_by_side_save_path}")

# Display the figure (optional)
plt.show()


# Total pixels
total_pixels = height * width

# Pixels classified as Water by NDWI
ndwi_water_pixels = np.sum(water_mask_ndwi) # Since mask is 0 or 1

# Pixels classified as Cloud by K-Means
kmeans_cloud_pixels = np.sum(kmeans_cloud_mask) # Since mask is 0 or 1

# Pixels classified as BOTH Water (NDWI) and Cloud (K-Means) - Expected to be low
water_and_cloud_pixels = np.sum(water_mask_ndwi & kmeans_cloud_mask)

# Pixels classified as Water (NDWI) but NOT Cloud (K-Means)
water_only_ndwi = np.sum(water_mask_ndwi & ~kmeans_cloud_mask.astype(bool)) # Use boolean masks for bitwise AND

# Pixels classified as Cloud (K-Means) but NOT Water (NDWI)
cloud_only_kmeans = np.sum(~water_mask_ndwi.astype(bool) & kmeans_cloud_mask.astype(bool))

# Pixels classified as NEITHER Water (NDWI) NOR Cloud (K-Means) - Likely clear land/other
neither_water_nor_cloud = np.sum(~water_mask_ndwi.astype(bool) & ~kmeans_cloud_mask.astype(bool))


print("\nQuantitative Comparison (Water Mask vs Cloud Mask):")
print(f"Total Pixels: {total_pixels}")
print(f"NDWI Water Pixels: {ndwi_water_pixels} ({ndwi_water_pixels/total_pixels:.2%})")
print(f"K-Means Cloud Pixels: {kmeans_cloud_pixels} ({kmeans_cloud_pixels/total_pixels:.2%})")
print(f"Pixels classified as BOTH Water and Cloud: {water_and_cloud_pixels} ({water_and_cloud_pixels/total_pixels:.2%})")
print(f"Pixels classified as Water ONLY (by NDWI): {water_only_ndwi} ({water_only_ndwi/total_pixels:.2%})")
print(f"Pixels classified as Cloud ONLY (by K-Means): {cloud_only_kmeans} ({cloud_only_kmeans/total_pixels:.2%})")
print(f"Pixels classified as NEITHER Water nor Cloud: {neither_water_nor_cloud} ({neither_water_nor_cloud/total_pixels:.2%})")

# A simple measure of overall agreement (clear/land)
agreement = neither_water_nor_cloud / total_pixels
print(f"Agreement (Neither Water nor Cloud): {agreement:.3f}")

```
## Generate a confusion matrix 
```
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report # Ensure classification_report is imported

# --- Re-load your stacked radiance data (if not already in memory) ---
nc_dir = '/content/drive/MyDrive/FINLAND CLOUD DATA/FINLAND SENTINEL 3 DATA/S3B_OL_1_EFR____20250527T083223_20250527T083523_20250527T114347_0179_107_064_1800_ESA_O_NR_004.SEN3'
radiance_npy_path = os.path.join(nc_dir, 'radiance.npy')

try:
    stacked_olci_data = np.load(radiance_npy_path)
    print(f"Successfully loaded radiance.npy with shape: {stacked_olci_data.shape}")
except FileNotFoundError:
    print(f"Error: radiance.npy not found at {radiance_npy_path}. Please ensure it was created correctly.")
    exit()

height, width, num_bands = stacked_olci_data.shape

# --- 1. Calculate NDWI and create NDWI water mask ---
green_band_idx = 5  # Oa06_radiance (560nm)
nir_band_idx = 16   # Oa17_radiance (865nm)

green_band = stacked_olci_data[:, :, green_band_idx]
nir_band = stacked_olci_data[:, :, nir_band_idx]

epsilon = 1e-8
# Ensure bands are floating point for division
green_band_float = green_band.astype(float)
nir_band_float = nir_band.astype(float)

# Handle potential zeros in denominator gracefully (add epsilon)
denominator = green_band_float + nir_band_float
# Avoid division by zero completely by checking where the denominator is not close to zero
ndwi = np.full_like(green_band_float, np.nan) # Initialize with NaN or another indicator
valid_denominator = np.abs(denominator) > epsilon # Check if denominator is significantly non-zero
ndwi[valid_denominator] = (green_band_float[valid_denominator] - nir_band_float[valid_denominator]) / denominator[valid_denominator]


ndwi_threshold = 0.2 # Adjust this threshold based on your data and visual inspection
# Use the original 0/1 mask for confusion matrix comparison
water_mask_ndwi = (ndwi > ndwi_threshold).astype(np.uint8) # Binary mask (0 or 1)


# --- 2. Perform K-Means and identify the cloud clusters ---
selected_band_indices = [1, 2, 7, 11, 14, 16, 20]
features_for_clustering = stacked_olci_data[:, :, selected_band_indices]
reshaped_data = features_for_clustering.reshape(-1, len(selected_band_indices))

data_min = np.min(reshaped_data)
data_max = np.max(reshaped_data)
if data_max - data_min == 0:
    print("Warning: Data values are uniform. Normalization will result in division by zero or values remaining the same.")
    normalized_data = reshaped_data
else:
    normalized_data = (reshaped_data - data_min) / (data_max - data_min)

n_clusters = 4 # Use the same K you used for your main classification
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

cluster_labels = kmeans.fit_predict(normalized_data)

cluster_map = cluster_labels.reshape(height, width)


# --- Identify the K-Means CLOUD cluster ID(s) ---

kmeans_cloud_cluster_ids = [1, 2] # <--- replace with observed cloud cluster IDs

# Create a combined K-Means cloud mask based on the identified cluster IDs
# Use 0/1 mask for confusion matrix
kmeans_cloud_mask = np.zeros_like(cluster_map, dtype=np.uint8)
for cloud_id in kmeans_cloud_cluster_ids:
    kmeans_cloud_mask[cluster_map == cloud_id] = 1 # Set to 1 for cloud pixels


print(f"K-Means cloud mask (0/1) created. Cloud pixels: {np.sum(kmeans_cloud_mask > 0)} out of {height * width}")
print(f"NDWI water mask (0/1) created. Water pixels: {np.sum(water_mask_ndwi > 0)} out of {height * width}")


# --- Prepare data for Confusion Matrix ---
# Flatten the 2D masks into 1D arrays
# Compare NDWI Water (True labels) vs K-Means Cloud (Predicted labels).
# Define classes: 0 = Not Water/Not Cloud, 1 = Water/Cloud

true_labels = water_mask_ndwi.flatten()
predicted_labels = kmeans_cloud_mask.flatten() # Comparing against the cloud mask

# Filter out potential NaNs in NDWI if they exist
valid_pixels = ~np.isnan(true_labels) # Check if true_labels has NaNs
if np.any(~valid_pixels):
     print(f"Warning: Found {np.sum(~valid_pixels)} NaN values in NDWI mask. Filtering these pixels.")
     true_labels_filtered = true_labels[valid_pixels].astype(int) # Ensure they are integers
     predicted_labels_filtered = predicted_labels[valid_pixels].astype(int) # Ensure they are integers
else:
    true_labels_filtered = true_labels.astype(int)
    predicted_labels_filtered = predicted_labels.astype(int)


# Labels for the matrix: 0 for "Not Water/Not Cloud", 1 for "Water/Cloud"
labels = [0, 1]

# Display labels for the matrix axes
display_pred_labels = ["Non-Cloud (KMeans)", "Cloud (KMeans)"] # Labels for columns (Predicted)


# --- 3. Compute and Plot the Confusion Matrix ---
cm = confusion_matrix(true_labels_filtered, predicted_labels_filtered, labels=labels)

print("\nConfusion Matrix (Rows=NDWI Water, Columns=K-Means Cloud):")
print(cm)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8)) # Adjust size as needed
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_pred_labels) # Use predicted labels for columns
cmp.plot(ax=ax, cmap='Blues', values_format='d') # 'd' for integer format
ax.set_title('Confusion Matrix: NDWI Water vs K-Means Cloud')
ax.set_xlabel('K-Means Cloud Classification')
ax.set_ylabel('NDWI Water Classification') # Label rows based on NDWI (True labels)

# --- Add Save Functionality for the Confusion Matrix Plot ---
# Assuming 'save_dir' is defined from a previous cell
confusion_matrix_filename = 'ndwi_kmeans_cloud_confusion_matrix.png'
confusion_matrix_save_path = os.path.join(save_dir, confusion_matrix_filename)

# Ensure the save directory exists (already done before, but harmless)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# Save the figure 
plt.savefig(confusion_matrix_save_path, bbox_inches='tight')
print(f"Saved confusion matrix plot to: {confusion_matrix_save_path}")

# Display the figure
plt.show()
```
## Assessment of Environmental Cost of Research Project
##Satellite Operations: Sentinel 3 OLCI
This is the most expensive aspect of this project. This is due to the required electronics, metals as well as the energy intesive nature of building such a complex satellite. Sending the Sentinel 3- OLCI into orbit requires a large amount of fuel and releases a large amount of greenhouse gases into the atmosphere, contributing to the greenhouse affect. However, this is compensated by the fact that Sentinel-3 is designed for a 7 year operational lifetime with 120 kg of hydrazine propellant allowing up to 12 years of continuous operations. This upfront cost of buildning the satellite is shared amongst its users.

To process satellite data, large amounts of energy is used for cooling systems and electricity to power data centres.  Processing Sentinel 3 Data requires huge data centres such as (EUMETSAT, ESA, Copernicus) which expend loads of energy on cooling systems. 

## Computational Power (Machine Learning - K-Means)

Running the K-means learning algorithm on large datasets such as stacked_olci_data, increases electricity usage due to requiring computational power from the computer's CPU. The clustering process can be equally computationally intensive for large images and clusters but for this project k=4, which is a moderate amount of centroids. This reduces the complexity of the computations, reducing energy consumption. 

## AI 

The use of AI in an assistive role contributes to electricty and water usage. Large amounts of water and electricity is used to power data centres and water is used to cool the servers. 

To summarise, this research project contributed to energy consumption and gas emissions by tilising satellite data and computational analysis. This can be mitigated by optimizing code to reduce computation time, only using AI when necessary and opting to choose data centres that use more renewable energy sources. 

<!-- ACKNOWLEDGMENTS -->

This project was created for GEOL0069 at University College London, taught by Dr. Michel Tsamados and Weibin Chen.



