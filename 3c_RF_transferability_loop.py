import geemap
import ee

#ee.Authenticate(force=True)
ee.Initialize(project='ee-sbouchiba')

## Import Geneva footprint
import geopandas as gpd
import json

###
# User input:
# city names: 'basel', 'bern', 'geneva', 'lausanne', 'lugano', 'zurich', 'allcity'
train_city = 'allcity'
test_city = 'lausanne'
###

##################################
### IMPORT FOOTPRINTS AND TRAIN DATASETS
##################################

allcity_roi_path = './input/footprints/study_areas.geojson'

geneva_roi_gpd  = gpd.read_file(geneva_roi_path)
geneva_roi_gpd  = geneva_roi_gpd.to_crs("EPSG:4326") # convert projected CRS into WGS84 longitude/latitude.
allcity_roi_gpd = gpd.read_file(allcity_roi_path)
allcity_roi_gpd = allcity_roi_gpd.to_crs("EPSG:4326") # convert projected CRS into WGS84 longitude/latitude.


#geom = geneva_roi_gpd.geometry.iloc[0]
city_names = ['basel', 'bern', 'geneva', 'lausanne', 'lugano', 'zurich']
geom = {
    "geneva":   allcity_roi_gpd.geometry.iloc[0],
    "lausanne": allcity_roi_gpd.geometry.iloc[1],
    "bern":     allcity_roi_gpd.geometry.iloc[2],
    "basel":    allcity_roi_gpd.geometry.iloc[3],
    "zurich":   allcity_roi_gpd.geometry.iloc[4],
    "lugano":   allcity_roi_gpd.geometry.iloc[5],
    "allcity":  allcity_roi_gpd.union_all(),
}

roi = ee.Geometry(geom[train_city].__geo_interface__)
roi_fc = ee.FeatureCollection([ee.Feature(roi)])

## Import train/test datasets

#train_polygons = ee.FeatureCollection('./input/geneva_2022_training_polygons.geojson');
#test_polygons  = ee.FeatureCollection('./input/geneva_2022_testing_polygons.geojson');
import geopandas as gpd
import pandas as pd



## Training samples
centroid_gdfs = []
polygon_gdfs  = []

if train_city=='allcity':
    for city in city_names:
        centroid_gdf = gpd.read_file(
            f"./input/samples/{city}_2022_training_centroids.geojson"
        )
        polygon_gdf = gpd.read_file(
            f"./input/samples/{city}_2022_training_polygons.geojson"
        )
    
        # Optional: keep track of origin city
        centroid_gdf["city"] = city
        polygon_gdf["city"] = city
    
        centroid_gdfs.append(centroid_gdf)
        polygon_gdfs.append(polygon_gdf)
    
    # Merge all cities
    train_centroids_gdf = gpd.GeoDataFrame(
        pd.concat(centroid_gdfs, ignore_index=True),
        crs=centroid_gdfs[0].crs
    )
    
    train_polygons_gdf = gpd.GeoDataFrame(
        pd.concat(polygon_gdfs, ignore_index=True),
        crs=polygon_gdfs[0].crs
    )

else:
    train_centroids_gdf = gpd.read_file(f"./input/samples/{train_city}_2022_testing_centroids.geojson")
    train_polygons_gdf = gpd.read_file(f"./input/samples/{train_city}_2022_testing_polygons.geojson")

# Remove signature class for test datasets
train_centroids_gdf = (train_centroids_gdf[train_centroids_gdf["macroclass_id"] != 1].copy())
train_polygons_gdf = (train_polygons_gdf[train_polygons_gdf["macroclass_id"] != 1].copy())

# Re-map: define new class mapping that is simpler to handle
class_mapping = {
    2: 1,   # Compact MR
    5: 2,   # Open MR
    6: 3,   # Open LR
    8: 4,   # Large LR
    11: 5,  # Dense Trees
    14: 6,  # Low plants
    15: 7,  # Bare Rock/Paved
    17: 8   # Water
}

# Some data re-arrangement
train_centroids_gdf["LCZ_class"] = (train_centroids_gdf["macroclass_id"].map(class_mapping))
train_polygons_gdf["LCZ_class"]  = (train_polygons_gdf["macroclass_id"].map(class_mapping))

# Convert to EE datasets
train_centroids = geemap.geopandas_to_ee(train_centroids_gdf)
train_polygons  = geemap.geopandas_to_ee(train_polygons_gdf)

# Import embeddings
year = 2022;
startDate = ee.Date.fromYMD(year, 1, 1);
endDate = startDate.advance(1, 'year');

embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
embeddingsFiltered = embeddings\
  .filter(ee.Filter.date(startDate, endDate))\
  .filter(ee.Filter.bounds(roi));

embeddingsImage = embeddingsFiltered.mosaic();
#embeddingsImage = embeddings.mosaic();

#Overlay the samples on the image to get training data.
training = embeddingsImage.sampleRegions(collection = train_polygons, properties=['LCZ_class'], scale=10)

##################################
### TRAIN CLASSIFIER
##################################
# Used best Random Forest (optimised on Geneva -> Geneva)
numberOfTrees = 500
variablesPerSplit = 8
minLeafPopulation = 5

#classifier = ee.Classifier.smileGradientTreeBoost(n_trees).train(
classifier = ee.Classifier.smileRandomForest(numberOfTrees=numberOfTrees,
                                                         variablesPerSplit = variablesPerSplit,
                                                         minLeafPopulation = minLeafPopulation).\
                                        train(features=training, 
                                        classProperty='LCZ_class', 
                                        inputProperties=embeddingsImage.bandNames())

# Classify the input imagery.
classified = embeddingsImage.classify(classifier)

##################################
### VALIDATION
##################################

import numpy as np
import pandas as pd

test_city = 'bern'

test_centroids_gdf = gpd.read_file(f"./input/samples/{test_city}_2022_testing_centroids.geojson")
test_polygons_gdf = gpd.read_file(f"./input/samples/{test_city}_2022_testing_polygons.geojson")

test_centroids_gdf["LCZ_class"]  = (test_centroids_gdf["macroclass"].map(class_mapping))
test_polygons_gdf["LCZ_class"]   = (test_polygons_gdf["macroclass"].map(class_mapping))

test_centroids  = geemap.geopandas_to_ee(test_centroids_gdf)
test_polygons   = geemap.geopandas_to_ee(test_polygons_gdf)

validation = classified.sampleRegions(
    collection=test_centroids,
    properties=['LCZ_class'],
    scale=10
)



##################################
## Compute validation metrics and save in Excel file

labels = [
    "Compact MR",
    "Open MR",
    "Open LR",
    "Large LR",
    "Dense Trees",
    "Low Plants",
    "Bare/Paved",
    "Water"
]

# Get confusion matrix:
confusion_matrix = validation.errorMatrix(
    'LCZ_class',
    'classification'
)
cm = np.array(confusion_matrix.getInfo())
# EE internally allocates first row and first column of zeroes for some reason -> remove them
cm = cm[1:, 1:]

diag = np.diag(cm)
PA = 100 * diag / cm.sum(axis=1)
UA = 100 * diag / cm.sum(axis=0)
OA = 100 * diag.sum() / cm.sum()

# Convert to pandas dataframe
df = pd.DataFrame(cm,index=labels,columns=labels)

df["PA (%)"] = np.round(PA, 1)

ua_row = list(np.round(UA, 1))
ua_row.append(np.round(OA, 1))
df.loc["UA (%)"] = ua_row

import openpyxl
df.to_excel(f"./tables/CM_{train_city}_{test_city}_RF_{numberOfTrees}_{variablesPerSplit}_{minLeafPopulation}.xlsx")

##################################
### IMPORT REFERENCE NUMBERS (LCZ paper 2024)
##################################

from pathlib import Path
import numpy as np

CITY_CODES = {
    "Geneva": "gen",
    "Lausanne": "lau",
    "Basel": "bas",
    "Bern": "ber",
    "Zurich": "zur",
    "Lugano": "lug",
}


def read_accuracy_file(filepath):

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    pa = None
    ua = None
    oa = None

    for line in lines:

        if line.startswith("PA [%]"):
            pa = np.array([
                float(x)
                for x in line.strip().split(",")[1:]
                if x.strip()
            ])

        elif line.startswith("UA [%]"):
            ua = np.array([
                float(x)
                for x in line.strip().split(",")[1:]
                if x.strip()
            ])

        elif "Overall accuracy" in line:
            # extract first number appearing in the line
            import re
            oa = float(re.search(r"(\d+\.\d+)", line).group(1))

    if pa is None:
        raise ValueError(f"PA row not found in {filepath}")

    if ua is None:
        raise ValueError(f"UA row not found in {filepath}")

    if oa is None:
        raise ValueError(f"OA row not found in {filepath}")

    return {
        "PA": pa,
        "UA": ua,
        "OA": oa,
    }


def load_accuracy_data(root_dir):
    data = {}

    for city, code in CITY_CODES.items():

        data[city] = {}

        for sensor in ["S2", "L8"]:
            if code=='lau':
                filepath = (Path(root_dir)/city/f"Accuracy_{code}_{sensor}_2022.csv")
            else:
                filepath = (Path(root_dir)/city/f"Accuracy_{sensor}_{code}_2022.csv")

            data[city][sensor] = read_accuracy_file(filepath)

    return data

ROOT_DIR = '/home/bouchiba/notebooks/project/tables/LCZ_paper_2024/'
ref_data = load_accuracy_data(ROOT_DIR)


##################################
### IMPORT REFERENCE NUMBERS (LCZ paper 2024)
##################################