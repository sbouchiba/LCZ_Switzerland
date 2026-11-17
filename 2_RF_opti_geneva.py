import geemap
import ee
import geopandas as gpd
import json
import numpy as np
import pandas as pd

#ee.Authenticate(force=True)
ee.Initialize(project='ee-sbouchiba')

## Import Geneva footprint
geneva_roi_gpd = gpd.read_file('./input/geneva_footprint.geojson')
#print(geneva_roi_gpd)
#print(geneva_roi_gpd.crs)
geneva_roi_gpd = geneva_roi_gpd.to_crs("EPSG:4326") # convert projected CRS into WGS84 longitude/latitude.


geom = geneva_roi_gpd.geometry.iloc[0]
roi = ee.Geometry(geom.__geo_interface__)

roi_fc = ee.FeatureCollection([ee.Feature(roi)])


## Import train/test datasets
train_centroids_gdf = gpd.read_file('./input/geneva_2022_training_centroids.geojson')
train_polygons_gdf = gpd.read_file('./input/geneva_2022_training_polygons.geojson')
test_centroids_gdf = gpd.read_file('./input/geneva_2022_testing_centroids.geojson')
test_polygons_gdf = gpd.read_file('./input/geneva_2022_testing_polygons.geojson')

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
train_centroids_gdf["LCZ_class"] = (train_centroids_gdf["macroclass_id"].map(class_mapping))
train_polygons_gdf["LCZ_class"]  = (train_polygons_gdf["macroclass_id"].map(class_mapping))
test_centroids_gdf["LCZ_class"]  = (test_centroids_gdf["macroclass"].map(class_mapping))
test_polygons_gdf["LCZ_class"]   = (test_polygons_gdf["macroclass"].map(class_mapping))

# Convert to EE datasets
train_centroids = geemap.geopandas_to_ee(train_centroids_gdf)
train_polygons  = geemap.geopandas_to_ee(train_polygons_gdf)
test_centroids  = geemap.geopandas_to_ee(test_centroids_gdf)
test_polygons   = geemap.geopandas_to_ee(test_polygons_gdf)

#Get embedding vector for 2022
year = 2022;
startDate = ee.Date.fromYMD(year, 1, 1);
endDate = startDate.advance(1, 'year');

embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
embeddingsFiltered = embeddings\
  .filter(ee.Filter.date(startDate, endDate))\
  .filter(ee.Filter.bounds(roi));

embeddingsImage = embeddingsFiltered.mosaic();
training = embeddingsImage.sampleRegions(collection = train_centroids, 
                                         properties=['LCZ_class'], 
                                         scale=10)

# Run the optimization scan
numberOfTrees_list=[100,300,500];
variablesPerSplit_list=[4, 8, 16, 32];
minLeafPopulation_list=[1, 5, 10]

#numberOfTrees_list=[10,20];
#variablesPerSplit_list=[8];
#minLeafPopulation_list=[5]

#exportFolder = './assets/bouchiba/rf/';

results = []

for i in range(len(numberOfTrees_list)):
    for j in range(len(variablesPerSplit_list)):
        for k in range(len(minLeafPopulation_list)):
            numberOfTrees     = numberOfTrees_list[i]
            variablesPerSplit = variablesPerSplit_list[j]
            minLeafPopulation = minLeafPopulation_list[k]
            setup = f"({numberOfTrees},{variablesPerSplit},{minLeafPopulation})"
            
            print(f"--------")
            print(f"Processing classifier setup: {setup}")
            classifier = ee.Classifier.smileRandomForest(numberOfTrees=numberOfTrees,
                                                         variablesPerSplit = variablesPerSplit,
                                                         minLeafPopulation = minLeafPopulation).\
                                        train(features=training, 
                                        classProperty='LCZ_class', 
                                        inputProperties=embeddingsImage.bandNames())
            # Compute and save classified image
            print(f"Compute and save classified image..")
            classified = embeddingsImage.classify(classifier)

            # Validation set metrics
            print(f"Computation validation metrics..")
            validation_set = classified.sampleRegions(collection=test_centroids,
                                                      properties=['LCZ_class'],
                                                      scale=10)
            confusion_matrix = validation_set.errorMatrix('LCZ_class','classification')
            # EE built in producersAccuracy().getInfo() and consumersAccuracy().getInfo()
            # are in an annoying nested list form -> I redefine myself. I checked they are
            # coherent and giving the same results.
            
            cm = np.array(confusion_matrix.getInfo())
            # EE internally allocates first row and first column of zeroes for some reason.-> remove them
            cm = cm[1:, 1:]
            diag = np.diag(cm)
            PA = 100 * diag / cm.sum(axis=1)
            UA = 100 * diag / cm.sum(axis=0)
            OA = 100 * diag.sum() / cm.sum()
            # Training set (check for overfitting)
            training_set = classified.sampleRegions(collection=train_centroids,
                                                    properties=['LCZ_class'],
                                                    scale=10)

            confusion_matrix_train = training_set.errorMatrix('LCZ_class','classification')
            cmt = np.array(confusion_matrix_train.getInfo())
            # EE internally allocates first row and first column of zeroes for some reason.-> remove them
            cmt = cmt[1:, 1:]
            diag = np.diag(cmt)
            PAt = 100 * diag / cmt.sum(axis=1)
            UAt = 100 * diag / cmt.sum(axis=0)
            OAt = 100 * diag.sum() / cmt.sum()
            
            results.append({
                "numberOfTrees": numberOfTrees,
                "variablesPerSplit": variablesPerSplit,
                "minLeafPopulation": minLeafPopulation,
                "OA_train": OAt,
                "UA_train": UAt,
                "PA_train": PAt,
                "OA": OA,
                "UA": UA,
                "PA": PA
            })
            print(f"Metrics computed and appened to results dataframe!")
print("-------")
print("Done!")

# Convert to pandas df and export in pickle file
df = pd.DataFrame(results)
#df.to_pickle("./tables/RF_opti_Geneva2022_all.pkl")