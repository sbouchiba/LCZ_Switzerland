import geemap
import ee

#ee.Authenticate(force=True)
ee.Initialize(project='ee-sbouchiba')

## Import cities footprint
import geopandas as gpd
import json

city_names = ['basel', 'bern', 'geneva', 'lausanne', 'lugano', 'zurich']
train_city = 'geneva'
#test_city = 'lausanne'

allcity_roi_path = './input/footprints/study_areas.geojson'

#geneva_roi_gpd  = gpd.read_file(geneva_roi_path)
#geneva_roi_gpd  = geneva_roi_gpd.to_crs("EPSG:4326") # convert projected CRS into WGS84 longitude/latitude.
allcity_roi_gpd = gpd.read_file(allcity_roi_path)
allcity_roi_gpd = allcity_roi_gpd.to_crs("EPSG:4326") # convert projected CRS into WGS84 longitude/latitude.
#allcity_roi_gpd = allcity_roi_gpd.to_crs("EPSG:2056")

#geom = geneva_roi_gpd.geometry.iloc[0]
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
#roi = ee.Geometry(geom['allcity'].__geo_interface__)
roi_fc = ee.FeatureCollection([ee.Feature(roi)])



#train_polygons = ee.FeatureCollection('./input/geneva_2022_training_polygons.geojson');
#test_polygons  = ee.FeatureCollection('./input/geneva_2022_testing_polygons.geojson');
import geopandas as gpd
import pandas as pd


## Training samples
if train_city=='allcity':
    centroid_gdfs = []
    polygon_gdfs  = []
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
    if train_city == 'geneva':
        print("oh yeaaaah")
        train_centroids_gdf = gpd.read_file(f'./input/samples/geneva_2022_training_centroids_relabelled_v1.geojson')
        print(train_centroids_gdf[train_centroids_gdf['id_num']==178])
    else:
        train_centroids_gdf = gpd.read_file(f'./input/samples/{train_city}_2022_training_centroids.geojson')
        print(train_centroids_gdf[train_centroids_gdf['id_num']==178])
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
        
    train_polygons_gdf = gpd.read_file(f'./input/samples/{train_city}_2022_training_polygons.geojson')
    train_centroids_gdf['city'] = train_city
    train_polygons_gdf['city'] = train_city

train_centroids = geemap.geopandas_to_ee(train_centroids_gdf)
train_polygons  = geemap.geopandas_to_ee(train_polygons_gdf)
### Import test 
import numpy as np
import pandas as pd

test_city = 'geneva'

if test_city == 'geneva':
    #test_centroids_gdf = gpd.read_file(f"./input/samples/geneva_2022_training_centroids_relabelled_new_points_v1.geojson")
    test_centroids_gdf = gpd.read_file(f"./input/samples/geneva_2022_training_centroids_relabelled_v1.geojson")
    test_polygons_gdf = gpd.read_file(f"./input/samples/{test_city}_2022_testing_polygons.geojson")
else:
    test_centroids_gdf = gpd.read_file(f"./input/samples/{test_city}_2022_testing_centroids.geojson")
    test_polygons_gdf = gpd.read_file(f"./input/samples/{test_city}_2022_testing_polygons.geojson")
    test_centroids_gdf["LCZ_class"]  = (test_centroids_gdf["macroclass"].map(class_mapping))
    test_polygons_gdf["LCZ_class"]   = (test_polygons_gdf["macroclass"].map(class_mapping))

test_centroids  = geemap.geopandas_to_ee(test_centroids_gdf)
test_polygons   = geemap.geopandas_to_ee(test_polygons_gdf)
topo   = ee.Image("projects/ee-sbouchiba/assets/geneva_surface")
# If you want all 3 bands with nice names:
topo = topo.select(['b1','b2','b3']).rename(['ground_elev','building_elev','building_height'])
height = topo.select('building_height')
ground_elev = topo.select('ground_elev')
building_elev = topo.select('building_elev')

# Replace nodata (-9999) with 0 — means "no building here"


# unmask fills ALL nodata (including areas outside the tiff extent) with 0
height = height.where(height.eq(-9999), 0)
height = height.unmask(0)
ground_elev = ground_elev.where(ground_elev.eq(-9999), 0)
ground_elev = ground_elev.unmask(0)
building_elev = building_elev.where(building_elev.eq(-9999), 0)
building_elev = building_elev.unmask(0)
# Optional: also clip any floating point noise to 0
height = height.max(ee.Image(0))

print(topo.bandNames().getInfo())
print(topo.projection().getInfo())
print(topo.geometry().bounds().getInfo())

# Import embeddings
year = 2022;
startDate = ee.Date.fromYMD(year, 1, 1);
endDate = startDate.advance(1, 'year');

embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
embeddingsFiltered = embeddings\
  .filter(ee.Filter.date(startDate, endDate))\
  .filter(ee.Filter.bounds(roi));

embeddingsImage = embeddingsFiltered.mosaic();
embeddingsImage = embeddingsImage.addBands(height)
#embeddingsImage = embeddingsImage.addBands(ground_elev)
#embeddingsImage = embeddingsImage.addBands(building_elev)

training = embeddingsImage.sampleRegions(
    collection=train_centroids,
    properties=['LCZ_class'],
    scale=10
)

#Train classifier

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

# Get a confusion matrix representing resubstitution accuracy.
train_accuracy = classifier.confusionMatrix()
display('Resubstitution error matrix:', train_accuracy)
display('Training overall accuracy:', train_accuracy.accuracy())

# Get classified image
classified = embeddingsImage.classify(classifier)


validation = classified.sampleRegions(
    collection=test_centroids,
    properties=['LCZ_class'],
    scale=10,
    geometries=True
)

#validation.addBands(height)

