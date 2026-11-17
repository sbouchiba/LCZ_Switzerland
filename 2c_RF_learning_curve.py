'''
Cross-city generalization scan (part 1/2): compute OA_train / OA_test per city
==================================================================================

Trains TWO classifiers with your already-tuned hyperparameters (from
2a_RF_opti_geneva):
  - "allcity" : trained on pooled training points from ALL cities
  - "geneva"  : trained on Geneva's training points ONLY

Then, for EACH city (regardless of whether it was used in that regime's
training), evaluates both classifiers against:
  - that city's own TRAINING points (OA_train)
  - that city's own TEST points      (OA_test)

Reading the 4 resulting numbers per city tells you different things:
- allcity -> OA_train(city)  : does the pooled model still fit this city's
                                 own training examples well, or does mixing
                                 in other cities visibly hurt city-specific fit?
- allcity -> OA_test(city)   : real generalization of the pooled model to
                                 this city's held-out points
- geneva  -> OA_train(city)  : for Geneva itself, this is normal in-sample
                                 training accuracy. For OTHER cities, this is
                                 "how well does a Geneva-only model explain
                                 another city's labeled training points" --
                                 an extra out-of-sample check using points
                                 never touched in training.
- geneva  -> OA_test(city)   : the main cross-city transfer question -- how
                                 badly does a single-city model degrade when
                                 applied elsewhere?

NOTE ON class_mapping FIELD NAMES (kept exactly as in your pipeline):
- TRAINING files use column 'macroclass_id', and have the signature class
  (macroclass_id == 1) removed before mapping.
- TESTING files use column 'macroclass' (different name), with no signature
  class removal applied.
This asymmetry is intentional / already validated by you -- kept as-is.
'''

import ee
import geemap
import geopandas as gpd
import pandas as pd

#ee.Authenticate(force=True)
ee.Initialize(project='ee-sbouchiba')  # <- your project id, adjust if needed

LABEL_PROP = "LCZ_class"
PRED_PROP = "classification"

city_names = ['basel', 'bern', 'geneva', 'lausanne', 'lugano', 'zurich']

# ------------------------------------------------------------------------------
## ROI + embeddings (built once, covering all cities, exactly as in your script)

allcity_roi_path = './input/footprints/study_areas.geojson'
allcity_roi_gpd = gpd.read_file(allcity_roi_path)
allcity_roi_gpd = allcity_roi_gpd.to_crs("EPSG:4326")

geom = {
    "geneva":   allcity_roi_gpd.geometry.iloc[0],
    "lausanne": allcity_roi_gpd.geometry.iloc[1],
    "bern":     allcity_roi_gpd.geometry.iloc[2],
    "basel":    allcity_roi_gpd.geometry.iloc[3],
    "zurich":   allcity_roi_gpd.geometry.iloc[4],
    "lugano":   allcity_roi_gpd.geometry.iloc[5],
    "allcity":  allcity_roi_gpd.union_all(),
}

roi = ee.Geometry(geom['allcity'].__geo_interface__)

year = 2022
startDate = ee.Date.fromYMD(year, 1, 1)
endDate = startDate.advance(1, 'year')

embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
embeddingsFiltered = embeddings \
    .filter(ee.Filter.date(startDate, endDate)) \
    .filter(ee.Filter.bounds(roi))
embeddingsImage = embeddingsFiltered.mosaic()

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

# ------------------------------------------------------------------------------
## Per-city loaders (kept faithful to your original field-name/filtering choices)

def load_train_gdf(city):
    gdf = gpd.read_file(f"./input/samples/{city}_2022_training_centroids.geojson")
    gdf["city"] = city
    gdf = gdf[gdf["macroclass_id"] != 1].copy() 
    gdf["LCZ_class"] = gdf["macroclass_id"].map(class_mapping)
    return gdf


def load_test_gdf(city):
    gdf = gpd.read_file(f"./input/samples/{city}_2022_testing_centroids.geojson")
    gdf["city"] = city
    gdf["LCZ_class"] = gdf["macroclass"].map(class_mapping)
    return gdf


print("Loading per-city training/testing centroids...")
train_gdfs = {c: load_train_gdf(c) for c in city_names}
test_gdfs = {c: load_test_gdf(c) for c in city_names}

# 'allcity' training gdf = concatenation of all per-city training gdfs
allcity_train_gdf = gpd.GeoDataFrame(
    pd.concat([train_gdfs[c] for c in city_names], ignore_index=True),
    crs=train_gdfs[city_names[0]].crs
)

# Convert everything to ee.FeatureCollection, then attach the 64 embedding bands
def to_features(gdf):
    fc = geemap.geopandas_to_ee(gdf)
    return embeddingsImage.sampleRegions(collection=fc, properties=[LABEL_PROP], scale=10, geometries=False)

train_features_by_city = {c: to_features(train_gdfs[c]) for c in city_names}
test_features_by_city = {c: to_features(test_gdfs[c]) for c in city_names}

training_by_regime = {
    "allcity": to_features(allcity_train_gdf),
    "geneva": train_features_by_city["geneva"],
}

for c in city_names:
    print(f"  {c}: train={train_features_by_city[c].size().getInfo()}, "
          f"test={test_features_by_city[c].size().getInfo()}")

# ------------------------------------------------------------------------------
## Hyperparameters -- EDIT to match your optimized values from 2a_RF_opti_geneva
# numberOfTrees = ...
# variablesPerSplit = ...
# minLeafPopulation = ...

contributing_cities = {"allcity": set(city_names), "geneva": {"geneva"}}


def oa_from_classified(fc_classified):
    err_matrix = fc_classified.errorMatrix(LABEL_PROP, PRED_PROP)
    return err_matrix.accuracy()


results = []
for regime, train_fc in training_by_regime.items():
    n_train = train_fc.size().getInfo()
    print(f"\nTraining classifier for regime '{regime}' (n={n_train} points)...")

    clf = ee.Classifier.smileRandomForest(
        numberOfTrees=numberOfTrees,
        variablesPerSplit=variablesPerSplit,
        minLeafPopulation=minLeafPopulation,
    ).train(
        features=train_fc,
        classProperty=LABEL_PROP,
        inputProperties=embeddingsImage.bandNames(),
    )

    for city in city_names:
        oa_train = oa_from_classified(train_features_by_city[city].classify(clf)).getInfo()
        oa_test = oa_from_classified(test_features_by_city[city].classify(clf)).getInfo()

        results.append({
            "regime": regime,
            "city": city,
            "is_home_city": city in contributing_cities[regime],
            "OA_train": oa_train,
            "OA_test": oa_test,
        })
        print(f"  [{regime}] {city}: OA_train={oa_train:.4f}  OA_test={oa_test:.4f}")

df = pd.DataFrame(results)
df.to_csv("./tables/cross_city_scan_results.csv", index=False)
print("\nSaved: cross_city_scan_results.csv")
print(df)