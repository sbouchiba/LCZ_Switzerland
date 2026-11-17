"""
Random Forest LCZ classification: train-city -> test-city transferability grid.

For every combination of (train_city, test_city) in CITY_NAMES:
  - Train a Random Forest classifier on the train_city's training samples
    (trained once per train_city, then reused for every test_city -> only
    N_cities classifier trainings, not N_cities^2)
  - Classify the Alpha Earth embedding image
  - Validate the classification on test_city's testing samples
  - Compute OA (scalar), PA (per class), UA (per class), and the raw
    confusion matrix

Results are stored in a nested dict and pickled to disk:
    results[train_city][test_city] = {
        "OA": float,
        "PA": np.ndarray (n_classes,),
        "UA": np.ndarray (n_classes,),
        "confusion_matrix": np.ndarray (n_classes, n_classes),
    }

Run with:
    python run_rf_city_city.py
"""

import pickle
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np

# ----------------------------------------------------------------------------
# Configuration - edit these to match your setup
# ----------------------------------------------------------------------------

EE_PROJECT = "ee-sbouchiba"

CITY_NAMES = ["basel", "bern", "geneva", "lausanne", "lugano", "zurich"]

# Raw macroclass_id -> compact 1..8 LCZ_class id used throughout
CLASS_MAPPING = {
    2: 1,   # Compact MR
    5: 2,   # Open MR
    6: 3,   # Open LR
    8: 4,   # Large LR
    11: 5,  # Dense Trees
    14: 6,  # Low plants
    15: 7,  # Bare Rock/Paved
    17: 8,  # Water
}
CLASS_LABELS = {
    1: "Compact MR",
    2: "Open MR",
    3: "Open LR",
    4: "Large LR",
    5: "Dense Trees",
    6: "Low plants",
    7: "Bare/Paved",
    8: "Water",
}
N_CLASSES = len(CLASS_LABELS)

# Best RF hyperparameters (from 2_RF_opti)
NUMBER_OF_TREES = 500
VARIABLES_PER_SPLIT = 8
MIN_LEAF_POPULATION = 5

YEAR = 2022

INPUT_DIR = Path("./input")
OUTPUT_PICKLE = Path("./tables/city_city_results.pkl")


# ----------------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------------

def load_roi():
    """Union footprint of all study areas, used to bound the embedding mosaic."""
    footprint_path = INPUT_DIR / "footprints" / "study_areas.geojson"
    gdf = gpd.read_file(footprint_path).to_crs("EPSG:4326")
    union_geom = gdf.union_all()
    return ee.Geometry(union_geom.__geo_interface__)


def get_embeddings_image(roi, year=YEAR):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    coll = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filter(ee.Filter.date(start, end))
        .filter(ee.Filter.bounds(roi))
    )
    return coll.mosaic()


def load_training_centroids(city):
    """Training centroids for one city, remapped to compact LCZ_class ids."""
    path = INPUT_DIR / "samples" / f"{city}_2022_training_centroids.geojson"
    gdf = gpd.read_file(path)
    gdf = gdf[gdf["macroclass_id"] != 1].copy()  # drop the "signature" class
    gdf["LCZ_class"] = gdf["macroclass_id"].map(CLASS_MAPPING)
    gdf["city"] = city
    return geemap.geopandas_to_ee(gdf)


def load_testing_centroids(city):
    """Testing centroids for one city, remapped to compact LCZ_class ids."""
    path = INPUT_DIR / "samples" / f"{city}_2022_testing_centroids.geojson"
    gdf = gpd.read_file(path)
    gdf["LCZ_class"] = gdf["macroclass"].map(CLASS_MAPPING)
    fc = geemap.geopandas_to_ee(gdf)
    fc = fc.map(lambda f: f.set("point_id", f.get("system:index")))
    return fc


# ----------------------------------------------------------------------------
# Model + evaluation
# ----------------------------------------------------------------------------

def train_classifier(train_centroids_fc, embeddings_image):
    training = embeddings_image.sampleRegions(
        collection=train_centroids_fc, properties=["LCZ_class"], scale=10
    )
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=NUMBER_OF_TREES,
        variablesPerSplit=VARIABLES_PER_SPLIT,
        minLeafPopulation=MIN_LEAF_POPULATION,
    ).train(
        features=training,
        classProperty="LCZ_class",
        inputProperties=embeddings_image.bandNames(),
    )
    return classifier


def evaluate(classified_image, test_centroids_fc):
    """Sample the classified map at test points and compute OA / PA / UA."""
    validation = classified_image.sampleRegions(
        collection=test_centroids_fc,
        properties=["LCZ_class", "point_id"],
        scale=10,
        geometries=False,
    )

    error_matrix = validation.errorMatrix("LCZ_class", "classification")
    cm = np.array(error_matrix.getInfo())
    # EE always allocates a leading zero row/col -> drop it
    cm = cm[1:, 1:]

    # Pad in case a class is entirely absent from this test city (keeps
    # matrices comparable / same shape across all city pairs)
    if cm.shape[0] < N_CLASSES:
        padded = np.zeros((N_CLASSES, N_CLASSES))
        padded[: cm.shape[0], : cm.shape[1]] = cm
        cm = padded

    diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = 100 * diag / cm.sum(axis=1)  # recall, per true class (row-wise)
        ua = 100 * diag / cm.sum(axis=0)  # precision, per predicted class (col-wise)
    oa = 100 * diag.sum() / cm.sum()

    return {"OA": oa, "PA": pa, "UA": ua, "confusion_matrix": cm}


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main():
    ee.Initialize(project=EE_PROJECT)

    roi = load_roi()
    embeddings_image = get_embeddings_image(roi)

    results = {}

    for train_city in CITY_NAMES:
        print(f"\n=== Training on {train_city} ===")
        train_centroids_fc = load_training_centroids(train_city)
        classifier = train_classifier(train_centroids_fc, embeddings_image)
        classified = embeddings_image.classify(classifier)

        results[train_city] = {}

        for test_city in CITY_NAMES:
            test_centroids_fc = load_testing_centroids(test_city)
            metrics = evaluate(classified, test_centroids_fc)
            results[train_city][test_city] = metrics
            print(f"  train={train_city:9s} test={test_city:9s} OA={metrics['OA']:5.1f}%")

    OUTPUT_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(
            {
                "results": results,
                "class_labels": CLASS_LABELS,
                "city_names": CITY_NAMES,
                "params": {
                    "numberOfTrees": NUMBER_OF_TREES,
                    "variablesPerSplit": VARIABLES_PER_SPLIT,
                    "minLeafPopulation": MIN_LEAF_POPULATION,
                },
            },
            f,
        )
    print(f"\nSaved results to {OUTPUT_PICKLE.resolve()}")


if __name__ == "__main__":
    main()
