import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


CITY_NAMES = [
    "basel",
    "bern",
    "geneva",
    "lausanne",
    "lugano",
    "zurich",
]


def main(city):

    ##################################################
    # Import city footprints

    allcity_roi_path = "./input/footprints/study_areas.geojson"

    allcity_roi_gpd = gpd.read_file(allcity_roi_path)
    allcity_roi_gpd = allcity_roi_gpd.to_crs("EPSG:2056")

    geom = {
        "geneva": allcity_roi_gpd.geometry.iloc[0],
        "lausanne": allcity_roi_gpd.geometry.iloc[1],
        "bern": allcity_roi_gpd.geometry.iloc[2],
        "basel": allcity_roi_gpd.geometry.iloc[3],
        "zurich": allcity_roi_gpd.geometry.iloc[4],
        "lugano": allcity_roi_gpd.geometry.iloc[5],
        "allcity": allcity_roi_gpd.union_all(),
    }

    roi_shapely = geom[city]

    xmin, ymin, xmax, ymax = roi_shapely.bounds

    ##################################################
    # Read COPC URLs

    copc = pd.read_csv(
        "./topo/swissSURFACE3D/swissSURFACE3D_CloudOPC_all.csv",
        header=None,
        names=["url"],
    )

    copc[["E", "N"]] = copc["url"].str.extract(
        r"_(\d{4})-(\d{4})_"
    )

    copc["E"] = copc["E"].astype(int)
    copc["N"] = copc["N"].astype(int)

    ##################################################
    # Read LiDAR URLs

    lidar = pd.read_csv(
        "./topo/swissSURFACE3D/swissSURFACE3D_LiDAR_all.csv",
        header=None,
        names=["url"],
    )

    lidar[["E", "N"]] = lidar["url"].str.extract(
        r"_(\d{4})-(\d{4})_"
    )

    lidar["E"] = lidar["E"].astype(int)
    lidar["N"] = lidar["N"].astype(int)

    ##################################################
    # Generate candidate tiles

    tiles = []

    for e in range(
        int(xmin // 1000),
        int(xmax // 1000) + 1,
    ):
        for n in range(
            int(ymin // 1000),
            int(ymax // 1000) + 1,
        ):
            tiles.append((e, n))

    tile_set = {
        f"{e}-{n}"
        for e, n in tiles
    }

    ##################################################
    # Filter matching URLs

    selected_copc = copc[
        copc["url"].apply(
            lambda x: any(
                tile in x
                for tile in tile_set
            )
        )
    ]

    selected_lidar = lidar[
        lidar["url"].apply(
            lambda x: any(
                tile in x
                for tile in tile_set
            )
        )
    ]

    ##################################################
    # Save URL lists

    outdir = Path(
        "./topo/swissSURFACE3D/extracted"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copc_csv = outdir / f"{city}_copc_urls.csv"
    lidar_csv = outdir / f"{city}_lidar_urls.csv"

    selected_copc.to_csv(
        copc_csv,
        index=False,
    )

    selected_lidar.to_csv(
        lidar_csv,
        index=False,
    )

    print(
        f"Saved {len(selected_copc)} COPC URLs to {copc_csv}"
    )

    print(
        f"Saved {len(selected_lidar)} LiDAR URLs to {lidar_csv}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract swissSURFACE3D URLs for a city."
    )

    parser.add_argument(
        "--city",
        required=True,
        choices=CITY_NAMES,
        help="City name",
    )

    args = parser.parse_args()

    main(args.city)