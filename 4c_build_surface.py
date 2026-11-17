import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pdal
import rasterio

from rasterio.merge import merge
from scipy.ndimage import generic_filter
from tqdm import tqdm


# --------------------------------------------------
# PARAMETERS

CITY_NAMES = [
    "basel",
    "bern",
    "geneva",
    "lausanne",
    "lugano",
    "zurich",
]

RESOLUTION = 10

# Interpolation: window size in pixels (3 = 30m neighbourhood at 10m res)
INTERP_WINDOW = 3

# Alert if more than this fraction of building pixels needed interpolation
INTERP_WARNING_THRESHOLD = 0.05  # 5%


# --------------------------------------------------
# ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument(
    "--city",
    required=True,
    choices=CITY_NAMES,
)

args = parser.parse_args()
city = args.city


# --------------------------------------------------
# CITY FOOTPRINT

allcity_roi_path = "./input/footprints/study_areas.geojson"

allcity_roi_gpd = gpd.read_file(allcity_roi_path)
allcity_roi_gpd = allcity_roi_gpd.to_crs("EPSG:2056")

geom = {
    "geneva":   allcity_roi_gpd.geometry.iloc[0],
    "lausanne": allcity_roi_gpd.geometry.iloc[1],
    "bern":     allcity_roi_gpd.geometry.iloc[2],
    "basel":    allcity_roi_gpd.geometry.iloc[3],
    "zurich":   allcity_roi_gpd.geometry.iloc[4],
    "lugano":   allcity_roi_gpd.geometry.iloc[5],
}

roi = geom[city]
xmin, ymin, xmax, ymax = roi.bounds


# --------------------------------------------------
# PATHS

input_dir = Path(f"./topo/swissSURFACE3D/extracted/{city}")

output_dir = Path("./topo/swissSURFACE3D/geotiff")
output_dir.mkdir(exist_ok=True)

tmp_ground   = output_dir / "tmp_ground"
tmp_building = output_dir / "tmp_building"

tmp_ground.mkdir(exist_ok=True)
tmp_building.mkdir(exist_ok=True)

ground_tif   = output_dir / f"{city}_ground.tif"
building_tif = output_dir / f"{city}_building.tif"
surface_tif  = output_dir / f"{city}_surface.tif"


# --------------------------------------------------
# INPUT FILES

files = sorted(
    f for ext in ("*.copc.laz", "*.las") for f in input_dir.glob(ext)
)

print(f"{len(files)} point-cloud files found")


# --------------------------------------------------
# HELPERS

def get_reader_type(pointcloud_file):
    return (
        "readers.copc" if str(pointcloud_file).endswith(".copc.laz")
        else "readers.las"
    )


def count_points_per_class(pointcloud_file, reader, classes):
    """
    Count points for each classification using one lightweight pipeline per
    class (no writers — avoids the PDAL crash on zero-point writers).
    Returns a dict {class_int: count}.
    """
    counts = {}
    for cls in classes:
        pipeline = [
            {"type": reader, "filename": str(pointcloud_file)},
            {"type": "filters.range", "limits": f"Classification[{cls}:{cls}]"},
        ]
        counts[cls] = pdal.Pipeline(json.dumps(pipeline)).execute()
    return counts


def rasterize_class(pointcloud_file, reader, classification, out_file, output_type):
    """
    Rasterize a single point classification to a GeoTIFF using a simple
    linear pipeline: reader -> range filter -> gdal writer.
    """
    tmp_out = out_file.with_suffix(".tmp.tif")

    pipeline = [
        {"type": reader, "filename": str(pointcloud_file)},
        {
            "type": "filters.range",
            "limits": f"Classification[{classification}:{classification}]",
        },
        {
            "type": "writers.gdal",
            "filename": str(tmp_out),
            "resolution": RESOLUTION,
            "dimension": "Z",
            "output_type": output_type,
            "nodata": -9999,
        },
    ]

    pdal.Pipeline(json.dumps(pipeline)).execute()

    if tmp_out.exists():
        tmp_out.rename(out_file)
        return True

    return False


def rasterize_tile_both(pointcloud_file, ground_out, building_out):
    """
    Process ground (class 2) and building (class 6) for a single tile.

    Uses two separate linear pipelines — one per class — because PDAL's
    Python binding does not support DAG-style branching (tag / inputs).
    The count_points_per_class pre-check ensures writers.gdal never
    receives zero points (which would raise a RuntimeError).

    Returns (ground_ok, building_ok) booleans.
    """
    reader = get_reader_type(pointcloud_file)

    # Pre-check: count points per class (lightweight, no writers)
    counts = count_points_per_class(pointcloud_file, reader, [2, 6])
    has_ground   = counts[2] > 0
    has_building = counts[6] > 0

    if not has_ground:
        tqdm.write(f"  {pointcloud_file.name}: no ground points (class 2)")
    if not has_building:
        tqdm.write(f"  {pointcloud_file.name}: no building points (class 6)")

    if not has_ground and not has_building:
        return False, False

    # Rasterize each class with its own pipeline
    ground_ok   = False
    building_ok = False

    if has_ground:
        ground_ok = rasterize_class(
            pointcloud_file, reader,
            classification=2,
            out_file=ground_out,
            output_type="min",
        )

    if has_building:
        building_ok = rasterize_class(
            pointcloud_file, reader,
            classification=6,
            out_file=building_out,
            output_type="max",
        )

    return ground_ok, building_ok


# --------------------------------------------------
# RASTERIZE ALL TILES

print("\nRasterizing ground + buildings...")

ground_tiles   = []
building_tiles = []

for f in tqdm(files, desc="Tiles"):

    g_out = tmp_ground   / f"{f.stem}.tif"
    b_out = tmp_building / f"{f.stem}.tif"

    # Re-use cached tiles if both already exist
    if g_out.exists() and b_out.exists():
        ground_tiles.append(g_out)
        building_tiles.append(b_out)
        continue

    g_ok, b_ok = rasterize_tile_both(f, g_out, b_out)

    if g_ok:
        ground_tiles.append(g_out)
    if b_ok:
        building_tiles.append(b_out)


# --------------------------------------------------
# SAFETY CHECKS

if len(ground_tiles) == 0:
    raise RuntimeError("No ground tiles generated.")

if len(building_tiles) == 0:
    raise RuntimeError("No building tiles generated.")


# --------------------------------------------------
# MERGE

def merge_tiles(tile_list, out_file):
    """
    Merge a list of single-band GeoTIFFs into one mosaic.
    Files are opened, merged, then all closed in a finally block.
    """
    print(f"\nMerging {len(tile_list)} tiles -> {out_file.name} ...")

    with rasterio.open(tile_list[0]) as ref:
        profile = ref.profile.copy()

    srcs = [rasterio.open(f) for f in tile_list]
    try:
        mosaic, transform = merge(srcs)
    finally:
        for s in srcs:
            s.close()

    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )

    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(mosaic)


merge_tiles(ground_tiles,   ground_tif)
merge_tiles(building_tiles, building_tif)


# --------------------------------------------------
# ALIGN BUILDING RASTER TO GROUND RASTER GRID
#
# Ground and building tiles may not cover exactly the same extent (e.g. a
# tile with no ground points is absent from ground_tiles but present in
# building_tiles, or vice versa). This produces mosaics with different
# shapes, which breaks the pixel-wise arithmetic below.
# Solution: reproject/resample the building mosaic onto the ground grid.

def align_raster_to_reference(src_path, ref_path, out_path):
    """
    Reproject src_path onto the exact grid (CRS, transform, shape) of
    ref_path using nearest-neighbour resampling and write to out_path.
    No-op if src and ref already share the same grid.
    """
    from rasterio.warp import reproject, Resampling

    with rasterio.open(ref_path) as ref:
        ref_profile  = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs      = ref.crs
        ref_shape    = (ref.height, ref.width)

    with rasterio.open(src_path) as src:
        # Check if grids already match
        if (src.height, src.width) == ref_shape and src.transform == ref_transform:
            return  # nothing to do

        print(f"\n[ALIGN] Reprojecting {src_path.name} onto ground grid "
              f"({src.height}x{src.width} -> {ref_shape[0]}x{ref_shape[1]}) ...")

        data = src.read(1).astype(np.float32)
        src_nodata  = src.nodata if src.nodata is not None else -9999.0
        src_transform = src.transform
        src_crs      = src.crs

    dst_data = np.full(ref_shape, src_nodata, dtype=np.float32)

    reproject(
        source=data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        src_nodata=src_nodata,
        dst_nodata=src_nodata,
        resampling=Resampling.nearest,
    )

    ref_profile.update(dtype="float32", count=1, nodata=src_nodata)
    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(dst_data, 1)


align_raster_to_reference(building_tif, ground_tif, building_tif)


# --------------------------------------------------
# HEIGHT COMPUTATION WITH GROUND INTERPOLATION

print("\nComputing height raster...")

with rasterio.open(ground_tif) as src:
    ground  = src.read(1).astype(np.float32)
    profile = src.profile.copy()

with rasterio.open(building_tif) as src:
    building = src.read(1).astype(np.float32)

NODATA = -9999.0
ground[ground     == NODATA] = np.nan
building[building == NODATA] = np.nan

# --------------------------------------------------
# Interpolate missing ground pixels under buildings
#
# Only fills pixels where a building value exists but ground is missing.
# Uses a median filter over a INTERP_WINDOW x INTERP_WINDOW neighbourhood.
# Warns if interpolation is used heavily (threshold: INTERP_WARNING_THRESHOLD).

building_mask  = ~np.isnan(building)
ground_missing =  np.isnan(ground)
needs_interp   = building_mask & ground_missing
n_needs_interp = int(needs_interp.sum())

if n_needs_interp > 0:

    ground_filled = generic_filter(
        ground,
        np.nanmedian,
        size=INTERP_WINDOW,
        mode="nearest",
    )

    ground = np.where(needs_interp, ground_filled, ground)

    still_missing   = building_mask & np.isnan(ground)
    n_still_missing = int(still_missing.sum())
    n_building_pixels = int(building_mask.sum())
    frac = n_needs_interp / max(n_building_pixels, 1)

    print(
        f"\n[INTERPOLATION] {n_needs_interp} ground pixels were missing under buildings "
        f"({frac:.1%} of building pixels). "
        f"Filled by {INTERP_WINDOW}x{INTERP_WINDOW} neighbourhood median."
    )

    if n_still_missing > 0:
        print(
            f"[INTERPOLATION] WARNING: {n_still_missing} pixels could not be filled "
            f"(isolated buildings with no ground neighbours). These will be nodata."
        )

    if frac > INTERP_WARNING_THRESHOLD:
        print(
            f"\n[INTERPOLATION] WARNING: interpolation was used for {frac:.1%} of "
            f"building pixels, exceeding the {INTERP_WARNING_THRESHOLD:.0%} threshold.\n"
            f"  -> Consider using swissALTI3D as the ground reference instead:\n"
            f"     https://www.swisstopo.admin.ch/en/height-model-swissalti3d"
        )

else:
    print("\n[INTERPOLATION] No missing ground pixels under buildings — no interpolation needed.")


# --------------------------------------------------
# Write output (3-band GeoTIFF: ground | building | height)

height = building - ground

profile.update(
    count=3,
    dtype="float32",
    nodata=NODATA,
)

with rasterio.open(surface_tif, "w", **profile) as dst:

    dst.write(np.where(np.isnan(ground),   NODATA, ground).astype(np.float32),   1)
    dst.write(np.where(np.isnan(building), NODATA, building).astype(np.float32), 2)
    dst.write(np.where(np.isnan(height),   NODATA, height).astype(np.float32),   3)

    dst.set_band_description(1, "ground")
    dst.set_band_description(2, "building")
    dst.set_band_description(3, "height")

print(f"\nSaved: {surface_tif}")