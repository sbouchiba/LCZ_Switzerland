import argparse
import zipfile
from pathlib import Path

import pandas as pd
import requests

CITY_NAMES = [
    "basel",
    "bern",
    "geneva",
    "lausanne",
    "lugano",
    "zurich",
]

DATASETS = ["copc", "lidar"]

BASE_DIR = Path("D:\BNFproject\topo\extracted")


def extract_zip(zip_path: Path, outdir: Path, delete_after: bool = True):
    """
    Extract a .zip (or .las.zip) file into outdir.
    Writes a marker file on success so future runs can detect this was
    already extracted, even after the zip itself is deleted.
    Deletes the zip afterwards by default to save disk space.
    """
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(outdir)
        print("    extracted")

        # Marker lives next to outdir (not deleted with the zip), named
        # after the zip so it survives even once the zip is removed.
        marker = marker_path_for_outdir(outdir, zip_path.name)
        marker.touch()

        if delete_after:
            zip_path.unlink()
    except zipfile.BadZipFile:
        print(f"    WARNING: {zip_path.name} is not a valid zip file, skipping extraction")


def marker_path_for_outdir(outdir: Path, filename: str) -> Path:
    """Marker file path for a given zip filename, kept in outdir."""
    return outdir / f".{filename}.extracted"


def download_files(city: str, dataset: str, delete_zips: bool = True):
    csv_file = BASE_DIR / f"{city}_{dataset}_urls.csv"
    outdir = BASE_DIR / city

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    outdir.mkdir(parents=True, exist_ok=True)

    urls = pd.read_csv(csv_file)["url"]

    for i, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        outfile = outdir / filename

        print(f"[{i}/{len(urls)}] {filename}")

        # Check if this file was already downloaded AND extracted, using
        # the marker file (works even after the zip itself was deleted,
        # and doesn't depend on guessing the extracted file's name).
        if filename.endswith(".zip") and marker_path_for_outdir(outdir, filename).exists():
            print("    already extracted, skipping")
            continue

        if outfile.exists():
            print("    already downloaded")
        else:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(outfile, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        # Extract if it's a zip (covers both "file.zip" and "file.las.zip")
        if outfile.suffix == ".zip":
            extract_zip(outfile, outdir, delete_after=delete_zips)

    print(f"\nFinished downloading {dataset} files for {city}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download swissSURFACE3D files for a city."
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=CITY_NAMES,
        help="City name",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="Dataset type: copc or lidar",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep the downloaded .zip files after extraction (default: delete them)",
    )
    args = parser.parse_args()

    download_files(args.city, args.dataset, delete_zips=not args.keep_zips)