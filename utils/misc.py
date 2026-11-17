from pathlib import Path
import numpy as np

def build_image(city, roi, year=2022):
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = startDate.advance(1, 'year')
    emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
            .filter(ee.Filter.date(startDate, endDate)) \
            .filter(ee.Filter.bounds(roi)) \
            .mosaic()
    if topo_flag == True:
        height = load_topo(city)
        return emb.addBands(height)
    else:
        return emb

# -----------------------------------------------
# Comparison with reference numbers (Moix24)

CITY_CODES = {
    "Geneva": "gen",
    "Lausanne": "lau",
    "Basel": "bas",
    "Bern": "ber",
    "Zurich": "zur",
    "Lugano": "lug",
}

def read_accuracy_file(filepath): #filepath = excel file with reference OA, UA, PA of Moix24.

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