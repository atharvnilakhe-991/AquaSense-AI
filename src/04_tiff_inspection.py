import os
import numpy as np
import rasterio

print("=" * 70)
print("AquaSense AI - TIFF DATA INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

tiff_dir = os.path.join(
    project_root,
    "data",
    "raw",
    "groundwater"
)

print("\nTIFF DIRECTORY")
print("-" * 70)
print(tiff_dir)

# ------------------------------------------------------------
# 2. FIND TIFF FILES
# ------------------------------------------------------------

tiff_files = sorted(
    [
        file
        for file in os.listdir(tiff_dir)
        if file.lower().endswith(".tif")
    ]
)

print("\nAVAILABLE TIFF FILES")
print("-" * 70)

for file in tiff_files:
    print(" -", file)

print("\nTotal TIFF files:", len(tiff_files))

# ------------------------------------------------------------
# 3. INSPECT EACH TIFF
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TIFF INFORMATION")
print("=" * 70)

for file in tiff_files:

    filepath = os.path.join(tiff_dir, file)

    print("\n" + "-" * 70)
    print(file)
    print("-" * 70)

    try:

        with rasterio.open(filepath) as src:

            print("Width          :", src.width)
            print("Height         :", src.height)
            print("Bands          :", src.count)
            print("Data type      :", src.dtypes)
            print("CRS            :", src.crs)
            print("Resolution     :", src.res)
            print("Bounds         :", src.bounds)
            print("NoData value   :", src.nodata)

            # Read first band
            data = src.read(1)

            # Remove NoData values
            if src.nodata is not None:

                valid_data = data[data != src.nodata]

            else:

                valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:

                print("Minimum value  :", np.min(valid_data))
                print("Maximum value  :", np.max(valid_data))
                print("Mean value     :", np.mean(valid_data))
                print("Median value   :", np.median(valid_data))
                print("Valid pixels   :", len(valid_data))

            else:

                print("No valid pixels found.")

    except Exception as e:

        print("ERROR:", e)

print("\n" + "=" * 70)
print("TIFF INSPECTION COMPLETED")
print("=" * 70)