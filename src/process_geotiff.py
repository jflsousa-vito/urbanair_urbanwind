from rasterio.windows import from_bounds
from rasterio.transform import Affine
import rasterio
import numpy as np

def fix_geotiff_transform(input_path: str, output_path: str):
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUs"):
        with rasterio.open(input_path) as src:
            tr = src.transform
            H, W = src.height, src.width

            # Flip the data array vertically
            bands = []
            for b in range(1, src.count + 1):
                arr = src.read(b)
                bands.append(arr[::-1, :])  # vertical flip
            bands = np.stack(bands, axis=0)

            # Update transform to preserve the same map extent
            e_prime = -tr.e
            f_prime = tr.f + tr.e * (H - 1)
            tr_fixed = Affine(tr.a, tr.b, tr.c, tr.d, e_prime, f_prime)

            meta = src.meta.copy()
            meta.update(transform=tr_fixed)

            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(bands)
                if src.nodata is not None:
                    dst.nodata = src.nodata
                dst.update_tags(**src.tags())
                for i in range(1, src.count + 1):
                    dst.update_tags(i, **src.tags(i))
                print( f"Fixed GeoTIFF saved to {output_path}")

