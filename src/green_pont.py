import pandas as pd
import json
import re
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import s3fs

from shapely.geometry import Polygon

from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.mask import mask as mask_raster
import os

def antwerp_green_map(cf):
    """
    How it was calculated in antwerp project
    """

    wind_comfort = cf["maps"]["wind_comfort"]
    air_quality = cf["maps"]["air_quality"]
    wbgt = cf["maps"]["wbgt"]

    # Resaample tiff files:
    ref_map = air_quality
    wind_comfort_resampled = (
        cf["green_potential"]["output_folder"] + "wind_comfort_resampled.tif"
    )
    resample_to_match(
        src_path=wind_comfort,
        ref_path=ref_map,
        out_path=wind_comfort_resampled,
        resampling="nearest",
    )

    maps = [wind_comfort_resampled, air_quality]

    names = ["comfort", "aq"]
    criteria = [35, 2]

    data_all = {}
    i = 0
    for tiff_path in maps:
        data_all[names[i]], xs, ys, meta = read_tiff_with_coords(tiff_path, cf)
        i = i + 1

    data_all["recomendation"] = data_all["aq"] * 0
    data_all["recomendation"][(data_all["aq"] > criteria[0])] = 1  # Not recomended
    data_all["recomendation"][
        ((data_all["aq"] < criteria[0]) & (data_all["comfort"] > criteria[1]))
    ] = 2  # recomended for wind

    result = data_all["recomendation"]
    output_path = cf["green_potential"]["output_folder"] + "recomendations_test.tif"
    save_raster_file(output_path, result, meta, indexes=1)

def create_green_map(cf, method, aq_weight=1, comfort_weight=1, heat_weight=1):
    """
        Create green potential map based on air quality, wind comfort and heat stress maps
        method 1: simple thresholding
        method 2: health risk increase calculation based on literature
        method 3: scoring system based on thresholds
        
    """ 
    
    wind_comfort=cf['maps']['wind_comfort']
    air_quality=cf['maps']['air_quality']
    wbgt=cf['maps']['wbgt']

    out_folder = cf['green_potential']['output_folder']
    if not "s3." in out_folder and not os.path.isdir(out_folder): 
       os.makedirs(out_folders)
    
    # Resaaample tiff files:
    ref_map=air_quality
    wind_comfort_resampled=cf['green_potential']['output_folder']+"wind_comfort_resampled.tif"
    if not file_exists(wind_comfort_resampled):
        resample_to_match(cf, src_path=wind_comfort, ref_path=ref_map, out_path=wind_comfort_resampled, resampling = "nearest")
    
    wbgt_resampled=cf['green_potential']['output_folder']+"wbgt_max_resampled.tif"
    if not file_exists(wbgt_resampled): 
        resample_to_match(cf, src_path=wbgt, ref_path=ref_map, out_path=wbgt_resampled, resampling = "nearest")
    
    maps=[air_quality,wind_comfort_resampled,wbgt_resampled]
    names = ["aq", "comfort", "wbgt"]
    criteria = [35, 1, 25]

    thresholds = {
        "aq": cf["green_potential"]["aq_thresholds"],
        "comfort": cf["green_potential"]["wind_comfort_thresholds"],
        "wbgt": cf["green_potential"]["wbgt_thresholds"],
    }

    data_all = {}
    i = 0
    for tiff_path in maps:
        data_all[names[i]], xs, ys, meta = read_tiff_with_coords(tiff_path)
        i = i + 1

    if method == 1:
        data_all["recomendation"] = data_all["aq"] * 0
        data_all["recomendation"][(data_all["aq"] > criteria[0])] = 1  # Not recomended
        data_all["recomendation"][
            ((data_all["aq"] < criteria[0]) & (data_all["comfort"] > criteria[1]))
        ] = 2  # recomended for wind
        data_all["recomendation"][
            ((data_all["aq"] < criteria[0]) & (data_all["wbgt"] > criteria[2]))
        ] = 3  # recomended for heat
        data_all["recomendation"][
            (
                (data_all["aq"] < criteria[0])
                & (data_all["comfort"] > criteria[1])
                & (data_all["wbgt"] > criteria[2])
            )
        ] = 4  # recomended for wind and heat

        # health risk map
        output_path = cf["green_potential"]["output_folder"] + "recomendation.tif"
        result = data_all["recomendation"]
        save_raster_file(output_path, result, meta, indexes=1)

    elif method == 2:
        b = 1 / ((thresholds["aq"][1] / (thresholds["aq"][0]) + 1))
        m = b / thresholds["aq"][0]

        data_all["health_risk_aq"] = (
            m * data_all["aq"] + b
        )  # 0.09 is the risk increase per 1 ug/m3 PM2.5 increase, based on WHO (2013) and Burnett et al (2014)
        # data_all['health_risk_aq']=  0.009*data_all['aq'] +0.09  # 0.09 is the risk increase per 1 ug/m3 PM2.5 increase, based on WHO (2013) and Burnett et al (2014)
        data_all["health_risk_aq"][data_all["aq"] < thresholds["aq"][0]] = 0
        # data_all['health_risk_heat'] = 0.018*data_all['wbgt'] + 0.45  # 0.018 is the risk increase per 1 degree C increase, based on Mora et al (2017)

        b = 1 / ((thresholds["wbgt"][1] / (thresholds["wbgt"][0]) + 1))
        m = b / thresholds["wbgt"][0]

        data_all["health_risk_heat"] = m * data_all["wbgt"] + b
        data_all["health_risk_heat"][data_all["wbgt"] < thresholds["wbgt"][0]] = 0
        data_all["health_risk"] = (
            data_all["health_risk_aq"] + data_all["health_risk_heat"]
        )

        # health risk map
        output_path = cf["green_potential"]["output_folder"] + "health_risk.tif"
        result = data_all["health_risk"]
        save_raster_file(output_path, result, meta, indexes=1)
        
        data_all["health_dominates"] = data_all["aq"] * 0
        data_all["health_dominates"][
            (data_all["health_risk_aq"] < data_all["health_risk_heat"])
        ] = 1.0  # heat dominates
        data_all["health_dominates"][
            (data_all["health_risk_aq"] > data_all["health_risk_heat"])
        ] = 2.0  # aq dominates
        data_all["health_dominates"][(data_all["health_risk"] == 0)] = 0.0  # no risk

        output_path = cf["green_potential"]["output_folder"] + "risk_domination.tif"
        result = data_all["health_dominates"]
        save_raster_file(output_path, result, meta, indexes=1)

    elif method == 3:
        aq_thres = cf["green_potential_all"]["air_quality_thresholds"]
        data_all["aq"] = data_all["aq"] * aq_weight

        data_all["aq_val"] = data_all["aq"] * 0
        data_all["aq_val"][(data_all["aq"] < aq_thres["0"])] = 3
        data_all["aq_val"][
            ((data_all["aq"] >= aq_thres["0"]) & (data_all["aq"] < aq_thres["1"]))
        ] = 2
        data_all["aq_val"][
            ((data_all["aq"] >= aq_thres["1"]) & (data_all["aq"] < aq_thres["2"]))
        ] = 1
        data_all["aq_val"][(data_all["aq"] >= aq_thres["2"])] = 0

        comfort_thres = cf["green_potential_all"]["wind_comfort_thresholds"]
        data_all["comfort"] = data_all["comfort"] * comfort_weight
        data_all["comfort_val"] = data_all["aq"] * 0
        data_all["comfort_val"][(data_all["comfort"] <= comfort_thres["0"])] = 0
        data_all["comfort_val"][
            (
                (data_all["comfort"] > comfort_thres["0"])
                & (data_all["comfort"] <= comfort_thres["1"])
            )
        ] = 1
        data_all["comfort_val"][
            (
                (data_all["comfort"] > comfort_thres["1"])
                & (data_all["comfort"] <= comfort_thres["2"])
            )
        ] = 2
        data_all["comfort_val"][(data_all["comfort"] >= comfort_thres["2"])] = 3

        wbgt_thres = cf["green_potential_all"]["wbgt_thresholds"]
        data_all["wbgt"] = data_all["wbgt"] * heat_weight
        data_all["wbgt_val"] = data_all["aq"] * 0
        data_all["wbgt_val"][(data_all["wbgt"] < wbgt_thres["0"])] = 0
        data_all["wbgt_val"][
            (
                (data_all["wbgt"] >= wbgt_thres["0"])
                & (data_all["wbgt"] < wbgt_thres["1"])
            )
        ] = 1
        data_all["wbgt_val"][
            (
                (data_all["wbgt"] >= wbgt_thres["1"])
                & (data_all["wbgt"] < wbgt_thres["2"])
            )
        ] = 2
        data_all["wbgt_val"][(data_all["wbgt"] >= wbgt_thres["2"])] = 3

        data_all["recomendation"] = (
            data_all["aq_val"] + data_all["comfort_val"] + data_all["wbgt_val"]
        )

        output_path = (
            cf["green_potential"]["output_folder"] + "recomendation_method3.tif"
        )
        result = data_all["recomendation"]
        save_raster_file(output_path, result, meta, indexes=1)

        mask_files = [cf["geometry"]["buildings"], cf["geometry"]["waterbodies"]]
        reproject_tiff(
            output_path,
            output_path[:-1*len(".tif")] + "_4326.tif",
            dst_crs="EPSG:4326",
            mask=True,
            mask_frames=mask_files,
        )

        for tt in ["aq", "comfort", "wbgt"]:
            output_path = (
                cf["green_potential"]["output_folder"]
                + "recomendation_method3_"
                + tt
                + ".tif"
            )
            result = data_all[tt + "_val"]
            save_raster_file(output_path, result, meta, indexes=1)

def reproject_tiff(src_tif, dst_tif, dst_crs="4326", mask=True, mask_frames=None):
    # Open source GeoTIFF
    src = open_raster_file(src_tif, "rb")
    # Define target CRS
    # Compute transform and new dimensions
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    # Create output metadata
    kwargs = src.meta.copy()
    kwargs.update(
        {"crs": dst_crs, "transform": transform, "width": width, "height": height}
    )

    # Reproject
    dst = open_raster_file(dst_tif, "wb", kwargs)
    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=rasterio.band(dst, i),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,  # or bilinear/cubic
        )
    dst.close()
    src.close()

    if mask:
        # Step 4: Apply the mask to the GeoTIFF
        for mask_files in mask_frames:
            mask_frame = read_shape_file(mask_files)
            mask_frame.to_crs(epsg=4326, inplace=True)  # ensure

            src = open_raster_file(dst_tif, "rb")
            geometries = mask_frame.geometry.values
            out_image, out_transform = mask_raster(
                src, geometries, crop=True, invert=True, nodata=np.nan
            )

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": src.crs,
                }
            )
            src.close()

        # Step 5: Save the masked GeoTIFF
        print("Saving GeoTIFF to " + dst_tif)
        save_raster_file(dst_tif, out_image, out_meta)

    print("✅ Reprojected GeoTIFF saved at:", dst_tif)

def create_special_green_map(cf):
    """
    Create points along building boundaries where:
        greening could be placed if facade values above ebg thresholds
        hedges could be placed if no2 values above threshold

    """
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    import math

    wind_comfort = cf["maps"]["wind_comfort"]
    air_quality = cf["maps"]["air_quality"]
    wbgt = cf["maps"]["wbgt"]

    ref_map = air_quality
    wind_comfort_resampled = (
        cf["green_potential"]["output_folder"] + "wind_comfort_resampled.tif"
    )
    resample_to_match(
        cf,
        src_path=wind_comfort,
        ref_path=ref_map,
        out_path=wind_comfort_resampled,
        resampling="nearest",
    )

    wbgt_resampled = cf["green_potential"]["output_folder"] + "wbgt_max_resampled.tif"
    resample_to_match(
        cf, src_path=wbgt, ref_path=ref_map, out_path=wbgt_resampled, resampling="nearest"
    )

    RASTER_PATHs = [
        cf["maps"]["air_quality"],
        cf["maps"]["wbgt"],
    ]  # "data/rasters/pm25_2019_1km.tif"
    thresholds = [
        cf["green_potential_all"]["threshold_hedges_no2"],
        cf["green_potential_all"]["threshold_green_walls_wbgt"],
    ]  # 35.5 #26.75
    output = ["hedges", "green_walls"]

    USER_SPACING = 2.0  # desired spacing between points along lines (in raster CRS units); if None or <=0, will use ~half pixel size
    BAND = 1
    # Read polygons
    gdf = read_shape_file(cf["geometry"]["buildings"])

    # Extract boundary as LineString/MultiLineString
    gdf = gdf.explode(index_parts=True).reset_index()
    gdf["boundary"] = gdf.geometry.boundary

    # gdf["rings"] = gdf.geometry.apply(extract_rings)

    gdf_lines = gdf.copy()
    gdf_lines["geometry"] = gdf_lines.boundary
    gdf_lines = gdf_lines.set_geometry("geometry")

    # gdf_lines = gdf.boundary.explode(index_parts=True).reset_index()
    gdf_lines = gpd.GeoDataFrame(geometry=gdf_lines["boundary"], crs=gdf.crs)

    # Apply to each polygon
    # gdf_lines.to_file("data/geometry/bld_lines.shp")

    def densify_line(line, spacing):
        """Return points every 'spacing' units along the line, including the end."""
        length = line.length
        if length == 0:
            return [line.interpolate(0.0)]
        n = max(1, int(math.floor(length / spacing)))
        dists = [i * spacing for i in range(n)] + [length]
        return [line.interpolate(d) for d in dists]

    for i in range(len(RASTER_PATHs)):
        # Read inputs
        gdf = (
            gdf_lines.copy()
        )  # gpd.read_file(LINES_PATH, layer=LAYER_NAME) if LAYER_NAME else gpd.read_file(LINES_PATH)

        gdf = gdf.explode(
            index_parts=True, ignore_index=True
        )  # flatten MultiLineString
        RASTER_PATH = RASTER_PATHs[i]
        with rasterio.open(RASTER_PATH) as src:
            # Align CRS
            if gdf.crs.to_string() != src.crs.to_string():
                gdf = gdf.to_crs(src.crs)

            # Pixel size & default spacing
            px = float(np.mean([abs(src.transform.a), abs(src.transform.e)]))
            spacing = (
                USER_SPACING
                if (USER_SPACING and USER_SPACING > 0)
                else max(0.5 * px, 1e-9)
            )
            # Build a VRT using the selected resampling for point sampling
            with WarpedVRT(src, resampling=Resampling.nearest) as vrt:
                # pick an id column if one exists, else create one
                id_col = next(
                    (
                        c
                        for c in ["id", "ID", "fid", "FID", "objectid", "OBJECTID"]
                        if c in gdf.columns
                    ),
                    None,
                )
                if id_col is None:
                    gdf = gdf.reset_index(names="line_id")
                    id_col = "line_id"

                records = []
                for _, row in gdf.iterrows():
                    lid = row[id_col]
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue

                    pts = densify_line(geom, spacing)
                    coords = [(p.x, p.y) for p in pts]

                    # Sample selected band
                    for order, (p, arr) in enumerate(zip(pts, vrt.sample(coords))):
                        # arr shape: (nbands,)
                        val = float(arr[BAND - 1]) if np.size(arr) >= BAND else np.nan
                        # Mask NODATA
                        if (
                            src.nodata is not None
                            and np.isfinite(src.nodata)
                            and val == float(src.nodata)
                        ):
                            val = np.nan
                        records.append(
                            {
                                id_col: lid,
                                "order": order,
                                "x": p.x,
                                "y": p.y,
                                "value": val,
                                "geometry": p,
                            }
                        )

        # Points with sampled values
        pts_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)
        pts_gdf = pts_gdf[pts_gdf["value"] > thresholds[i]]
        pts_gdf.to_file(
            cf["green_potential"]["output_folder"] + "/" + output[i] + ".geojson",
            driver="GEOjson",
        )
        print(
            "Save recomendation for ", output[i], "with threshold:", str(thresholds[i])
        )
        print(
            "at: "
            + cf["green_potential"]["output_folder"]
            + "/"
            + output[i]
            + ".geojson"
        )


def plot_green_map(cf):

    print("Implement me")


def extract_rings(poly: Polygon):
    if poly.is_empty:
        return []
    rings = [poly.exterior] + list(poly.interiors)
    rings = rings[0]
    return rings


def load_json_with_comments(path):
    with open(path, "r") as file:
        content = file.read()
        # remove //... and /*...*/ comments
        content = re.sub(r"//.*", "", content)  # single-line //
        content = re.sub(r"#.*", "", content)  # single-line #
        content = re.sub(r"/\*[\s\S]*?\*/", "", content)  # block comments
        return json.loads(content)

def file_exists(file):
    if "s3." in file:
        access_file = "/home/jovyan/access_keys.json"
        with open(access_file, 'r') as fh:
            s3_keys = json.load(fh)
        print("Attempting to find S3 file: "+file)
        parts = file.split("/")
        url = "https://" + parts[0]
        bucket = parts[1]
        objct = parts[2]
        fs = s3fs.S3FileSystem(
            endpoint_url=url,
            key=s3_keys["accessKey"],
            secret=s3_keys["secretKey"]
        )
        bl = fs.ls(bucket)
        return (bucket + "/" + objct) in bl
    else:
        return os.path.isfile(file)

def read_shape_file(path):
    if "s3." in path:
        fo = open_s3_file(path, "rb")
        gdf = gpd.read_file(fo)
    else:
        gdf = gpd.read_file(path)
    return gdf

def open_s3_file(file, mode='rb'):
    access_file = "/home/jovyan/access_keys.json"
    with open(access_file, 'r') as fh:
        s3_keys = json.load(fh)
    print("Attempting to open S3 file: " + file + " in mode: " + mode)
    # append "https://" to beginning of url - the load_json_with_comments function above removes the // and causes the url to be misformed
    parts = file.split("/")
    url = "https://" + parts[0]
    bucket = parts[1]
    object = parts[2]
    fs = s3fs.S3FileSystem(
        endpoint_url=url,
        key=s3_keys["accessKey"],
        secret=s3_keys["secretKey"]
    )
    fo = fs.open(bucket + "/" + object, mode)
    return fo

def save_raster_file(output_path, result, meta={}, indexes=None):
    if "s3." in output_path:
        # open with S3
        fo = open_s3_file(output_path, mode="wb")
        with rasterio.open(fo, "w", **meta) as dst:
            dst.write(result.astype(rasterio.float32), indexes=indexes)
        fo.close()
    else:
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(result.astype(rasterio.float32), indexes=indexes)

def open_raster_file(input_path, mode="rb", kwargs={}):
    if "b" in mode:
        rio_mode = mode.strip("b")
    else:
        rio_mode = mode

    if "s3." in input_path:
        # open with S3
        fo = open_s3_file(input_path, mode)
        src = rasterio.open(fo, rio_mode, **kwargs)
    else:
        src = rasterio.open(input_path, rio_mode, **kwargs)
    return src

def read_tiff_with_coords(file):
    if "s3." in file:
        # open with S3
        fo = open_s3_file(file)
        src = rasterio.open(fo)
    else:
        src = rasterio.open(file)
    data = src.read(1)  # Read first band
    transform = src.transform  # Affine transform
    meta = src.meta

    # Build coordinate arrays
    rows, cols = np.indices(data.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)

    return data, xs, ys, meta


def resample_to_match(
    cf: dict,
    src_path: str,
    ref_path: str,
    out_path: str | None = None,
    resampling: str = "bilinear",
    dst_dtype: str | None = None,
):
    """
    Resample/reproject src raster to the grid of ref raster.

    Parameters
    ----------
    src_path : path to source GeoTIFF to be interpolated
    ref_path : path to reference GeoTIFF whose grid we will match
    out_path : optional path to write a GeoTIFF; if None, returns array+profile
    resampling : one of _RESAMPLING keys (e.g., 'nearest', 'bilinear', 'cubic', ...)
    dst_dtype : optional output dtype (e.g., 'float32'); default = src dtype

    Returns
    -------
    (dst_array, dst_profile) if out_path is None; otherwise writes file and returns (out_path, dst_profile)
    """
    if resampling == "nearest":
        resamp = Resampling.nearest
    elif resampling == "bilinear":
        resamp = Resampling.bilinear
    elif resampling == "cubic":
        resamp = Resampling.cubic
    else:
        raise ValueError(f"Unsupported resampling method: {resampling}")

    # 1) Read reference grid specs
    if "s3." in ref_path:
        ref_ref = open_s3_file(ref_path)
    else:
        ref_ref = ref_path
        
    with rasterio.open(ref_ref) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height, dst_width = ref.height, ref.width

    # 2) Open source and prepare destination
    if "s3." in src_path:
        src_ref = open_s3_file(src_path)
    else:
        src_ref = src_path
        
    with rasterio.open(src_ref) as src:
        count = src.count
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        dtype = dst_dtype or src.dtypes[0]

        # Choose a sensible dst_nodata
        if src_nodata is not None:
            dst_nodata = src_nodata
        else:
            # default: NaN for float outputs; 0 otherwise
            dst_nodata = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0

        dst = np.full((count, dst_height, dst_width), dst_nodata, dtype=dtype)

        # 3) Reproject band by band
        for b in range(1, count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=dst[b - 1],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
                resampling=resamp,
                num_threads=2,
            )

        # 4) Build output profile
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": dst_height,
                "width": dst_width,
                "transform": dst_transform,
                "crs": dst_crs,
                "dtype": dtype,
                "nodata": dst_nodata,
                "count": count,
                "compress": "deflate",
                # "predictor": 2 if np.issubdtype(np.dtype(dtype), np.floating) else 1,
                "predictor": 1,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

    # 5) Either write to disk or return arrays
    if out_path:
        save_raster_file(out_path, dst, profile)
        return out_path, profile
    else:
        return dst, profile
