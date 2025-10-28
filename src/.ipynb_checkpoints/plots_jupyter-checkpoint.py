
from IPython.display import Image, display
import io, base64
import rasterio, numpy as np, io
from PIL import Image
from base64 import b64encode
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask
from matplotlib import animation
from IPython.display import HTML

def plot_green_potential(cf,static):

    tif_path =cf['green_potential']['output_folder']+"recomendation_method3_4326.tif"
    # --- Read band + bounds ---
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        band = src.read(1)

    # --- Define categories (string bins) ---
    # The raster values must match these numeric keys
    categories = {
        1: "Very Low",
        2: "",
        3: "",
        4: "",
        5: "",
        6: "",
        7: "",
        8: "",
        9: "Very high"
    }

    # --- Assign colors to each category ---
    colors = [
        "#313695", "#4575b4", "#74add1", "#abd9e9",
        "#ffffbf", "#fdae61", "#f46d43", "#d73027", "#a50026"
    ]
    cmap = ListedColormap(colors)
    bins = list(categories.keys())
    norm = BoundaryNorm(bins + [bins[-1] + 1], cmap.N)

    # --- Apply color mapping ---
    #valid = np.isfinite(band)
    #arr = np.zeros_like(band, dtype=np.float32)
    #arr[valid] = band[valid]
    
    arr=band
    rgba = (cmap(norm(arr)) * 255).astype(np.uint8)
    
    # Make NaN pixels fully transparent
    nan_mask = ~np.isfinite(band)
    rgba[nan_mask, :] = [0, 0, 0, 0]  # fully transparent pixels (RGBA = 0,0,0,0)

    # --- Convert to base64 PNG ---
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    image_url = "data:image/png;base64," + b64encode(buf.getvalue()).decode("utf-8")

    # --- Leaflet bounds ---
    image_bounds = ((bounds.bottom, bounds.left), (bounds.top, bounds.right))

    # --- Create map ---

    if static:
        img_data = base64.b64decode(image_url.split(",")[1])
        img = Image.open(io.BytesIO(img_data))
    
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(img)
        ax.axis('off')
        ax.set_title("Green Potential - Recommendation Map", fontsize=14, fontweight="bold")
    
        # --- Add color scale legend ---
        cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_ticks([b + 0.5 for b in bins])
        cb.set_ticklabels(list(categories.values()))
        cb.ax.tick_params(labelsize=10)
        cb.set_label("Green Potential Index", fontsize=12)

        plt.show()
        
    
    else:
        from ipyleaflet import Map, ImageOverlay, basemaps, WidgetControl,TileLayer
        from ipywidgets import HTML

        m = Map(
        center=((bounds.top + bounds.bottom)/2, (bounds.left + bounds.right)/2),
        zoom=14,
        basemap=basemaps.CartoDB.Positron  # <-- grayscale/light tiles
    )
        m.layout.height = "600px"
    
    
    
        # --- Overlay raster ---
        overlay = ImageOverlay(url=image_url, bounds=image_bounds, opacity=0.8)
        m.add_layer(overlay)
    
        # --- Create a categorical legend using HTML ---
        legend_items = "".join(
            f'<div style="display:flex;align-items:center;">'
            f'<div style="background:{c};width:20px;height:15px;margin-right:8px;"></div>'
            f'<span>{categories[k]}</span></div>'
            for k, c in zip(categories.keys(), colors)
        )
        legend_html = HTML(f"""
        <div style="background-color:white;padding:8px;border-radius:5px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.3);font-size:13px;">
        <b>Recomendation for Urban Green Potential</b><br>
        {legend_items}
        </div>
        """)
    
        legend_control = WidgetControl(widget=legend_html, position="bottomright")
        m.add_control(legend_control)
    
        display(m)
        


def plot_urban_wind(tif_path, vmin=0, vmax=1, mask_shp=None, static=True):

    # --- path to your GeoTIFF ---
    #tif_path = "/projects/urbanair/urbanair_greenpotential/data/maps/recomendation_method3_4326.tif"  # change this

    # --- choose a color map ---
    cmap_name = "jet"  # options: 'viridis', 'plasma', 'jet', 'turbo', 'coolwarm', etc.
    cmap = plt.get_cmap(cmap_name)
    
    # Example shapefile or GeoJSON polygon
   
    gdf = gpd.read_file(mask_shp)
    gdf.to_crs(epsg=4326, inplace=True)  # ensure it's in WGS84
    geoms = gdf.geometry

    # --- read first band + bounds ---
    with rasterio.open(tif_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True, invert=True)
        bounds = src.bounds
        band = out_image[0]
        
    band =band  
    
    
    print("Min:", np.nanmin(band))
    print("Max:", np.nanmax(band))
    print("Mean:", np.nanmean(band))
    print("Unique values:", np.unique(band[~np.isnan(band)])[:20])
    print("Count of NaN:", np.isnan(band).sum())
    
    
    
    band = np.nan_to_num(band, nan=0)
    
    
    print(band)
    # --- Define categories (string bins) ---   
    # The raster values must match these numeric keys
    categories = {
        0.1: "Very Low",
        0.2: "Low",
        0.3: "Moderate",
        0.4: "High",
        0.5: "Very High",
        0.6: "Extremely High",
        0.7: "Very High",
        0.8: "Extremely High",
        0.9: "Catastrophic"
    }

    # --- Assign colors to each category ---
    colors = [
        "#313695", "#4575b4", "#74add1", "#abd9e9",
        "#ffffbf", "#fdae61", "#f46d43", "#d73027", "#a50026"
    ]
    cmap = ListedColormap(colors)
    bins = list(categories.keys())
    norm = BoundaryNorm(bins + [bins[-1] + 1], cmap.N)

    # --- Apply color mapping ---
    valid = np.isfinite(band)
    arr = np.zeros_like(band, dtype=np.float32)
    arr[valid] = band[valid]
    rgba = (cmap(norm(arr)) * 255).astype(np.uint8)
    
    # Make NaN pixels fully transparent
    nan_mask = ~np.isfinite(band)
    rgba[nan_mask, :] = [0, 0, 0, 0]  # fully transparent pixels (RGBA = 0,0,0,0)

    # --- Convert to base64 PNG ---
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    image_url = "data:image/png;base64," + b64encode(buf.getvalue()).decode("utf-8")

    if static:
    
        from IPython.display import Image, display
        import io, base64
        
        # Extract the base64 part only (remove the "data:image/png;base64," prefix)
        img_data = image_url.split(",")[1]
        display(Image(data=base64.b64decode(img_data)))
    

    else:
        from ipyleaflet import Map, ImageOverlay, basemaps, WidgetControl,TileLayer
        from ipywidgets import HTML
        # --- Leaflet bounds ---
        image_bounds = ((bounds.bottom, bounds.left), (bounds.top, bounds.right))
    
        # --- Create map ---
        m = Map(
        center=((bounds.top + bounds.bottom)/2, (bounds.left + bounds.right)/2),
        zoom=14,
        basemap=basemaps.CartoDB.Positron  # <-- grayscale/light tiles
        )
        m.layout.height = "600px"
    
        # --- Overlay raster ---
        overlay = ImageOverlay(url=image_url, bounds=image_bounds, opacity=0.8)
        m.add_layer(overlay)
    
        # --- Create a categorical legend using HTML ---
        legend_items = "".join(
            f'<div style="display:flex;align-items:center;">'
            f'<div style="background:{c};width:20px;height:15px;margin-right:8px;"></div>'
            f'<span>{categories[k]}</span></div>'
            for k, c in zip(categories.keys(), colors)
        )
        legend_html = HTML(f"""
        <div style="background-color:white;padding:8px;border-radius:5px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.3);font-size:13px;">
        <b>Recomendation for Urban Green Potential</b><br>
        {legend_items}
        </div>
        """)
    
        legend_control = WidgetControl(widget=legend_html, position="bottomright")
        m.add_control(legend_control)
    
        display(m)

def wind_animation(wind_local,vmin=0,vmax=4):
    
    times = sorted(wind_local.keys())
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(wind_local[times[0]], cmap='jet', animated=True, vmin=0, vmax=4)
    
    title = ax.set_title(f"Wind field at {times[0]}")
    ax.axis("off")
    
    # Update function for animation
    def update(frame):
        data = wind_local[times[frame]]
        im.set_array(data)
        title.set_text(f"Wind field at {times[frame]}")
        return [im, title]
    
    # Build the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(times), interval=700, blit=True
    )
    
    # Show in Jupyter
    HTML(ani.to_jshtml())

    