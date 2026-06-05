"""
flood_dem.py
============
Helper module for the ADAPT app's "Flood Maps" tab.

Implements a lightweight, no-hosting bathtub flood-inundation pipeline:

  * Terrain comes from the USGS 3DEP 1/3 arc-second (~10 m) seamless DEM,
    which is public-domain, NAVD88 (metres), and stored as Cloud-Optimized
    GeoTIFFs on the public AWS bucket `prd-tnm`. We read ONLY the small
    region-of-interest window over HTTP range requests via GDAL's
    `/vsicurl/` — nothing is downloaded in full and nothing is bundled
    with the app.

  * The inundation model is a pure bathtub (no hydraulic connectivity),
    matching `flood_maps_mastic_beach.m`:
        depth = water_level - ground_elevation     (depth < 0 -> dry)
    Because 3DEP is bare-earth LAND elevation (no bathymetry, water
    surfaces flattened), open water is delineated by `Z <= 0` and NoData
    rather than the topobathy script's `Z < 0`. The flooded-LAND footprint
    is what this changes nothing about; only how permanent water is drawn.

  * Water levels are in feet NAVD88 (as in the rest of the pipeline); the
    DEM is metres NAVD88, so we convert once.

Only `read_dem_roi()` needs rasterio, and it imports it lazily, so this
module is import-safe even where rasterio is not installed. The rendering
helper uses Pillow (a Streamlit dependency).

To point the app at a different DEM (e.g. your own Cloud-Optimized GeoTIFF
of the New England Topobathy mosaic, which would restore pixel-identical
output to the MATLAB maps), set DEM_COG_OVERRIDE_URL to that single COG's
URL and read_dem_roi() will window-read it instead of the 3DEP tiles.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# DEM source configuration
# ---------------------------------------------------------------------------
# Per-tile 3DEP 1/3 arc-second COGs, addressed by 1-degree NW-corner tile
# name, e.g. n41w073 covers latitude [40,41], longitude [-73,-72].
DEM_COG_URL_TEMPLATE = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
    "current/{tile}/USGS_13_{tile}.tif"
)

# If set to a URL string, read this single COG instead of the 3DEP tiles
# (use for a self-hosted topobathy COG). It must be a COG (internal tiling
# + overviews) so windowed range-reads stay cheap, and vertically NAVD88.
DEM_COG_OVERRIDE_URL = None

FT_PER_M = 1.0 / 0.3048

# ---------------------------------------------------------------------------
# Discrete depth colour scheme (feet) — matches the MATLAB workshop maps.
# Six bins; the last catches depths >= 5 ft. RGB given 0-255.
# ---------------------------------------------------------------------------
WS_BINS_FT = [0, 1, 2, 3, 4, 5, 6]          # 7 edges; last edge is a tick label
WS_COLORS = [
    (74, 0, 130),     # 0-1 ft  indigo
    (31, 143, 255),   # 1-2 ft  blue
    (0, 204, 204),    # 2-3 ft  cyan/teal
    (255, 235, 0),    # 3-4 ft  yellow
    (255, 140, 0),    # 4-5 ft  orange
    (191, 13, 13),    # 5+  ft  dark red
]


def dem_tiles_for_bbox(bbox):
    """Return the 3DEP 1-degree tile name(s) covering a lon/lat bbox.

    bbox = (lon_min, lat_min, lon_max, lat_max), lon negative in the US.
    A tile `nA wB` covers latitude [A-1, A] and longitude [-B, -(B-1)].
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    a_min = int(math.floor(lat_min)) + 1
    a_max = int(math.ceil(lat_max))
    b_min = int(math.floor(-lon_max)) + 1   # -lon_max is the smaller west-magnitude
    b_max = int(math.ceil(-lon_min))
    tiles = []
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            tiles.append(f"n{a:02d}w{b:03d}")
    return tiles


def roi_from_lonlat(lon, lat, buffer_m=600.0, min_span_km=1.5, max_span_km=25.0):
    """Robust bounding box around point coordinates.

    Centred on the MEDIAN coordinate. Points farther than a MAD-based
    threshold from the centre are dropped as outliers before the box is
    sized, so a handful of stray/garbage coordinates can't blow the extent
    up (the failure that produced the whole-region map). Padded by buffer_m
    and clamped to [min_span_km, max_span_km]. Returns
    (lon_min, lat_min, lon_max, lat_max).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    m = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[m], lat[m]
    if lon.size == 0:
        raise ValueError("No finite coordinates to build a region of interest.")

    clon = float(np.median(lon))
    clat = float(np.median(lat))
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * max(0.1, math.cos(math.radians(clat)))

    dx = (lon - clon) * m_per_lon
    dy = (lat - clat) * m_per_lat
    dist = np.hypot(dx, dy)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    thr = max(med + 5.0 * 1.4826 * mad, 2000.0)   # keep within ~max(., 2 km)
    keep = dist <= thr
    if not keep.any():
        keep = np.ones_like(dist, dtype=bool)
    lon_k, lat_k = lon[keep], lat[keep]

    half_lo = 500.0 * min_span_km
    half_hi = 500.0 * max_span_km
    hx_m = min(max((lon_k.max() - lon_k.min()) * 0.5 * m_per_lon + buffer_m, half_lo), half_hi)
    hy_m = min(max((lat_k.max() - lat_k.min()) * 0.5 * m_per_lat + buffer_m, half_lo), half_hi)
    dlon = hx_m / m_per_lon
    dlat = hy_m / m_per_lat
    return (clon - dlon, clat - dlat, clon + dlon, clat + dlat)


def maybe_swap_lonlat(lon, lat):
    """Guard against transposed columns. In the US longitude is large-negative
    (~-65..-125) and latitude is ~20..50, so if |lon| looks like a latitude and
    |lat| looks like a longitude, swap them. Returns (lon, lat, swapped)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    mlon = np.nanmedian(np.abs(lon))
    mlat = np.nanmedian(np.abs(lat))
    if np.isfinite(mlon) and np.isfinite(mlat) and mlon < 60.0 and mlat > 60.0:
        return lat, lon, True
    return lon, lat, False


def read_dem_roi(bbox, target_res_m=10.0):
    """Read the DEM over `bbox` onto a regular lon/lat grid (~target_res_m).

    Reads only the ROI window from the remote COG(s) via GDAL `/vsicurl/`
    range requests. Returns (Z_m, extent) where:
        Z_m    : float32 array, shape (nlat, nlon), row 0 = NORTH edge,
                 metres NAVD88, np.nan where NoData.
        extent : (lon_min, lat_min, lon_max, lat_max) actually gridded.

    Raises RuntimeError if no source tile could be read.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Lazy imports so the module stays importable without rasterio installed.
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as transform_from_bounds

    latc = 0.5 * (lat_min + lat_max)
    dlat = target_res_m / 111320.0
    dlon = target_res_m / (111320.0 * max(0.1, math.cos(math.radians(latc))))
    nlon = max(2, int(round((lon_max - lon_min) / dlon)))
    nlat = max(2, int(round((lat_max - lat_min) / dlat)))

    # Safety: never build an absurdly large grid (e.g. if the ROI is still
    # big). Coarsen uniformly so the longer side stays <= 2500 cells.
    longest = max(nlon, nlat)
    if longest > 2500:
        shrink = longest / 2500.0
        nlon = max(2, int(nlon / shrink))
        nlat = max(2, int(nlat / shrink))

    dst_transform = transform_from_bounds(lon_min, lat_min, lon_max, lat_max,
                                          nlon, nlat)
    dst = np.full((nlat, nlon), np.nan, dtype="float32")

    if DEM_COG_OVERRIDE_URL:
        urls = [DEM_COG_OVERRIDE_URL]
    else:
        urls = [DEM_COG_URL_TEMPLATE.format(tile=t) for t in dem_tiles_for_bbox(bbox)]

    # GDAL/CURL env: don't probe sidecar files, restrict to .tif, reuse
    # connections — keeps the windowed range-reads fast.
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MULTIPLEX="YES",
        VSI_CACHE="TRUE",
    )

    any_ok = False
    errors = []
    with env:
        for url in urls:
            vsi = url if url.startswith("/vsicurl/") else "/vsicurl/" + url
            try:
                with rasterio.open(vsi) as src:
                    src_nodata = src.nodata
                    with WarpedVRT(src, crs="EPSG:4326",
                                   transform=dst_transform,
                                   width=nlon, height=nlat,
                                   resampling=Resampling.bilinear) as vrt:
                        arr = vrt.read(1).astype("float32")
                if src_nodata is not None:
                    arr[arr == src_nodata] = np.nan
                arr[arr <= -500.0] = np.nan       # guard residual NoData sentinels
                fill = np.isnan(dst) & np.isfinite(arr)
                dst[fill] = arr[fill]
                any_ok = True
            except Exception as exc:                # noqa: BLE001
                errors.append(f"{vsi.split('/')[-1]}: {exc}")

    if not any_ok:
        raise RuntimeError(
            "Could not read any DEM tile for this area. "
            "Tried: %s. Errors: %s"
            % (", ".join(u.split("/")[-1] for u in urls), " | ".join(errors))
        )

    return dst, (lon_min, lat_min, lon_max, lat_max)


def bathtub_depth_ft(Z_m, wl_ft):
    """Bathtub inundation depth in feet for a water level (ft NAVD88).

    depth = wl - Z, masked where dry (depth < 0), where Z <= 0 (permanent
    water for a land-only DEM) and where Z is NaN. Returns float32 with
    np.nan on dry/water cells.
    """
    wl_m = float(wl_ft) * 0.3048
    depth_m = wl_m - Z_m
    invalid = ~np.isfinite(Z_m) | (Z_m <= 0.0)
    depth_m = np.where(invalid, np.nan, depth_m)
    depth_m = np.where(depth_m < 0.0, np.nan, depth_m)
    return depth_m.astype("float32") * FT_PER_M


def depth_to_rgba_data_uri(depth_ft):
    """Render a depth-in-feet array to a base64 PNG data URI (RGBA).

    Flooded cells are coloured by the discrete WS bins; dry/water cells are
    fully transparent. Row 0 of `depth_ft` is treated as the NORTH edge, so
    the PNG aligns with a Plotly image-layer whose first corner is top-left
    (lon_min, lat_max).
    """
    from PIL import Image  # Pillow ships with Streamlit

    h, w = depth_ft.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    n = len(WS_COLORS)
    for b, (r, g, bl) in enumerate(WS_COLORS):
        if b < n - 1:
            m = (depth_ft >= WS_BINS_FT[b]) & (depth_ft < WS_BINS_FT[b + 1])
        else:
            m = depth_ft >= WS_BINS_FT[b]            # catch-all top bin
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = bl
        rgba[m, 3] = 255

    import io
    import base64
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def mapbox_zoom_for_bbox(extent, width_px=700, height_px=360, pad=1.12):
    """Approximate a Plotly mapbox zoom that frames the bbox in a typical
    panel. Fits the more constraining of the lon/lat spans."""
    lon_min, lat_min, lon_max, lat_max = extent
    lon_span = max((lon_max - lon_min) * pad, 1e-4)
    lat_span = max((lat_max - lat_min) * pad, 1e-4)
    z_lon = math.log2(360.0 / lon_span) + math.log2(max(width_px, 1) / 512.0)
    z_lat = math.log2(180.0 / lat_span) + math.log2(max(height_px, 1) / 512.0)
    return max(1.0, min(15.0, min(z_lon, z_lat)))


def legend_html():
    """Small inline HTML legend for the discrete depth bins (feet)."""
    labels = ["0–1", "1–2", "2–3", "3–4", "4–5", "5+"]
    items = []
    for (r, g, b), lab in zip(WS_COLORS, labels):
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
            f'<span style="width:14px;height:14px;background:rgb({r},{g},{b});'
            f'display:inline-block;margin-right:5px;border:1px solid #999;"></span>'
            f'{lab} ft</span>'
        )
    return (
        '<div style="font-size:0.85rem;color:#334155;margin:0.25rem 0 0.5rem;">'
        '<b>Flood depth</b>&nbsp;&nbsp;' + "".join(items) + "</div>"
    )
