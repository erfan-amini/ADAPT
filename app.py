"""
ADAPT - Assessment of Damage and Adaptation Planning Tool
Center for Climate Systems Research
The Climate School, Columbia University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import openpyxl
import math
from types import SimpleNamespace

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ADAPT | Flood Risk Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR STYLING
# ============================================================================
st.markdown("""
<style>
    /* Reduce top padding on main content */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Tagline - reduced margin */
    .main-tagline {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 0.75rem;
        margin-top: 0;
        text-align: center;
    }
    
    /* Reduce whitespace around images */
    [data-testid="stImage"] {
        margin-top: -0.5rem;
        margin-bottom: -0.5rem;
    }
    
    /* Tab titles - smaller */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 0.6rem 1.25rem;
    }
    
    /* Section headers/subheaders inside tabs */
    h2 {
        font-size: 1.75rem !important;
    }
    h3 {
        font-size: 1.5rem !important;
    }
    
    /* Streamlit subheader override */
    [data-testid="stSubheader"] {
        font-size: 1.75rem !important;
    }
    
    /* Tab description */
    .tab-description {
        font-size: 1.1rem;
        color: #64748b;
        font-style: italic;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f8fafc;
        border-radius: 0.25rem;
        border-left: 3px solid #0ea5e9;
    }
    
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    .stSelectbox label {
        color: #1e293b !important;
    }
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #1e293b !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] p {
        color: #334155 !important;
    }
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
    }
    .footer-org {
        font-weight: 500;
        color: #334155;
    }
    .footer-license {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    /* Compact buttons — shorter overall height by trimming vertical padding.
       Applies to every st.button in the app for visual consistency. Width
       and font size are untouched, so button labels still fit on one line.
       `white-space: nowrap` on the inner paragraph forces the icon and the
       label to stay on a single horizontal line — without it, Streamlit
       would break "🔍 Find" into two stacked lines whenever the column
       got narrow, which doubled the button's height. */
    .stButton > button {
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
        min-height: 0 !important;
        line-height: 1.3 !important;
        white-space: nowrap !important;
    }
    .stButton > button p {
        white-space: nowrap !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_filename(filename):
    """Parse filename to extract location."""
    name = os.path.basename(filename)
    # Strip all known extensions
    for ext in ['.xlsx', '.XLSX', '.csv', '.CSV']:
        name = name.replace(ext, '')
    
    location = "Unknown Location"
    location_patterns = [
        ('MasticBeach', 'Mastic Beach'),
        ('Mastic_Beach', 'Mastic Beach'),
        ('WestPoint', 'West Point'),
        ('West_Point', 'West Point'),
        ('Shinnecock', 'Shinnecock'),
        ('Pamunkey', 'Pamunkey'),
        ('Hampton', 'Hampton'),
        ('Montauk', 'Montauk'),
    ]
    
    for pattern, display_name in location_patterns:
        if pattern.lower() in name.lower():
            location = display_name
            break
    
    if location == "Unknown Location":
        parts = name.split('_')
        for part in parts:
            if part not in ['CSV1', 'CSV2', 'Aggregated', 'PerBuilding', 'Results', 'RES', 'NONRES', 'COM', 'ALL']:
                if len(part) > 3:
                    location = part.replace('_', ' ')
                    break
    
    return location


def is_residential(occupancy_type):
    """Check if occupancy type is residential"""
    if pd.isna(occupancy_type):
        return False
    occ = str(occupancy_type).upper()
    return occ.startswith('RES')


# NSI foundation_type codes that mean "this building has a basement that
# could plausibly be wet-floodproofed". Anything else (Pier, Slab,
# Crawlspace, Solid wall, etc.) has no basement, so the WFP Basement
# retrofit is physically meaningless for it. Stored as a set so the
# applicability check is a single membership test.
_BASEMENT_FOUNDATION_CODES = {'B'}

# DFE-status strings (lowercased + whitespace-stripped) that mean
# "this building is already above design flood elevation". For such
# buildings, the Elevate retrofit provides no further benefit — the
# data generator typically encodes this as a no-op, but we drop the
# row explicitly as a backstop so it doesn't slip through into hovers
# or charts due to numerical drift.
_ABOVE_DFE_STATUS_STRINGS = {'above dfe', 'above_dfe', 'abovedfe'}


def retrofit_applies(action, foundation_type=None, dfe_status=None):
    """Return True if `action` physically applies to a building with the
    given foundation type / DFE status.

    A retrofit that doesn't apply (e.g., wet-floodproofing a basement
    that doesn't exist, or elevating a building that's already above
    DFE) should not appear in the UI — its "damage" value is an
    artifact of the math running regardless of whether the retrofit is
    meaningful, and showing it as a $0 / -100% saving is actively
    misleading.

    Rules:
      * 'WFP B' (Wet Floodproof Basement): requires foundation_type == 'B'.
        Missing foundation type → conservatively excluded.
      * 'Elevate': hidden for buildings already above DFE.
        Missing DFE status → NOT hidden (we don't have grounds to drop it).
      * Everything else ('No mitigation', 'Raise Utilities', 'WFP 1st'):
        applies universally.
    """
    if action == 'WFP B':
        if pd.isna(foundation_type):
            return False
        return str(foundation_type).strip().upper() in _BASEMENT_FOUNDATION_CODES
    if action == 'Elevate':
        if pd.isna(dfe_status):
            return True
        return str(dfe_status).strip().lower() not in _ABOVE_DFE_STATUS_STRINGS
    return True


def filter_by_occupancy(df, occupancy_selection):
    """Filter dataframe by occupancy type selection"""
    if df is None:
        return None
    
    if occupancy_selection == "All":
        return df
    
    if 'occupancy_type' not in df.columns:
        return df
    
    if occupancy_selection == "Residential":
        return df[df['occupancy_type'].apply(is_residential)].copy()
    elif occupancy_selection == "Non-Residential":
        return df[~df['occupancy_type'].apply(is_residential)].copy()
    
    return df


def convert_floodplain_status(status):
    """Normalize a building's DFE-status string to canonical values.

    Canonical bundles already ship `DFE_Status` with the values
    'Above DFE' / 'Under DFE' — those pass through unchanged. Legacy
    bundles shipped the column as 'In floodplain' / 'Out of floodplain';
    we map those to the canonical wording here so every downstream
    comparison can assume a single vocabulary. Anything we don't
    recognize is returned unchanged."""
    if pd.isna(status):
        return status
    if 'in floodplain' in str(status).lower() or 'in_floodplain' in str(status).lower():
        return 'Under DFE'
    elif 'out of floodplain' in str(status).lower() or 'out_of_floodplain' in str(status).lower():
        return 'Above DFE'
    return status


def format_currency(value):
    """Format large numbers as currency. Trailing zeros after the decimal
    point are dropped, e.g. $4.00M -> $4M but $12.79B -> $12.79B."""
    if pd.isna(value) or value == 0:
        return "$0"
    elif value >= 1e9:
        return f"${_strip_trailing_zeros(f'{value/1e9:.2f}')}B"
    elif value >= 1e6:
        return f"${_strip_trailing_zeros(f'{value/1e6:.2f}')}M"
    elif value >= 1e3:
        return f"${_strip_trailing_zeros(f'{value/1e3:.1f}')}K"
    else:
        return f"${value:.0f}"


# Damages whose absolute value is below this threshold render as "$0" in
# value labels (matches the workshop visualization convention).
ZERO_THRESH_DISPLAY = 1000


def _strip_trailing_zeros(num_str):
    """Strip trailing zeros (and a stranded '.') from a decimal number string.
    '1.00' -> '1' ; '12.50' -> '12.5' ; '12.79' -> '12.79' ; '7' -> '7'."""
    if '.' not in num_str:
        return num_str
    return num_str.rstrip('0').rstrip('.')


def fmt_money_short(value):
    """Compact currency label for chart labels/axes. Trailing zeros after the
    decimal point are dropped so '$4.00M' renders as '$4M' but '$12.79B' stays."""
    if pd.isna(value):
        return ""
    if value == 0:
        return "$0"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}${_strip_trailing_zeros(f'{v/1e9:.2f}')}B"
    elif v >= 1e6:
        return f"{sign}${_strip_trailing_zeros(f'{v/1e6:.2f}')}M"
    elif v >= 1e3:
        # $k with no decimals when >= $10k, 1 decimal otherwise for small values
        if v >= 1e4:
            return f"{sign}${v/1e3:.0f}k"
        else:
            return f"{sign}${_strip_trailing_zeros(f'{v/1e3:.1f}')}k"
    else:
        return f"{sign}${v:.0f}"


def fmt_money_rounded(value):
    """Rounded currency label for on-plot value annotations (box-and-whisker
    labels etc.). Millions and billions show ONE decimal place, with the
    decimal dropped when it's zero — e.g. $229.23M → $229.2M, $229.04M →
    $229M, $12.79B → $12.8B, $12.00B → $12B. Sub-million values are rounded
    to integers: $4.8k → $5k, $4800 → $4800. Zero renders as $0."""
    if pd.isna(value):
        return ""
    if value == 0:
        return "$0"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}${_strip_trailing_zeros(f'{v/1e9:.1f}')}B"
    elif v >= 1e6:
        return f"{sign}${_strip_trailing_zeros(f'{v/1e6:.1f}')}M"
    elif v >= 1e3:
        return f"{sign}${round(v/1e3):.0f}k"
    else:
        return f"{sign}${v:.0f}"


def nice_round_up(value):
    """Round value up to a 'nice' breakpoint of form (1, 2, 2.5, 5) × 10^k.
    Used for dynamic bin edges so legend labels look clean."""
    if value is None or pd.isna(value) or value <= 0:
        return 0
    magnitude = 10 ** np.floor(np.log10(value))
    normalized = value / magnitude
    if   normalized <= 1.0: return 1.0  * magnitude
    elif normalized <= 2.0: return 2.0  * magnitude
    elif normalized <= 2.5: return 2.5  * magnitude
    elif normalized <= 5.0: return 5.0  * magnitude
    else:                   return 10.0 * magnitude


def smart_money_ticks(max_value, target_n=5):
    """Return (tickvals, ticktext) spanning [0, max_value] with nice rounded
    tick spacing, labels formatted with fmt_money_short."""
    if max_value is None or max_value <= 0 or pd.isna(max_value):
        return [0], ["$0"]
    raw_step = max_value / max(target_n, 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized < 1.5:
        nice_step = 1 * magnitude
    elif normalized < 3.5:
        nice_step = 2 * magnitude
    elif normalized < 7.5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    ticks = []
    t = 0.0
    # include one tick past the max so the axis has headroom
    while t <= max_value * 1.02:
        ticks.append(t)
        t += nice_step
    if ticks[-1] < max_value:
        ticks.append(ticks[-1] + nice_step)
    labels = [fmt_money_short(t) for t in ticks]
    return ticks, labels


@st.cache_data
def load_csv_file(filepath):
    """Load CSV file from path"""
    return pd.read_csv(filepath)


# ----------------------------------------------------------------------------
# CSV-bundle loader (current ADAPT data format)
# ----------------------------------------------------------------------------
# The upstream pipeline ships a per-location bundle of seven files:
#
#   {LOCATION}_metadata.csv                            — key/value metadata
#   {LOCATION}_bldg_lookup.csv                         — analysis-ready bldg attrs
#   {LOCATION}_bldg_CumulativeDamage.csv               — bldg × (year, action, slr) × pcts
#   {LOCATION}_CumulativeDamage_categories.csv         — 4 leaf categories × ... × pcts
#   {LOCATION}_skipped_buildings.csv                   — provenance log (optional)
#   DDD___{LOCATION}___NSI.xlsx                        — full NSI descriptors
#   DDD___{LOCATION}_MC_annual_max_waterlevels_P50.csv — Year × MC_0001..MC_1000
#   DDD___{LOCATION}_MC_annual_max_waterlevels_P90.csv — same for high-end SLR
#
# Canonical schema (matches the current Pamunkey rerun and the format every
# location will ship in going forward):
#   * `{LOCATION}_bldg_lookup.csv` exposes the column `DFE_Status` with
#     values 'Above DFE' / 'Under DFE' directly. An extra `ManufRestriction`
#     flag may also be present for manufactured-housing-heavy inventories;
#     it is passed through but not currently used.
#   * `TargetYear` ships as integer-like strings ('2025', '2040', '2055',
#     '2100'). Categories rows use the same year column.
#   * `{LOCATION}_CumulativeDamage_categories.csv` uses the four leaves
#     RES_UnderDFE / RES_AboveDFE / NONRES_UnderDFE / NONRES_AboveDFE.
#
# Legacy compatibility shims (kept so older archived bundles still load):
#   * `Floodplain_Status` is accepted as an alias for `DFE_Status` and the
#     legacy values 'In floodplain' / 'Out of floodplain' are normalized to
#     'Under DFE' / 'Above DFE' via convert_floodplain_status.
#   * `TargetYear == 'Potential'` is mapped to 2025 via
#     _bundle_normalize_target_year.
#   * The categories file is loaded but only used for an optional sanity
#     check, so the legacy RES_InFP / RES_OutFP naming doesn't matter at
#     runtime — community aggregates are now built from the per-building
#     table (sum-of-percentiles), eliminating the ~5% reconciliation gap
#     the old format had between sum-of-medians and median-of-sum.

# Percentile manifest the bundle ships (23 columns: dense at both tails
# plus quartiles).
BUNDLE_PCT_LIST = ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P10',
                   'P25','P50','P75',
                   'P90','P91','P92','P93','P94','P95','P96','P97','P98','P99']

# Subset kept on per-building damage rows. Mirrors what the existing tabs
# read (P05/P10/P25/P50/P75/P90/P95) plus deeper tails (P01/P99) for any
# future tail-risk panel.
PER_BLDG_PCT_KEEP = ['P01','P05','P10','P25','P50','P75','P90','P95','P99']


def _bundle_read_metadata(path):
    """Parse `{LOC}_metadata.csv` (Key,Value pairs) into a dict, splitting
    list-valued fields (TARGET_YEARS, ACTION_NAMES, etc.) into Python lists."""
    raw = pd.read_csv(path)
    out = {}
    for _, row in raw.iterrows():
        k = str(row['Key']).strip()
        v = str(row['Value']).strip().strip('"')
        out[k] = v
    for list_key in ('TARGET_YEARS', 'TARGET_YEAR_LABELS', 'ACTIONS',
                     'ACTION_NAMES', 'SCENARIOS', 'SCENARIO_LABELS',
                     'PERCENTILES'):
        if list_key in out:
            out[list_key] = [s.strip() for s in out[list_key].split(',')]
    return out


def _bundle_normalize_target_year(series):
    """'Potential'→2025 + cast to int. Bundle ships TargetYear as mixed
    strings; the rest of the app uses numeric comparisons."""
    return (series.astype(str)
                  .replace({'Potential': '2025'})
                  .astype(int))


def _bundle_build_attrs_table(lookup_df, nsi_df):
    """Combine bldg_lookup (analysis-ready) with NSI (descriptive).

    Lookup is the source of truth for fields that drove the damage
    calculation (lon/lat, structure/content values, FFE, DFE
    status, SOID). NSI contributes building_type, number_of_stories, area,
    foundation_type, foundation_height, year_built, address — fields the
    lookup doesn't carry.
    """
    nsi = nsi_df.rename(columns={'ID': 'BuildingID'}).copy()
    descriptive_only = ['building_type', 'number_of_stories', 'area',
                        'foundation_type', 'foundation_height',
                        'year_built', 'address']
    desc_cols = ['BuildingID'] + [c for c in descriptive_only if c in nsi.columns]

    out = lookup_df.merge(nsi[desc_cols], on='BuildingID', how='left')

    # Lowercase rename to the schema the rest of the app already references.
    # NB: We keep `FFE_ft`, `DFE_Status`, and `SOID` in their original
    # case because the existing UI code looks them up by those exact names.
    rename = {
        'BuildingID':         'id',
        'OccupancyType':      'occupancy_type',
        'OccupancyGroup':     'occupancy_group',
        'StructureValue':     'structure_value',
        'ContentValue':       'content_value',
        'GroundElevation_ft': 'ground_elevation',
        'Longitude':          'longitude',
        'Latitude':           'latitude',
        # The canonical bundle format ships the column as `DFE_Status`
        # with canonical values ('Above DFE' / 'Under DFE'). Legacy
        # bundles (pre-rerun) shipped it as `Floodplain_Status` with
        # 'In floodplain' / 'Out of floodplain' values — we accept that
        # name as a fallback and normalize the values via
        # convert_floodplain_status below. The function is a no-op for
        # values that are already in canonical DFE form.
        'Floodplain_Status':  'DFE_Status',
    }
    out = out.rename(columns=rename)
    if 'DFE_Status' in out.columns:
        out['DFE_Status'] = out['DFE_Status'].apply(convert_floodplain_status)
    return out


def _bundle_wl_percentiles_from_mc(wl_mc):
    """Compute (Year × Pxx) percentile DataFrame from a raw MC sheet.

    The bundle ships the underlying 1,000 MC realizations (Year × MC_0001..
    MC_1000) per SLR scenario instead of pre-baked percentiles. We expose
    a percentile DataFrame for any consumer that wants P05/P50/P95-style
    bands; the raw MC matrix is also kept on the data_store for any new
    threshold-exceedance / per-building exposure feature.
    """
    mc_cols = [c for c in wl_mc.columns if c.startswith('MC_')]
    arr = wl_mc[mc_cols].to_numpy(dtype=float)
    out = pd.DataFrame({'Year': wl_mc['Year'].astype(int).values})
    for p in [1,2,3,4,5,6,7,8,9,10,25,50,75,90,91,92,93,94,95,96,97,98,99]:
        out[f'P{p:02d}'] = np.percentile(arr, p, axis=1)
    return out


# ============================================================================
# FLOOD-MAP (BATHTUB) SUPPORT — inlined so the app is a single file.
# Terrain: USGS 3DEP 1/3 arc-second (~10 m), public-domain NAVD88 metres,
# Cloud-Optimized GeoTIFFs on AWS `prd-tnm`; only the ROI window is read via
# GDAL /vsicurl HTTP range requests. Bathtub model (no hydraulic connectivity).
# To use a self-hosted topobathy COG instead, set DEM_COG_OVERRIDE_URL.
# ============================================================================
DEM_COG_URL_TEMPLATE = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
    "current/{tile}/USGS_13_{tile}.tif"
)
DEM_COG_OVERRIDE_URL = None
FT_PER_M = 1.0 / 0.3048
WS_BINS_FT = [0, 1, 2, 3, 4, 5, 6]
WS_COLORS = [
    (74, 0, 130), (31, 143, 255), (0, 204, 204),
    (255, 235, 0), (255, 140, 0), (191, 13, 13),
]


def dem_tiles_for_bbox(bbox):
    """3DEP 1-degree tile name(s) covering a lon/lat bbox (lon negative).
    Tile `nA wB` covers latitude [A-1, A], longitude [-B, -(B-1)]."""
    lon_min, lat_min, lon_max, lat_max = bbox
    a_min = int(math.floor(lat_min)) + 1
    a_max = int(math.ceil(lat_max))
    b_min = int(math.floor(-lon_max)) + 1
    b_max = int(math.ceil(-lon_min))
    return [f"n{a:02d}w{b:03d}" for a in range(a_min, a_max + 1)
            for b in range(b_min, b_max + 1)]


def roi_from_lonlat(lon, lat, buffer_m=600.0, min_span_km=1.5, max_span_km=25.0):
    """Robust bbox: median-centred, MAD outlier rejection, clamped span."""
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
    thr = max(med + 5.0 * 1.4826 * mad, 2000.0)
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
    """Swap if columns look transposed (US lon ~-65..-125, lat ~20..50)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    mlon = np.nanmedian(np.abs(lon))
    mlat = np.nanmedian(np.abs(lat))
    if np.isfinite(mlon) and np.isfinite(mlat) and mlon < 60.0 and mlat > 60.0:
        return lat, lon, True
    return lon, lat, False


def read_dem_roi(bbox, target_res_m=10.0):
    """Read the DEM over `bbox` onto a regular lon/lat grid (~target_res_m),
    fetching only the ROI window from remote COG(s) via GDAL /vsicurl.
    Returns (Z_m [row 0 = north, metres NAVD88, NaN=NoData], extent)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as transform_from_bounds

    latc = 0.5 * (lat_min + lat_max)
    dlat = target_res_m / 111320.0
    dlon = target_res_m / (111320.0 * max(0.1, math.cos(math.radians(latc))))
    nlon = max(2, int(round((lon_max - lon_min) / dlon)))
    nlat = max(2, int(round((lat_max - lat_min) / dlat)))
    longest = max(nlon, nlat)
    if longest > 2500:
        shrink = longest / 2500.0
        nlon = max(2, int(nlon / shrink))
        nlat = max(2, int(nlat / shrink))

    dst_transform = transform_from_bounds(lon_min, lat_min, lon_max, lat_max, nlon, nlat)
    dst = np.full((nlat, nlon), np.nan, dtype="float32")
    if DEM_COG_OVERRIDE_URL:
        urls = [DEM_COG_OVERRIDE_URL]
    else:
        urls = [DEM_COG_URL_TEMPLATE.format(tile=t) for t in dem_tiles_for_bbox(bbox)]

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
                    with WarpedVRT(src, crs="EPSG:4326", transform=dst_transform,
                                   width=nlon, height=nlat,
                                   resampling=Resampling.bilinear) as vrt:
                        arr = vrt.read(1).astype("float32")
                if src_nodata is not None:
                    arr[arr == src_nodata] = np.nan
                arr[arr <= -500.0] = np.nan
                fill = np.isnan(dst) & np.isfinite(arr)
                dst[fill] = arr[fill]
                any_ok = True
            except Exception as exc:                # noqa: BLE001
                errors.append(f"{vsi.split('/')[-1]}: {exc}")
    if not any_ok:
        raise RuntimeError(
            "Could not read any DEM tile for this area. Tried: %s. Errors: %s"
            % (", ".join(u.split("/")[-1] for u in urls), " | ".join(errors))
        )
    return dst, (lon_min, lat_min, lon_max, lat_max)


def bathtub_depth_ft(Z_m, wl_ft, mask_water=False):
    """Bathtub depth (ft) for a water level (ft NAVD88); NaN where dry.
    By default masks only NoData, so low-lying land below NAVD88 zero still
    floods. With mask_water=True, also hides Z<=0 (treats it as permanent
    open water) for a cleaner look on a land-only DEM."""
    wl_m = float(wl_ft) * 0.3048
    depth_m = wl_m - Z_m
    invalid = ~np.isfinite(Z_m)
    if mask_water:
        invalid = invalid | (Z_m <= 0.0)
    depth_m = np.where(invalid, np.nan, depth_m)
    depth_m = np.where(depth_m < 0.0, np.nan, depth_m)
    return depth_m.astype("float32") * FT_PER_M


def depth_to_rgba_data_uri(depth_ft):
    """Depth-in-feet array -> base64 PNG data URI (RGBA, discrete bins)."""
    from PIL import Image
    import io
    import base64
    h, w = depth_ft.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    n = len(WS_COLORS)
    for b, (r, g, bl) in enumerate(WS_COLORS):
        if b < n - 1:
            mm = (depth_ft >= WS_BINS_FT[b]) & (depth_ft < WS_BINS_FT[b + 1])
        else:
            mm = depth_ft >= WS_BINS_FT[b]
        rgba[mm, 0] = r
        rgba[mm, 1] = g
        rgba[mm, 2] = bl
        rgba[mm, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def mapbox_zoom_for_bbox(extent, width_px=340, height_px=420, pad=1.06):
    """Approximate a Plotly mapbox zoom that frames the bbox in a grid panel.
    Web-Mercator: world is square (360 deg per 512 px tile at z0); latitude
    needs a cos(lat) term. width/height default to a 2-column panel."""
    lon_min, lat_min, lon_max, lat_max = extent
    latc = 0.5 * (lat_min + lat_max)
    lon_span = max((lon_max - lon_min) * pad, 1e-4)
    lat_span = max((lat_max - lat_min) * pad, 1e-4)
    z_lon = math.log2(360.0 / lon_span) + math.log2(max(width_px, 1) / 512.0)
    z_lat = (math.log2(360.0 / lat_span) + math.log2(max(height_px, 1) / 512.0)
             + math.log2(max(0.05, math.cos(math.radians(latc)))))
    return max(1.0, min(16.0, min(z_lon, z_lat)))


def legend_html(area_note=None):
    """Large, readable inline HTML legend for the discrete depth bins (feet)."""
    labels = ["0–1", "1–2", "2–3", "3–4", "4–5", "5+"]
    items = []
    for (r, g, b), lab in zip(WS_COLORS, labels):
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:24px;margin-bottom:8px;">'
            f'<span style="width:26px;height:26px;background:rgb({r},{g},{b});'
            f'display:inline-block;margin-right:9px;border:1px solid #888;border-radius:4px;"></span>'
            f'<span style="font-size:1.15rem;color:#1f2937;">{lab} ft</span></span>'
        )
    extra = ""
    if area_note:
        extra = (f'<div style="font-size:1.1rem;color:#374151;margin-top:4px;">{area_note}</div>')
    return (
        '<div style="margin:0.5rem 0 0.9rem;padding:0.6rem 0.8rem;background:#f8fafc;'
        'border:1px solid #e2e8f0;border-radius:8px;">'
        '<span style="font-size:1.4rem;font-weight:700;color:#0f172a;margin-right:16px;">Flood depth</span>'
        '<span style="display:inline-flex;flex-wrap:wrap;align-items:center;vertical-align:middle;">'
        + "".join(items) + "</span>" + extra + "</div>"
    )


def depth_to_rgba(depth_ft, alpha=217):
    """Depth-in-feet array -> (H,W,4) uint8 RGBA (discrete bins; dry transparent)."""
    h, w = depth_ft.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    n = len(WS_COLORS)
    for b, (r, g, bl) in enumerate(WS_COLORS):
        if b < n - 1:
            mm = (depth_ft >= WS_BINS_FT[b]) & (depth_ft < WS_BINS_FT[b + 1])
        else:
            mm = depth_ft >= WS_BINS_FT[b]
        rgba[mm, 0] = r
        rgba[mm, 1] = g
        rgba[mm, 2] = bl
        rgba[mm, 3] = alpha
    return rgba


_TILE_PROVIDERS = {
    "OSM (color)": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Light": "https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png",
    "Dark": "https://basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
}


def _lonlat_to_tilexy(lon, lat, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_zoom_for_bbox(bbox, target_px=1000, tile_px=256, max_zoom=18):
    """Integer slippy zoom so the bbox renders ~target_px wide."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_span = max(lon_max - lon_min, 1e-6)
    z = math.log2(target_px * 360.0 / (tile_px * lon_span))
    return int(max(1, min(max_zoom, round(z))))


def fetch_basemap(bbox, zoom, provider_label, tile_px=256, max_tiles=40):
    """Fetch & stitch XYZ tiles for bbox at integer slippy zoom, cropped to bbox.
    Returns (rgb uint8 [H,W,3], bbox). Fetched SERVER-SIDE with a User-Agent,
    which avoids OpenStreetMap's browser 403 ('Referer required by tile policy')
    and uses no WebGL (so any number of maps can be shown). The zoom is reduced
    if needed so no more than `max_tiles` tiles are requested (courtesy/speed)."""
    import io as _io
    import requests
    from PIL import Image
    lon_min, lat_min, lon_max, lat_max = bbox
    url_tmpl = _TILE_PROVIDERS.get(provider_label, _TILE_PROVIDERS["OSM (color)"])
    # Keep the tile count modest by dropping a zoom level if the span is large.
    while zoom > 4:
        _x0, _y0 = _lonlat_to_tilexy(lon_min, lat_max, zoom)
        _x1, _y1 = _lonlat_to_tilexy(lon_max, lat_min, zoom)
        _c = int(math.floor(_x1)) - int(math.floor(_x0)) + 1
        _r = int(math.floor(_y1)) - int(math.floor(_y0)) + 1
        if _c * _r <= max_tiles:
            break
        zoom -= 1
    x0f, y0f = _lonlat_to_tilexy(lon_min, lat_max, zoom)   # NW corner -> top-left
    x1f, y1f = _lonlat_to_tilexy(lon_max, lat_min, zoom)   # SE corner -> bottom-right
    xt0, yt0 = int(math.floor(x0f)), int(math.floor(y0f))
    xt1, yt1 = int(math.floor(x1f)), int(math.floor(y1f))
    nmax = int(2 ** zoom) - 1
    xt0 = max(0, xt0); yt0 = max(0, yt0)
    xt1 = min(nmax, xt1); yt1 = min(nmax, yt1)
    cols = xt1 - xt0 + 1
    rows = yt1 - yt0 + 1
    mosaic = Image.new("RGB", (cols * tile_px, rows * tile_px), (235, 235, 235))
    headers = {"User-Agent": "ADAPT-FloodTool/1.0 (Columbia CCSR research app)"}
    for ix, xt in enumerate(range(xt0, xt1 + 1)):
        for iy, yt in enumerate(range(yt0, yt1 + 1)):
            try:
                resp = requests.get(url_tmpl.format(z=zoom, x=xt, y=yt),
                                    headers=headers, timeout=12)
                if resp.status_code == 200:
                    tile = Image.open(_io.BytesIO(resp.content)).convert("RGB")
                    mosaic.paste(tile, (ix * tile_px, iy * tile_px))
            except Exception:                              # noqa: BLE001
                pass
    left = (x0f - xt0) * tile_px
    top = (y0f - yt0) * tile_px
    right = (x1f - xt0) * tile_px
    bottom = (y1f - yt0) * tile_px
    crop = mosaic.crop((int(round(left)), int(round(top)),
                        max(int(round(right)), int(round(left)) + 1),
                        max(int(round(bottom)), int(round(top)) + 1)))
    return np.asarray(crop, dtype=np.uint8), (lon_min, lat_min, lon_max, lat_max)


def compose_flood_png(basemap_rgb, depth_ft):
    """Alpha-composite the flood overlay onto the basemap; return PNG bytes."""
    import io as _io
    from PIL import Image
    base = Image.fromarray(np.asarray(basemap_rgb, dtype=np.uint8), "RGB").convert("RGBA")
    overlay = Image.fromarray(depth_to_rgba(depth_ft), "RGBA").resize(base.size, Image.NEAREST)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    buf = _io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def fetch_osm_roads(bbox, timeout=60):
    """Query the Overpass API for 'highway' ways in bbox (server-side).
    Returns a list of dicts: {'coords' Nx2 (lon,lat), 'name', 'ref', 'highway'}.
    Mirrors download_osm_roads_pamunkey.py but runs live for the map ROI."""
    import urllib.request
    import urllib.parse
    import json as _json
    lon_min, lat_min, lon_max, lat_max = bbox
    query = (f"[out:json][timeout:{timeout}];"
             f'way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});'
             f"out body geom;")
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter", data=data,
        headers={"User-Agent": "ADAPT-FloodTool/1.0 (Columbia CCSR research app)"})
    with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
        raw = _json.loads(resp.read().decode("utf-8"))
    roads = []
    for el in raw.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = np.array([[p["lon"], p["lat"]] for p in el["geometry"]], dtype=float)
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        roads.append({
            "coords": coords,
            "name": tags.get("name", "") or "",
            "ref": tags.get("ref", "") or "",
            "highway": tags.get("highway", "unknown"),
        })
    return roads


def sample_dem_bilinear(Zm, ext, lons, lats):
    """Bilinear-sample the DEM (row 0 = north) at arrays of lon/lat; NaN off-grid."""
    lon_min, lat_min, lon_max, lat_max = ext
    nlat, nlon = Zm.shape
    fc = (np.asarray(lons, float) - lon_min) / max(lon_max - lon_min, 1e-9) * (nlon - 1)
    fr = (lat_max - np.asarray(lats, float)) / max(lat_max - lat_min, 1e-9) * (nlat - 1)
    out = np.full(fc.shape, np.nan, dtype=float)
    inb = (fc >= 0) & (fc <= nlon - 1) & (fr >= 0) & (fr <= nlat - 1)
    if inb.any():
        c0 = np.floor(fc[inb]).astype(int)
        r0 = np.floor(fr[inb]).astype(int)
        c1 = np.minimum(c0 + 1, nlon - 1)
        r1 = np.minimum(r0 + 1, nlat - 1)
        tx = fc[inb] - c0
        ty = fr[inb] - r0
        z00 = Zm[r0, c0]; z01 = Zm[r0, c1]; z10 = Zm[r1, c0]; z11 = Zm[r1, c1]
        top = z00 * (1 - tx) + z01 * tx
        bot = z10 * (1 - tx) + z11 * tx
        out[inb] = top * (1 - ty) + bot * ty
    return out


def _dilate_mask(mask, iters):
    """8-connectivity binary dilation by `iters` cells (numpy; no scipy)."""
    m = mask.copy()
    for _ in range(max(0, int(iters))):
        d = m.copy()
        d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
        d[1:, 1:] |= m[:-1, :-1]; d[:-1, :-1] |= m[1:, 1:]
        d[1:, :-1] |= m[:-1, 1:]; d[:-1, 1:] |= m[1:, :-1]
        m = d
    return m


def classify_roads(Zm, ext, roads, wl_ft, prox_m=30.0, sample_m=8.0):
    """Classify road samples as 0 dry / 1 proximate / 2 flooded for a water level
    (ft NAVD88). Open water (Z<=0) is excluded from the flood mask, matching the
    flood-map tab. Returns (segments, counts) where segments is a list of
    ((lon0,lat0),(lon1,lat1),status)."""
    nlat, nlon = Zm.shape
    lon_min, lat_min, lon_max, lat_max = ext
    wl_m = float(wl_ft) * 0.3048
    Zmask = np.where(np.isnan(Zm), 9999.0, Zm)
    flood = (wl_m - Zmask > 0) & (Zmask > 0)
    # proximity buffer in grid cells from the cell size
    dy_m = (lat_max - lat_min) * 111320.0 / max(nlat, 1)
    dx_m = (lon_max - lon_min) * 111320.0 * math.cos(math.radians(0.5 * (lat_min + lat_max))) / max(nlon, 1)
    prox_cells = max(1, int(math.ceil(prox_m / max(min(dx_m, dy_m), 1e-6))))
    prox = _dilate_mask(flood, prox_cells) & (~flood) & (Zmask > 0)

    def to_rc(lons, lats):
        c = np.clip(((lons - lon_min) / max(lon_max - lon_min, 1e-9) * (nlon - 1)).round().astype(int), 0, nlon - 1)
        r = np.clip(((lat_max - lats) / max(lat_max - lat_min, 1e-9) * (nlat - 1)).round().astype(int), 0, nlat - 1)
        return r, c

    segs = []
    nf = npx = nd = 0
    for rd in roads:
        coords = rd["coords"]
        lons, lats = coords[:, 0], coords[:, 1]
        # keep roads that intersect the ROI
        if not np.any((lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)):
            continue
        dlat = np.diff(lats) * 111320.0
        dlon = np.diff(lons) * 111320.0 * math.cos(math.radians(float(np.mean(lats))))
        seglen = np.sqrt(dlat ** 2 + dlon ** 2)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total = float(cum[-1])
        if total < 1.0:
            continue
        n = max(int(total / sample_m), 2)
        sd = np.linspace(0.0, total, n)
        slat = np.interp(sd, cum, lats)
        slon = np.interp(sd, cum, lons)
        r, c = to_rc(slon, slat)
        status = np.zeros(n, dtype=int)
        status[prox[r, c]] = 1
        status[flood[r, c]] = 2
        for i in range(n - 1):
            s = int(max(status[i], status[i + 1]))
            segs.append(((float(slon[i]), float(slat[i])), (float(slon[i + 1]), float(slat[i + 1])), s))
            if s == 2:
                nf += 1
            elif s == 1:
                npx += 1
            else:
                nd += 1
    tot = nf + npx + nd
    counts = {
        "flood": nf, "prox": npx, "dry": nd, "total": tot,
        "pct_flood": 100.0 * nf / tot if tot else 0.0,
        "pct_prox": 100.0 * npx / tot if tot else 0.0,
        "pct_dry": 100.0 * nd / tot if tot else 0.0,
    }
    return segs, counts


def compose_road_png(basemap_rgb, depth_ft, segments, ext):
    """Basemap + flood-depth overlay + colored road segments -> PNG bytes.
    Segment colors: green dry, orange proximate (<buffer), red flooded."""
    import io as _io
    from PIL import Image, ImageDraw
    base = Image.fromarray(np.asarray(basemap_rgb, dtype=np.uint8), "RGB").convert("RGBA")
    W, H = base.size
    overlay = Image.fromarray(depth_to_rgba(depth_ft, alpha=150), "RGBA").resize((W, H), Image.NEAREST)
    base = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(base)
    lon_min, lat_min, lon_max, lat_max = ext

    def px(lon, lat):
        x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * (W - 1)
        y = (lat_max - lat) / max(lat_max - lat_min, 1e-9) * (H - 1)
        return (x, y)

    colors = {0: (34, 139, 34, 235), 1: (255, 140, 0, 240), 2: (220, 20, 20, 250)}
    base_w = max(2, int(round(W / 450)))
    widths = {0: base_w, 1: base_w + 1, 2: base_w + 2}
    for status in (0, 1, 2):                       # dry, then proximate, then flooded on top
        col = colors[status]
        w = widths[status]
        for p0, p1, s in segments:
            if s != status:
                continue
            draw.line([px(*p0), px(*p1)], fill=col, width=w)
    out = base.convert("RGB")
    buf = _io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# Namespace so the Flood Maps tab can keep calling fdem.<fn>(...).
fdem = SimpleNamespace(
    dem_tiles_for_bbox=dem_tiles_for_bbox,
    roi_from_lonlat=roi_from_lonlat,
    maybe_swap_lonlat=maybe_swap_lonlat,
    read_dem_roi=read_dem_roi,
    bathtub_depth_ft=bathtub_depth_ft,
    depth_to_rgba=depth_to_rgba,
    tile_zoom_for_bbox=tile_zoom_for_bbox,
    fetch_basemap=fetch_basemap,
    compose_flood_png=compose_flood_png,
    fetch_osm_roads=fetch_osm_roads,
    sample_dem_bilinear=sample_dem_bilinear,
    classify_roads=classify_roads,
    compose_road_png=compose_road_png,
    legend_html=legend_html,
)


@st.cache_data(show_spinner=False)
def _cached_dem_roi(bbox, res_m):
    """Cache wrapper around read_dem_roi (keyed on rounded bbox)."""
    return fdem.read_dem_roi(bbox, res_m)


@st.cache_data(show_spinner=False)
def _cached_basemap(bbox, zoom, provider_label):
    """Cache wrapper around fetch_basemap (one fetch per ROI/zoom/provider)."""
    return fdem.fetch_basemap(bbox, zoom, provider_label)


@st.cache_data(show_spinner=False)
def _cached_osm_roads(bbox):
    """Cache wrapper around fetch_osm_roads (one Overpass query per ROI)."""
    return fdem.fetch_osm_roads(bbox)


@st.cache_data(show_spinner=False)
def load_bundle(data_folder, location_slug):
    """Load a single-location CSV bundle and return the data_store entry."""
    join = lambda *p: os.path.join(data_folder, *p)

    # 1. Metadata
    metadata = _bundle_read_metadata(join(f'{location_slug}_metadata.csv'))
    bfe_ft = float(metadata.get('BFE_FT_NAVD88', 9))

    # Map TargetYear int → display label (2025 → 'Potential', etc.)
    target_year_labels = {}
    if ('TARGET_YEARS' in metadata
            and 'TARGET_YEAR_LABELS' in metadata
            and len(metadata['TARGET_YEARS']) == len(metadata['TARGET_YEAR_LABELS'])):
        for ys, lab in zip(metadata['TARGET_YEARS'], metadata['TARGET_YEAR_LABELS']):
            try:
                target_year_labels[int(ys)] = lab
            except ValueError:
                continue

    # 2. NSI (271 rows incl. the building that gets skipped — joined later)
    nsi = pd.read_excel(join(f'DDD___{location_slug}___NSI.xlsx'))

    # 3. Building lookup (270 rows — only buildings that were actually analyzed)
    lookup = pd.read_csv(join(f'{location_slug}_bldg_lookup.csv'))
    bldg_attrs = _bundle_build_attrs_table(lookup, nsi)

    # 4. Skipped-buildings log (small, possibly empty)
    try:
        skipped = pd.read_csv(join(f'{location_slug}_skipped_buildings.csv'))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        skipped = pd.DataFrame()

    # 5. Per-building cumulative damage
    bldg_dmg = pd.read_csv(join(f'{location_slug}_bldg_CumulativeDamage.csv'))
    bldg_dmg = bldg_dmg.rename(columns={'BuildingID': 'id'})
    bldg_dmg['TargetYear'] = _bundle_normalize_target_year(bldg_dmg['TargetYear'])
    pct_present = [p for p in BUNDLE_PCT_LIST if p in bldg_dmg.columns]
    pct_keep = [p for p in PER_BLDG_PCT_KEEP if p in pct_present]
    bldg_dmg = bldg_dmg.drop(columns=[p for p in pct_present if p not in pct_keep])
    bldg_dmg = bldg_dmg.rename(columns={p: f'CumEAD_{p}' for p in pct_keep})

    # ---- Defensive correction for MATLAB Elevate no-op artifact ----
    # The MATLAB damage generator (AAA___Main_v9.m) skips the elevation math
    # for buildings tagged `DFE_Status == "Above DFE"` and just
    # sets `Dmg(:,:,4,si) = dmg_baseline` — i.e. Elevate equals No mitigation
    # exactly, by construction. The DFE_Status flag is itself derived
    # from the building's first-floor elevation (FFE) being above the design
    # flood elevation (DFE = BFE + 2 ft), but FFE = ground + foundation height,
    # so a building can sit on low ground (yard/garage flood frequently) and
    # still be tagged Above-DFE because its raised foundation lifts
    # the FFE above the DFE. The combined effect: a non-trivial number of
    # Above-DFE buildings have substantial baseline damage AND a no-op'd Elevate
    # column, which makes them look like "Residual: even elevation can't help"
    # on the Adaptation Effectiveness map — when in reality elevation simply
    # wasn't computed for them. We detect the no-op (Elevate exactly equals
    # No-mitigation across every percentile column for a given (building,
    # year, SLR)) and replace the affected Elevate row's percentile values
    # with NaN. Downstream code already handles NaN as "retrofit not
    # available" rather than "elevation is fully ineffective", so the map
    # and the Distributions plots stop labeling these buildings Residual on
    # spurious grounds.
    pct_cols_in_bldg = [c for c in bldg_dmg.columns if c.startswith('CumEAD_P')]
    if 'Action' in bldg_dmg.columns and pct_cols_in_bldg:
        # Build a No-mitigation lookup by (id, TargetYear, SLR)
        nm_lookup = (bldg_dmg[bldg_dmg['Action'] == 'No mitigation']
                     [['id', 'TargetYear', 'SLR'] + pct_cols_in_bldg]
                     .rename(columns={c: f'_NM_{c}' for c in pct_cols_in_bldg}))
        bldg_dmg = bldg_dmg.merge(nm_lookup, on=['id', 'TargetYear', 'SLR'],
                                  how='left')
        # Compare Elevate row by row against its No-Mit twin. We use exact
        # equality across ALL percentile columns simultaneously — anything
        # else is an actual elevation calculation, even if it happens to be
        # numerically close to baseline. exact-match catches the no-op.
        is_elev = bldg_dmg['Action'] == 'Elevate'
        is_noop = is_elev.copy()
        for c in pct_cols_in_bldg:
            nm_col = f'_NM_{c}'
            # Treat both NaN as no-evidence-of-difference (don't trip noop on missing)
            same = (bldg_dmg[c] == bldg_dmg[nm_col]) | \
                   (bldg_dmg[c].isna() & bldg_dmg[nm_col].isna())
            is_noop &= same
        # Also require at least one column to be non-zero — a real
        # all-zeros Elevate row (no damage to begin with) shouldn't be
        # flagged as no-op.
        any_nonzero = pd.Series(False, index=bldg_dmg.index)
        for c in pct_cols_in_bldg:
            any_nonzero |= bldg_dmg[c].fillna(0) > 0
        is_noop &= any_nonzero
        n_noop = int(is_noop.sum())
        if n_noop > 0:
            # Replace the percentile values on no-op'd Elevate rows with NaN
            for c in pct_cols_in_bldg:
                bldg_dmg.loc[is_noop, c] = np.nan
            # Stash the count on the entry so we can surface it in the UI
            _matlab_noop_count = n_noop
        else:
            _matlab_noop_count = 0
        # Drop the temporary lookup columns
        bldg_dmg = bldg_dmg.drop(columns=[f'_NM_{c}' for c in pct_cols_in_bldg])
    else:
        _matlab_noop_count = 0

    # Merge attrs onto every (id, year, action, slr) row so each row carries
    # the descriptive context the Map / Details / Distributions tabs need.
    df_buildings = bldg_dmg.merge(bldg_attrs, on='id', how='left')

    # ------------------------------------------------------------------
    # Drop (building, action) rows for retrofits that don't physically
    # apply to the building.
    # ------------------------------------------------------------------
    # The upstream damage generator runs the math for every retrofit on
    # every building, regardless of whether the retrofit makes physical
    # sense for that building (e.g., it computes WFP Basement damage for
    # buildings with no basement, and Elevate damage for buildings
    # already above DFE). Those numbers are then displayed as if they
    # were real adaptation options, producing misleading entries like
    # "WFP Basement: $0 (-100%)" on a manufactured home sitting on piers.
    #
    # We fix this at the data layer: any (building, action) row where
    # the action doesn't physically apply is simply removed from
    # df_buildings. That makes the action vanish from the hover, the
    # map classification, the dropdowns, the box plots, the
    # trajectories, and every aggregate downstream — one filter, one
    # consistent story across the whole app. The wide-form pivot in
    # prepare_map_data() will produce NaN for the dropped (building,
    # action) cells, and the hover loop below skips NaN cells so
    # they don't render as a misleading $0.
    #
    # Vectorized so this stays cheap on multi-thousand-building inventories.
    if 'foundation_type' in df_buildings.columns:
        _foundation_norm = (
            df_buildings['foundation_type']
            .fillna('').astype(str).str.strip().str.upper()
        )
        _is_basement = _foundation_norm.isin(_BASEMENT_FOUNDATION_CODES)
        # Treat truly-missing foundation_type as "not basement" so we
        # don't surface WFP B for buildings we can't verify have one.
        _is_basement &= df_buildings['foundation_type'].notna()
    else:
        # No foundation column at all → no building can be confirmed as
        # having a basement, so WFP B applies nowhere.
        _is_basement = pd.Series(False, index=df_buildings.index)

    if 'DFE_Status' in df_buildings.columns:
        _dfe_norm = (
            df_buildings['DFE_Status']
            .fillna('').astype(str).str.strip().str.lower()
        )
        _is_above_dfe = _dfe_norm.isin(_ABOVE_DFE_STATUS_STRINGS)
    else:
        _is_above_dfe = pd.Series(False, index=df_buildings.index)

    _action = df_buildings['Action']
    _keep = pd.Series(True, index=df_buildings.index)
    _keep &= ~((_action == 'WFP B')   & ~_is_basement)
    _keep &= ~((_action == 'Elevate') &  _is_above_dfe)

    _n_dropped_wfpb = int(((_action == 'WFP B')   & ~_is_basement).sum())
    _n_dropped_elev = int(((_action == 'Elevate') &  _is_above_dfe).sum())
    df_buildings = df_buildings[_keep].copy()
    # Stash counts on the loader so the UI can report them if useful.
    # (Not currently surfaced, but cheap to keep available.)
    _applicability_drop_counts = {
        'WFP B (no basement)':   _n_dropped_wfpb,
        'Elevate (above DFE)':   _n_dropped_elev,
    }

    # 6. Aggregate community-total damage tables per occupancy filter.
    #
    # Earlier versions of this loader sourced the community totals from the
    # `{LOC}_CumulativeDamage_categories.csv` rollup file. That file is
    # supposed to give MC-correct community percentiles (computed from sums
    # of MC realizations across buildings, not from summing per-building
    # percentiles), but real data sometimes ships with the No-Mitigation
    # rows zeroed out — which makes the Summary metric read $0 while the
    # retrofit rows still hold real values, and the "Damage Reduction"
    # chart then plots NEGATIVE reductions because it computes
    # `baseline − retrofit = 0 − $235M`. To keep the Summary, the box-
    # plot, and the Map/Details visualizations all consistent, we now
    # build the aggregate **directly from the per-building damage table**
    # (the same source the per-building visuals read from). The
    # statistical concession is that we use sum-of-percentiles rather
    # than percentile-of-sum for the tails (P05/P95), which slightly
    # overstates community tails — but consistency is far more important
    # for decision support than that small bias, and the median (P50) is
    # essentially unaffected.
    is_res = bldg_attrs['occupancy_type'].apply(is_residential)
    bldg_counts = {
        'All':             int(len(bldg_attrs)),
        'Residential':     int(is_res.sum()),
        'Non-Residential': int((~is_res).sum()),
    }
    res_ids = set(bldg_attrs.loc[is_res, 'id'].astype(int).tolist())
    nonres_ids = set(bldg_attrs.loc[~is_res, 'id'].astype(int).tolist())
    occ_id_map = {
        'All':             None,        # no filter
        'Residential':     res_ids,
        'Non-Residential': nonres_ids,
    }

    grp_cols = ['TargetYear', 'Action', 'SLR']
    pct_cols_in_bldg = [c for c in df_buildings.columns
                        if c.startswith('CumEAD_P')]

    agg_by_occ = {}
    for occ, ids_filter in occ_id_map.items():
        if ids_filter is None:
            df_o = df_buildings
        else:
            if not ids_filter:
                agg_by_occ[occ] = pd.DataFrame()
                continue
            df_o = df_buildings[df_buildings['id'].astype(int).isin(ids_filter)]

        if df_o.empty or not pct_cols_in_bldg:
            agg_by_occ[occ] = pd.DataFrame()
            continue

        total = df_o.groupby(grp_cols, as_index=False)[pct_cols_in_bldg].sum()
        total = total.rename(columns={
            c: f"Total_CumEAD_{c.split('_')[1]}" for c in pct_cols_in_bldg
        })

        # InFP / OutFP P50 split from DFE_Status. The split sums
        # to Total_CumEAD_P50 by construction (every building falls into
        # exactly one DFE bucket), so the metric reconciles cleanly.
        if 'DFE_Status' in df_o.columns:
            df_in = df_o[df_o['DFE_Status'] == 'Under DFE']
            df_out = df_o[df_o['DFE_Status'] == 'Above DFE']
            in_p50 = (df_in.groupby(grp_cols, as_index=False)['CumEAD_P50']
                      .sum()
                      .rename(columns={'CumEAD_P50': 'InFP_CumEAD_P50'})
                      if not df_in.empty
                      else pd.DataFrame(columns=grp_cols + ['InFP_CumEAD_P50']))
            out_p50 = (df_out.groupby(grp_cols, as_index=False)['CumEAD_P50']
                       .sum()
                       .rename(columns={'CumEAD_P50': 'OutFP_CumEAD_P50'})
                       if not df_out.empty
                       else pd.DataFrame(columns=grp_cols + ['OutFP_CumEAD_P50']))
            merged = (total.merge(in_p50, on=grp_cols, how='left')
                           .merge(out_p50, on=grp_cols, how='left'))
            merged['InFP_CumEAD_P50']  = merged['InFP_CumEAD_P50'].fillna(0.0)
            merged['OutFP_CumEAD_P50'] = merged['OutFP_CumEAD_P50'].fillna(0.0)
        else:
            merged = total.copy()
            merged['InFP_CumEAD_P50']  = 0.0
            merged['OutFP_CumEAD_P50'] = 0.0

        merged['Num_Buildings'] = bldg_counts[occ]
        agg_by_occ[occ] = merged.reset_index(drop=True)

    # Optional sanity comparison: if the categories CSV is present, log
    # how its 'All' P50 differs from the per-building P50 we just built.
    # We don't surface this in the UI because the per-building source
    # is now canonical — the comparison just helps when debugging
    # data-pipeline issues for a new location.
    cat_csv_path = join(f'{location_slug}_CumulativeDamage_categories.csv')
    cat = None
    try:
        cat = pd.read_csv(cat_csv_path)
        cat['TargetYear'] = _bundle_normalize_target_year(cat['TargetYear'])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    # 7. Water levels — load raw MC + pre-compute percentile shim
    water_levels = {}
    for slr_key, fname_suffix in (('50th-percentile', 'P50'),
                                   ('90th-percentile', 'P90')):
        wl_path = join(f'DDD___{location_slug}_MC_annual_max_waterlevels_{fname_suffix}.csv')
        try:
            wl_mc = pd.read_csv(wl_path)
        except FileNotFoundError:
            continue
        if 'Year' not in wl_mc.columns:
            continue
        water_levels[slr_key] = _bundle_wl_percentiles_from_mc(wl_mc)
        # Raw MC ensemble, suffixed `_mc` so legacy code paths that iterate
        # `for slr in water_levels` and assume percentile rows ignore them.
        water_levels[f'{slr_key}_mc'] = wl_mc

    return {
        'buildings':          df_buildings,
        'agg':                agg_by_occ.get('All'),
        'agg_by_occ':         agg_by_occ,
        'bldg_attrs':         bldg_attrs,
        'water_levels':       water_levels,
        'metadata':           metadata,
        'skipped':            skipped,
        'bfe_ft':             bfe_ft,
        'target_year_labels': target_year_labels,
        'format':             'bundle',
        'location_slug':      location_slug,
        'matlab_elev_noop_rows': _matlab_noop_count,
    }


def _bundle_pretty_location_name(slug):
    """Turn 'MasticBeach' into 'Mastic Beach'. Falls back to a sensible
    rendering of the slug (CamelCase-split or underscore-replaced) when
    the slug isn't in the known-locations table."""
    pretty = parse_filename(slug + '.csv')
    # parse_filename's known-pattern path returns names with spaces
    # ('Mastic Beach'); its fallback path returns the slug unchanged. We
    # only trust the result when it actually inserted a space — otherwise
    # we run our own CamelCase-aware splitter to handle slugs the table
    # doesn't know about.
    if pretty != 'Unknown Location' and ' ' in pretty:
        return pretty
    # Generic fallback: replace underscores with spaces, then split CamelCase
    # boundaries (keeps acronyms intact: 'NYCBay' → 'NYC Bay').
    import re
    s = slug.replace('_', ' ')
    s = re.sub(r'(?<=[a-z])([A-Z])', r' \1', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.strip()


def load_data_from_folder(data_folder="."):
    """Discover bundles in `data_folder` and return the data_store.

    A bundle is identified by the presence of `{slug}_metadata.csv` plus
    the four required companion files. Locations missing any required
    file are skipped silently — we don't claim partial bundles.
    """
    data_store = {}
    available_locations = set()

    if not os.path.exists(data_folder):
        return data_store, []

    metas = sorted(glob.glob(os.path.join(data_folder, '*_metadata.csv')))
    for meta_path in metas:
        slug = os.path.basename(meta_path)[:-len('_metadata.csv')]
        required = [
            f'{slug}_bldg_lookup.csv',
            f'{slug}_bldg_CumulativeDamage.csv',
            f'{slug}_CumulativeDamage_categories.csv',
            f'DDD___{slug}___NSI.xlsx',
        ]
        if not all(os.path.exists(os.path.join(data_folder, r)) for r in required):
            continue
        try:
            entry = load_bundle(data_folder, slug)
        except Exception as e:
            # Don't kill the app on a malformed bundle — log and skip.
            print(f"[loader] failed to load '{slug}': {e}")
            continue
        # Use the metadata's own LOCATION when present; fall back to slug.
        location_name = entry['metadata'].get('LOCATION', '').strip().strip('"')
        if not location_name:
            location_name = _bundle_pretty_location_name(slug)
        else:
            # Even when LOCATION is set, prefer the human-readable form
            # from parse_filename if it knows the slug (e.g. MasticBeach →
            # Mastic Beach).
            pretty = _bundle_pretty_location_name(slug)
            if pretty != slug:
                location_name = pretty
        data_store[location_name] = entry
        available_locations.add(location_name)

    return data_store, sorted(list(available_locations))


def compute_damage_bin_breaks(df_buildings, scenario,
                              p_breaks=(0.20, 0.40, 0.60, 0.80),
                              thr=1000.0):
    """Compute stable damage-bin breakpoints across ALL years for the given
    SLR scenario. The breaks come from the pooled distribution of nonzero
    No-Mitigation P90 damages across every year, so the same building gets
    the same color regardless of which year is selected.
    
    This guarantees that switching the year on the Damage Bins map
    redistributes buildings *across* the same bins, rather than redefining
    the bins themselves — which makes year-to-year comparisons meaningful.
    
    The upper-tail percentile used here is **P90**, matching the workshop
    convention and the Distributions tab's "Building Counts by Adaptation
    Effectiveness" classifier — so the same building gets the same
    Damage-Bins color, the same Adaptation-Effectiveness category, and the
    same Distributions-tab bucket.

    When `nice_round_up` snapping causes two or more adjacent quantile
    breaks to collapse to the same value (common at locations like Pamunkey
    where most damages are clustered in a small dollar range), we return
    FEWER unique breaks rather than extrapolating upward. The earlier
    "pad-upward" behavior invented top breakpoints beyond the actual data
    range (e.g., a $25k top break for a dataset that maxes out around $12k),
    which inflated the legend and pushed mid-range buildings into a
    "red" bin despite being entirely typical for the location. Accepting
    a smaller number of bins is honest about what the data supports and
    keeps the downstream palette/labels/digitize logic intact (they all
    work with any 1–4 unique breaks → 2–5 bins).
    """
    df_nm = df_buildings[
        (df_buildings['Action'] == 'No mitigation') &
        (df_buildings['SLR'] == scenario)
    ]
    if df_nm.empty:
        return None
    
    pooled = df_nm['CumEAD_P90'].values.astype(float)
    pooled = pooled[~np.isnan(pooled)]
    nonzero = pooled[pooled > thr]
    if len(nonzero) == 0:
        return None
    
    raw = np.quantile(nonzero, list(p_breaks))
    nice = [nice_round_up(v) for v in raw]
    seen, unique_nice = set(), []
    for v in nice:
        if v > 0 and v not in seen:
            seen.add(v); unique_nice.append(v)
    return sorted(unique_nice)


def prepare_map_data(df_buildings, target_year, scenario):
    """Prepare building data for map display.
    
    Pulls per-action P05/P50/P90/P95 columns from the long-format
    per-building damage frame and pivots them into one wide row per
    building with `{action}_P05`, `{action}_P50`, `{action}_P90`,
    `{action}_P95` columns. P90 is the upper-tail proxy used by the
    Damage Bins and Adaptation Effectiveness map views (the Distributions
    tab uses the same convention); P95 is kept for any view that still
    wants the deeper-tail bound.
    """
    df_filtered = df_buildings[
        (df_buildings['TargetYear'] == target_year) &
        (df_buildings['SLR'] == scenario)
    ].copy()
    
    if df_filtered.empty:
        return None
    
    pct_cols = ['CumEAD_P05', 'CumEAD_P50', 'CumEAD_P90', 'CumEAD_P95']
    pct_cols = [c for c in pct_cols if c in df_filtered.columns]
    
    attr_cols = [col for col in df_filtered.columns if col not in 
                 ['Action', *pct_cols, 'TargetYear', 'SLR']]
    
    df_base = df_filtered[df_filtered['Action'] == 'No mitigation'][attr_cols].copy()
    
    if df_base.empty:
        first_action = df_filtered['Action'].iloc[0]
        df_base = df_filtered[df_filtered['Action'] == first_action][attr_cols].copy()
    
    for action in df_filtered['Action'].unique():
        df_action = df_filtered[df_filtered['Action'] == action][['id'] + pct_cols].copy()
        df_action.columns = ['id'] + [
            f'{action}_{c.split("_")[1]}' for c in pct_cols
        ]
        df_base = df_base.merge(df_action, on='id', how='left')
    
    if 'DFE_Status' in df_base.columns:
        df_base['DFE_Status'] = df_base['DFE_Status'].apply(convert_floodplain_status)
    
    return df_base


def aggregate_filtered_data(df_buildings, target_year, scenario):
    """Aggregate building-level data to compute community totals."""
    df_filtered = df_buildings[
        (df_buildings['TargetYear'] == target_year) &
        (df_buildings['SLR'] == scenario)
    ].copy()
    
    if df_filtered.empty:
        return None
    
    agg_data = []
    for action in df_filtered['Action'].unique():
        df_action = df_filtered[df_filtered['Action'] == action]
        
        row = {
            'TargetYear': target_year,
            'SLR': scenario,
            'Action': action,
            'Total_CumEAD_P05': df_action['CumEAD_P05'].sum(),
            'Total_CumEAD_P50': df_action['CumEAD_P50'].sum(),
            'Total_CumEAD_P95': df_action['CumEAD_P95'].sum(),
            'Num_Buildings': df_action['id'].nunique()
        }
        
        if 'DFE_Status' in df_action.columns:
            df_under = df_action[df_action['DFE_Status'] == 'Under DFE']
            df_above = df_action[df_action['DFE_Status'] == 'Above DFE']
            row['InFP_CumEAD_P50'] = df_under['CumEAD_P50'].sum()
            row['OutFP_CumEAD_P50'] = df_above['CumEAD_P50'].sum()
        
        agg_data.append(row)
    
    return pd.DataFrame(agg_data)


# ============================================================================
# SCIENTIFIC MAP EXPORT HELPERS
# ============================================================================
# These build a publication-quality version of the live map for download.
# The live map prioritizes interactivity (clicks, hover, no chrome). The
# export adds: a title block, a credits footer, a north arrow, and a
# geographically-correct scale bar (drawn in lon/lat coordinates so it
# stays accurate at the rendered zoom). PNG/PDF/SVG output goes through
# kaleido via Plotly's `fig.to_image`.

# Tile attributions for OSM / Carto basemaps. We display the appropriate
# string in the credits footer to comply with OpenStreetMap's attribution
# requirement (https://www.openstreetmap.org/copyright).
_BASEMAP_ATTRIB = {
    "open-street-map":  "© OpenStreetMap contributors",
    "Streets":          "© OpenStreetMap contributors",  # alias for the live-map radio
    "carto-positron":   "© OpenStreetMap contributors • © CARTO",
    "carto-darkmatter": "© OpenStreetMap contributors • © CARTO",
    "stamen-terrain":   "© OpenStreetMap contributors • Stamen Design",
    "stamen-toner":     "© OpenStreetMap contributors • Stamen Design",
    "Aerial":           "Tiles © Esri, Maxar, Earthstar Geographics",
    "white-bg":         "",
}


# ESRI World Imagery raster tile source — overlaid on a white-bg base
# to render aerial photography without needing a Mapbox access token.
# (Plotly's native "satellite" / "satellite-streets" styles do require
# a token; ESRI's public tiles do not, which keeps the app working out
# of the box for anyone running it locally.) An internet connection is
# required at render time; if the tile server is unreachable the map
# will simply show a white background and the building dots.
_ESRI_WORLD_IMAGERY_LAYER = {
    "below": "traces",
    "sourcetype": "raster",
    "sourceattribution": "Tiles © Esri",
    "source": [
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ],
}


def _basemap_config(name):
    """Translate a user-facing basemap label to the (style, layers) pair
    that plotly's `mapbox` layout sub-dict expects.

    "Streets" → OpenStreetMap, no extra layers.
    "Aerial"  → white background with the ESRI World Imagery raster
                layer drawn beneath the data traces.
    Anything else is passed through as a literal plotly mapbox style.
    """
    if name == "Aerial":
        return "white-bg", [_ESRI_WORLD_IMAGERY_LAYER]
    if name == "Streets":
        return "open-street-map", []
    return name, []


def _nice_scalebar_meters(target_m):
    """Snap a desired scale-bar length (in meters) to a 1/2/5 × 10ⁿ value
    so the bar reads as a round number on the map. e.g. 1234 → 1000."""
    if target_m <= 0:
        return 100
    import math
    exp = math.floor(math.log10(target_m))
    base = 10 ** exp
    for mult in (1, 2, 5, 10):
        if mult * base >= target_m:
            return int(mult * base)
    return int(10 * base)


def build_publication_map_figure(fig_map, *, location, occupancy, target_year,
                                 scenario_label, map_view, df_map,
                                 width_px=2400, height_px=1800,
                                 mapbox_style="open-street-map",
                                 center_lat_override=None,
                                 center_lon_override=None,
                                 zoom_override=None):
    """Return a deep-copied version of ``fig_map`` styled for publication:
    embedded title, credits footer, north arrow, and a geographically-correct
    scale bar. The original interactive figure is left untouched.
    
    Parameters
    ----------
    fig_map : plotly.graph_objects.Figure
        The live map figure as built in the Map tab.
    location, occupancy, target_year, scenario_label, map_view : str
        Pieces used to build the title and subtitle.
    df_map : pandas.DataFrame
        The per-building map dataframe — used to recompute the bbox-based
        center and zoom independently of the live figure (so the export
        always frames the data the same way regardless of user pan/zoom).
    width_px, height_px : int
        Pixel dimensions of the exported image. Defaults are sized for an
        8 × 6 inch figure at 300 DPI.
    mapbox_style : str
        Plotly mapbox style. ``open-street-map`` is the default and ships
        OSM tiles for free; ``carto-positron`` and ``carto-darkmatter`` are
        also free. Other styles may require a Mapbox token.
    center_lat_override, center_lon_override, zoom_override : float | None
        If any of these is provided, override the bbox-fit framing. All
        three are optional and independent — if only some are passed,
        the rest still come from bbox-fit. This is the WYSIWYG knob: the
        export panel exposes these so users can match what they're looking
        at on the live map.
    """
    import copy
    import math
    from datetime import date as _date
    
    fig = copy.deepcopy(fig_map)
    
    # ----- Frame the data: bbox-fit center + zoom (don't trust live map) -----
    lats = df_map['latitude'].dropna().to_numpy()
    lons = df_map['longitude'].dropna().to_numpy()
    if len(lats) >= 2:
        lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
        lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)
        # Padding so points don't kiss the frame
        lat_span = max((lat_max - lat_min) * 1.25, 1e-3)
        lon_span = max((lon_max - lon_min) * 1.25, 1e-3)
        # Web-Mercator zoom that just fits the longer span at this latitude.
        # For a viewport of width_px pixels: 360 / 2^zoom = visible_lon_span
        # (at the equator). Latitude span at zoom z covers
        # 360·cos(lat)/(2^z) × (height/width). We pick the more constraining.
        zoom_lon = math.log2(360.0 / lon_span)
        # Latitude span (from a horizontal-equivalent perspective): convert
        # lat span back to "equator-equivalent" longitude span via cos(lat)
        cos_lat = max(math.cos(math.radians(center_lat)), 1e-6)
        zoom_lat = math.log2(
            (360.0 / (lat_span * (width_px / height_px))) * cos_lat
        )
        zoom = float(min(zoom_lon, zoom_lat) - 0.4)  # small headroom
        zoom = max(min(zoom, 17.0), 4.0)
    else:
        center_lat = float(np.nanmean(lats)) if len(lats) else 40.86
        center_lon = float(np.nanmean(lons)) if len(lons) else -72.49
        zoom = 12.0
    
    # User-supplied overrides win when present (WYSIWYG path).
    if center_lat_override is not None:
        center_lat = float(center_lat_override)
    if center_lon_override is not None:
        center_lon = float(center_lon_override)
    if zoom_override is not None:
        zoom = float(zoom_override)

    # ----- Build the scale bar (paper-coord shapes anchored to the map area) -----
    # We previously drew the bar as a mapbox layer (geographic LineString),
    # but those only render reliably when an active basemap is loaded. To
    # guarantee the bar is visible across all basemap choices (including the
    # white-bg fallback), we render it as paper-coord shapes and compute the
    # geographic length that the bar represents at the chosen zoom.
    import math as _math
    EARTH_CIRC = 40_075_016.686  # meters
    cos_lat = max(_math.cos(_math.radians(center_lat)), 1e-6)
    # Width of the actual map drawing area (width minus left/right margins).
    # The annotations live in paper coords (0..1) over the FULL figure, so
    # to compute pixel widths we scale by full width.
    map_inner_w = max(width_px - 80, width_px * 0.5)  # margins l=40, r=40
    # m/px in the rendered figure at this zoom
    m_per_px = EARTH_CIRC * cos_lat / (256 * (2 ** zoom))
    visible_m_inner = m_per_px * map_inner_w
    # Pick a "nice" scale-bar length that occupies ~18% of the inner width
    target_m = visible_m_inner * 0.18
    bar_m = _nice_scalebar_meters(target_m)
    if bar_m >= 1000:
        scale_label = f"{bar_m/1000:g} km"
    else:
        scale_label = f"{bar_m} m"

    # Bar's pixel length and paper-coord left edge (5% in from left margin)
    bar_px = bar_m / m_per_px
    bar_paper_w = bar_px / width_px
    sb_x0_paper = 0.05  # 5% in from the left of the figure
    sb_x1_paper = sb_x0_paper + bar_paper_w
    sb_y_paper = 0.06   # 6% up from the bottom (above the credits footer)

    # Tick whisker height — fixed pixels, converted to paper-y units
    tick_h_paper_y = 8.0 / height_px

    # Halo rectangle (light bg behind the bar so it pops on busy basemaps)
    pad_x = bar_paper_w * 0.06
    pad_y = tick_h_paper_y * 3.5
    layout_shapes = [
        # White halo
        dict(type="rect", xref="paper", yref="paper",
             x0=sb_x0_paper - pad_x, x1=sb_x1_paper + pad_x,
             y0=sb_y_paper - pad_y, y1=sb_y_paper + pad_y,
             fillcolor="rgba(255,255,255,0.85)",
             line=dict(color="rgba(0,0,0,0.15)", width=0.5),
             layer="above"),
        # Main horizontal bar
        dict(type="line", xref="paper", yref="paper",
             x0=sb_x0_paper, x1=sb_x1_paper,
             y0=sb_y_paper, y1=sb_y_paper,
             line=dict(color="black", width=4),
             layer="above"),
        # Left tick whisker
        dict(type="line", xref="paper", yref="paper",
             x0=sb_x0_paper, x1=sb_x0_paper,
             y0=sb_y_paper - tick_h_paper_y, y1=sb_y_paper + tick_h_paper_y,
             line=dict(color="black", width=4),
             layer="above"),
        # Right tick whisker
        dict(type="line", xref="paper", yref="paper",
             x0=sb_x1_paper, x1=sb_x1_paper,
             y0=sb_y_paper - tick_h_paper_y, y1=sb_y_paper + tick_h_paper_y,
             line=dict(color="black", width=4),
             layer="above"),
    ]
    # Mapbox-layer scale bar removed — keeping mapbox.layers empty so the
    # bar renders identically regardless of basemap choice.
    mapbox_layers = []

    # Translate the "Aerial" pseudo-style into its real implementation
    # (white-bg background + ESRI World Imagery raster tile layer).
    # Doing this here means the rest of the export pipeline can keep
    # treating mapbox_style as a flat string. The export-time tile fetch
    # uses the same ESRI endpoint as the live map, so if the tiles fail
    # to load the export falls back to a plain white background and the
    # data markers stay readable.
    # NB: we remember the user-facing name in `attrib_key` BEFORE
    # overwriting `mapbox_style`, so the attribution lookup below still
    # resolves to the ESRI credit (white-bg by itself has no attribution
    # and would otherwise drop the basemap line from the export footer).
    attrib_key = mapbox_style
    if mapbox_style == "Aerial":
        mapbox_style = "white-bg"
        mapbox_layers = [_ESRI_WORLD_IMAGERY_LAYER]

    # ----- Title and footer in paper coordinates -----
    title_main = f"Building-level Flood Risk — {location}"
    if occupancy and occupancy != "All":
        title_main += f" ({occupancy})"
    title_sub = (f"Year {target_year}  •  {scenario_label}  •  "
                 f"{map_view}")
    today = _date.today().strftime("%B %Y")

    attrib = _BASEMAP_ATTRIB.get(attrib_key, "© OpenStreetMap contributors")
    # Skip the "Basemap:" segment when there's no tile attribution to show
    # (e.g. ``white-bg``), which avoids a dangling "Basemap:  •" in the footer.
    footer_parts = [
        "Data: National Structure Inventory  •  NACCS (USACE, 2015)  •  "
        "NYS Climate Impacts Assessment (2024)",
    ]
    if attrib:
        footer_parts.append(f"Basemap: {attrib}")
    footer_parts.append(f"Generated {today}")
    footer_lines = [
        ("ADAPT — Assessment of Damage and Adaptation Planning Tool   |   "
         "Center for Climate Systems Research, The Climate School, Columbia University"),
        "  •  ".join(footer_parts),
    ]

    annotations = [
        # Main title
        dict(text=f"<b>{title_main}</b>",
             xref="paper", yref="paper",
             x=0.5, y=1.03, xanchor="center", yanchor="bottom",
             showarrow=False,
             font=dict(family="Arial", size=24, color="#0f172a")),
        # Subtitle
        dict(text=title_sub,
             xref="paper", yref="paper",
             x=0.5, y=1.005, xanchor="center", yanchor="bottom",
             showarrow=False,
             font=dict(family="Arial", size=16, color="#475569")),
        # Footer line 1
        dict(text=footer_lines[0],
             xref="paper", yref="paper",
             x=0.5, y=-0.025, xanchor="center", yanchor="top",
             showarrow=False,
             font=dict(family="Arial", size=12, color="#334155")),
        # Footer line 2
        dict(text=footer_lines[1],
             xref="paper", yref="paper",
             x=0.5, y=-0.062, xanchor="center", yanchor="top",
             showarrow=False,
             font=dict(family="Arial", size=11, color="#64748b")),
        # Scale-bar label — centered above the bar at its midpoint
        dict(text=f"<b>{scale_label}</b>",
             xref="paper", yref="paper",
             x=(sb_x0_paper + sb_x1_paper) / 2,
             y=sb_y_paper + tick_h_paper_y * 4,
             xanchor="center", yanchor="bottom",
             showarrow=False,
             font=dict(family="Arial", size=14, color="#0f172a")),
        # North arrow — paper-coord glyph in the upper-right corner
        dict(text="<b>N</b><br>▲",
             xref="paper", yref="paper",
             x=0.96, y=0.96, xanchor="center", yanchor="top",
             showarrow=False,
             align="center",
             bgcolor="rgba(255,255,255,0.9)",
             bordercolor="rgba(0,0,0,0.3)", borderwidth=1, borderpad=6,
             font=dict(family="Arial Black", size=18, color="#0f172a")),
    ]

    # Existing legend — move it to a less-cluttered corner if possible,
    # and tighten its background so it reads well in print.
    legend_cfg = dict(
        yanchor="bottom", y=0.05, xanchor="right", x=0.98,
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(0,0,0,0.25)", borderwidth=1,
        font=dict(family="Arial", size=12, color="#0f172a"),
    )

    # Apply layout updates. Margins make room for title/footer.
    fig.update_layout(
        mapbox=dict(
            style=mapbox_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
            layers=mapbox_layers,
        ),
        margin=dict(l=40, r=40, t=80, b=140),
        width=width_px, height=height_px,
        paper_bgcolor="white",
        showlegend=True,
        legend=legend_cfg,
        annotations=annotations,
        shapes=layout_shapes,
        # Static images get a watermark-free, hover-free rendering
        hovermode=False,
    )
    return fig


def export_map_image(pub_fig, fmt="png", scale=2, allow_basemap_fallback=True):
    """Render a publication map figure to bytes via kaleido.

    Returns ``(bytes, mime_type, file_extension, info)``, where ``info`` is
    ``None`` on success, an error string on hard failure, or a short note
    when a basemap fallback was used (e.g. ``"basemap_fallback"``).

    Tile-based basemaps (open-street-map, carto-*) require live HTTP fetches
    during render; some tile providers refuse automated requests and return
    HTTP 403, which kaleido surfaces as ``"Map error."``. When
    ``allow_basemap_fallback`` is True we transparently retry with
    ``style="white-bg"`` so the user still gets a usable export, with the
    points/scale-bar/credits intact on a clean white canvas.
    """
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "svg": "image/svg+xml",
            "pdf": "application/pdf"}.get(fmt.lower(), "application/octet-stream")
    try:
        img_bytes = pub_fig.to_image(format=fmt, scale=scale)
        return img_bytes, mime, fmt.lower(), None
    except Exception as e:
        msg = str(e)
        msg_lc = msg.lower()
        # The two common failure modes have very different remedies, so we
        # detect them separately. Tile-render failures surface from kaleido
        # under several strings depending on version:
        #   * "Map error."         (modern Scattermap / map.* layout)
        #   * "Mapbox error."      (legacy Scattermapbox / mapbox.* layout)
        #   * "Error 525: …"       (kaleido v1's wrapper around either)
        is_map_error = (
            "map error" in msg_lc or
            "mapbox error" in msg_lc or
            "525" in msg
        )
        is_kaleido_missing = (
            ("kaleido" in msg_lc and "map" not in msg_lc) or
            "chrom" in msg_lc or
            isinstance(e, ImportError)
        )
        if is_map_error and allow_basemap_fallback:
            # Retry with white-bg so the user gets a usable export
            try:
                import copy
                fallback = copy.deepcopy(pub_fig)
                # Update both possible style keys (mapbox.* legacy and map.* new)
                if hasattr(fallback.layout, 'mapbox') and fallback.layout.mapbox:
                    fallback.update_layout(mapbox=dict(style="white-bg"))
                if hasattr(fallback.layout, 'map') and fallback.layout.map:
                    fallback.update_layout(map=dict(style="white-bg"))
                img_bytes = fallback.to_image(format=fmt, scale=scale)
                return img_bytes, mime, fmt.lower(), "basemap_fallback"
            except Exception as e2:
                return None, None, None, (
                    f"Export failed even after basemap fallback: {e2}"
                )
        if is_kaleido_missing:
            return None, None, None, (
                "Static-image export requires `kaleido` (>=1.0) and a Chrome "
                "browser. Install with `pip install -U kaleido` and run "
                "`plotly_get_chrome` once to download Chrome. Detail: " + msg
            )
        return None, None, None, msg


# ============================================================================
# SHARED PLOT CONSTRUCTION HELPERS
# ============================================================================

# Workshop palette (green = P50 SLR, orange = P90 SLR), shared by all tabs
CLR_P50_LINE = 'rgb(74, 124, 89)'      # workshop green
CLR_P90_LINE = 'rgb(212, 121, 25)'     # workshop orange
CLR_P50_FILL = 'rgba(74, 124, 89, 0.55)'
CLR_P90_FILL = 'rgba(212, 121, 25, 0.55)'

SCENARIO_SPECS = [
    ('50th-percentile', 'Median SLR (P50)',   CLR_P50_LINE, CLR_P50_FILL),
    ('90th-percentile', 'High-End SLR (P90)', CLR_P90_LINE, CLR_P90_FILL),
]

# CDF-linear interpolation factors for Q1 and Q3 from P05/P50/P95
# Q1 sits at CDF=0.25, between P05 and P50: (0.25-0.05)/(0.50-0.05)
# Q3 sits at CDF=0.75, between P50 and P95: (0.75-0.50)/(0.95-0.50)
Q1_FRAC = (0.25 - 0.05) / (0.50 - 0.05)
Q3_FRAC = (0.75 - 0.50) / (0.95 - 0.50)


def build_box_whisker_panel(group_labels, scenario_data, panel_title="",
                            y_label="Cumulative Damage", bg_color=None,
                            height=520, label_zero_thresh=None,
                            lower_label="P05", upper_label="P95",
                            lower_pct=0.05, upper_pct=0.95):
    """Construct a Plotly box-and-whisker panel with grouped pairs of boxes.

    Parameters
    ----------
    group_labels : list[str]
        X-axis category labels (one per group, e.g. one per strategy).
    scenario_data : dict[slr_key -> list[tuple or None]]
        Per-group statistics for each SLR scenario. Use ``None`` for
        missing groups. Each tuple is one of:
          * ``(p05, p50, p95)`` — 3-tuple. Q1/Q3 are estimated by
            linear-CDF interpolation between the supplied bounds.
          * ``(p05, p25, p50, p75, p95)`` — 5-tuple. Q1 and Q3 are taken
            **directly** from the supplied stored P25/P75 values, which is
            the preferred path when the workbook already contains them
            (the ALL format does). This eliminates the small bias that
            CDF-linear interpolation introduces for skewed damage tails.
        Both shapes can be mixed across groups; each group is interpreted
        on its own.
        Keys must match the first element of SCENARIO_SPECS tuples.
    panel_title : str
    y_label : str
    bg_color : str or None
        Optional plot background (e.g. light gray for the reduction panel).
    height : int
    label_zero_thresh : float or None
        Damages whose absolute value is below this threshold render as "$0"
        in value labels. Defaults to ZERO_THRESH_DISPLAY.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if label_zero_thresh is None:
        label_zero_thresh = ZERO_THRESH_DISPLAY

    # Linear-CDF interpolation factors used as the FALLBACK for callers that
    # only have (p05, p50, p95) available (e.g. the per-building benefit
    # tuples in the Details tab, where benefits are derived as differences
    # of percentiles rather than read from a column). These factors are
    # exact only for symmetric distributions; for skewed flood-damage
    # distributions they're a small approximation. Whenever the caller
    # supplies a 5-tuple with real stored P25/P75, those are used directly.
    _q1_frac = (0.25 - lower_pct) / max(1e-9, (0.50 - lower_pct))
    _q3_frac = (0.75 - 0.50)       / max(1e-9, (upper_pct - 0.50))
    _q1_frac = min(max(_q1_frac, 0.0), 1.0)
    _q3_frac = min(max(_q3_frac, 0.0), 1.0)

    fig = go.Figure()

    # Numeric x-axis: each group sits at integer x=0,1,2,... and each SLR
    # box within a group is offset by ±BOX_HALF_OFFSET from the group center.
    # Annotations use those same numeric x positions so labels land *exactly*
    # above their box, independent of rendered plot width.
    BOX_HALF_OFFSET = 0.20          # horizontal half-spacing of paired boxes
    BOX_WIDTH = 0.34                # visible width of each individual box

    annot_records = []   # (x_num, p50, p95, line_clr)
    y_min = 0.0
    y_max_whisker = 0.0

    for slr_key, slr_label, line_clr, fill_clr in SCENARIO_SPECS:
        if slr_key not in scenario_data:
            continue

        offset_sign = -1 if slr_key == '50th-percentile' else +1
        x_num, x_hover_labels, p05_arr, p50_arr, p95_arr, q1_arr, q3_arr = \
            [], [], [], [], [], [], []

        for gi, stats in enumerate(scenario_data[slr_key]):
            if stats is None:
                continue
            # Two accepted shapes — see the docstring above.
            if len(stats) == 5:
                p05, p25, p50, p75, p95 = stats
                q1, q3 = p25, p75
                if any(pd.isna(v) for v in (p05, p25, p50, p75, p95)):
                    continue
            elif len(stats) == 3:
                p05, p50, p95 = stats
                if any(pd.isna(v) for v in (p05, p50, p95)):
                    continue
                # Fallback: estimate Q1/Q3 by linear-CDF interpolation.
                q1 = p05 + _q1_frac * (p50 - p05)
                q3 = p50 + _q3_frac * (p95 - p50)
            else:
                # Unrecognized shape — skip rather than crash.
                continue

            x_center = gi + offset_sign * BOX_HALF_OFFSET
            x_num.append(x_center)
            x_hover_labels.append(group_labels[gi])
            p05_arr.append(p05); p50_arr.append(p50); p95_arr.append(p95)
            q1_arr.append(q1)
            q3_arr.append(q3)

            annot_records.append((x_center, p05, p50, p95, line_clr))

            if pd.notna(p95):
                y_max_whisker = max(y_max_whisker, float(p95))
            if pd.notna(p05):
                y_min = min(y_min, float(p05))

        if not x_num:
            continue

        fig.add_trace(go.Box(
            name=slr_label,
            x=x_num,                        # NUMERIC x — exact positions
            q1=q1_arr,
            median=p50_arr,
            q3=q3_arr,
            lowerfence=p05_arr,
            upperfence=p95_arr,
            fillcolor=fill_clr,
            line=dict(color=line_clr, width=1.6),
            marker_color=line_clr,
            whiskerwidth=0.55,
            width=BOX_WIDTH,
            customdata=x_hover_labels,
            hovertemplate=(
                f'<b>{slr_label}</b><br>'
                '%{customdata}<br>'
                f'{lower_label}: ' '%{lowerfence:$,.0f}<br>'
                'P50: %{median:$,.0f}<br>'
                f'{upper_label}: ' '%{upperfence:$,.0f}'
                '<extra></extra>'
            ),
        ))
        
        # Overlay an explicit "$0" marker for any box that collapses to zero
        # across all five summary stats. Plotly draws nothing visible when
        # q1=q3=median=lower=upper=0, which makes a true-zero strategy look
        # missing — but the message we want is "this strategy reduces damage
        # to zero," not "no data." The overlay is a small filled diamond
        # placed right at zero, in the same line color as the box, with a
        # hover that confirms the zero reading.
        zero_x, zero_labels = [], []
        for xc, p05_v, q1_v, p50_v, q3_v, p95_v, lab in zip(
            x_num, p05_arr, q1_arr, p50_arr, q3_arr, p95_arr, x_hover_labels
        ):
            if all(abs(v) < label_zero_thresh for v in (p05_v, q1_v, p50_v, q3_v, p95_v)):
                zero_x.append(xc)
                zero_labels.append(lab)
        if zero_x:
            fig.add_trace(go.Scatter(
                x=zero_x, y=[0.0] * len(zero_x),
                mode='markers',
                marker=dict(symbol='diamond', size=10,
                            color=line_clr,
                            line=dict(color='white', width=1.2)),
                customdata=zero_labels,
                hovertemplate=(
                    f'<b>{slr_label}</b><br>'
                    '%{customdata}<br>'
                    'All percentiles ≈ $0'
                    '<extra></extra>'
                ),
                showlegend=False,
            ))

    # Y-axis range with headroom for whisker labels
    y_span = max(abs(y_max_whisker), abs(y_min))
    if y_span <= 0:
        y_span = 1.0
    y_head = y_max_whisker * 1.22 if y_max_whisker > 0 else y_span * 0.2
    y_floor = y_min * 1.15 if y_min < 0 else 0
    label_gap = y_span * 0.025       # small vertical gap above each line/whisker

    # Annotations: every label sits directly above its reference line —
    #   * Median label sits just above the median line (yanchor='bottom')
    #   * P95 label sits just above the upper whisker
    # The lower whisker (P05) is intentionally NOT labeled.
    # X position is the exact numeric center of each box, so labels never
    # drift away from their own box regardless of plot width.
    for x_center, p05_v, p50_v, p95_v, line_clr in annot_records:
        med_text = fmt_money_rounded(p50_v) if abs(p50_v) >= label_zero_thresh else "$0"
        # Median — bold colored text on a semi-transparent white pill,
        # placed just above the median line so it doesn't overlap it.
        fig.add_annotation(
            x=x_center, y=p50_v + label_gap,
            text=f"<b>{med_text}</b>",
            showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color=line_clr),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=2, bordercolor='rgba(0,0,0,0)',
        )
        if abs(p95_v) >= label_zero_thresh:
            # P95 — bold colored text just above the upper whisker
            fig.add_annotation(
                x=x_center, y=p95_v + label_gap,
                text=f"<b>{fmt_money_rounded(p95_v)}</b>",
                showarrow=False,
                xanchor='center', yanchor='bottom',
                font=dict(size=12, color=line_clr),
            )

    tickvals, ticktext = smart_money_ticks(y_head, target_n=6)

    fig.update_layout(
        title=dict(text=panel_title, x=0.02, xanchor='left', font=dict(size=15)),
        height=height,
        plot_bgcolor=(bg_color if bg_color else 'white'),
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, bgcolor='rgba(255,255,255,0.85)'),
        margin=dict(l=60, r=20, t=70, b=80),
        boxmode='overlay',              # numeric x already places boxes correctly
        xaxis=dict(
            title="Adaptation Strategy",
            tickmode='array',
            tickvals=list(range(len(group_labels))),
            ticktext=group_labels,
            tickangle=-20,
            range=[-0.5, len(group_labels) - 0.5],
            showgrid=False, showline=True, linecolor='#cbd5e1',
            zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True, gridcolor='#e5e7eb',
            showline=True, linecolor='#cbd5e1', zeroline=False,
            tickmode='array', tickvals=tickvals, ticktext=ticktext,
            range=[y_floor, y_head],
        ),
    )
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # ========================================================================
    # PASSWORD PROTECTION
    # ========================================================================
    
    def check_password():
        """Returns True if the user entered the correct password."""
        
        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["password"] == "NY2026VA":
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store password
            else:
                st.session_state["password_correct"] = False
        
        # First run or password not correct
        if "password_correct" not in st.session_state:
            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 2.5rem; font-weight: bold; color: #0ea5e9; text-align: center;">🔒 ADAPT</p>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; color: #64748b; margin-bottom: 2rem;">Assessment of Damage and Adaptation Planning Tool</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                st.text_input(
                    "Enter password to access the tool:",
                    type="password",
                    on_change=password_entered,
                    key="password"
                )
            return False
        
        # Password correct
        elif st.session_state["password_correct"]:
            return True
        
        # Password incorrect
        else:
            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 2.5rem; font-weight: bold; color: #0ea5e9; text-align: center;">🔒 ADAPT</p>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; color: #64748b; margin-bottom: 2rem;">Assessment of Damage and Adaptation Planning Tool</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                st.text_input(
                    "Enter password to access the tool:",
                    type="password",
                    on_change=password_entered,
                    key="password"
                )
                st.error("😕 Incorrect password. Please try again.")
            return False
    
    if not check_password():
        st.stop()
    
    # ========================================================================
    # LOAD DATA FROM FOLDER
    # ========================================================================
    data_store, available_locations = load_data_from_folder(".")
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        # --- ADAPT branding at the top of the sidebar ---
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        else:
            st.markdown(
                '<div style="text-align:center; padding: 0.25rem 0 0.5rem 0;">'
                '<span style="font-size: 1.9rem; font-weight: 800; color:#0ea5e9; letter-spacing: 0.5px;">ADAPT</span>'
                '<div style="font-size: 0.75rem; color:#475569; font-weight:500; margin-top:-2px;">'
                'Assessment of Damage and Adaptation Planning Tool</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='margin: 0.5rem 0 0.75rem 0; border: none; border-top: 1px solid #e2e8f0;'/>",
                    unsafe_allow_html=True)
        
        st.success(f"✅ Data loaded: {len(available_locations)} location(s)")
        
        st.header("🎛️ Data Selection")
        
        if len(available_locations) > 0:
            # Default to Pamunkey when it's in the bundle; otherwise the
            # first available location (preserves prior behavior for any
            # data folder that doesn't ship Pamunkey).
            _default_loc_idx = 0
            for _i, _loc in enumerate(available_locations):
                if _loc == "Pamunkey":
                    _default_loc_idx = _i
                    break
            selected_location = st.selectbox(
                "📍 Location",
                options=available_locations,
                index=_default_loc_idx
            )
        else:
            selected_location = None
        
        selected_occupancy = st.selectbox(
            "🏠 Occupancy Type",
            options=["All", "Residential", "Non-Residential"],
            index=0,
            format_func=lambda x: f"🏘️🏢 All Buildings" if x == "All" else f"🏘️ Residential" if x == "Residential" else f"🏢 Non-Residential"
        )
        
        df_agg_raw = None
        df_buildings_raw = None
        loc_entry = None

        if selected_location and selected_location in data_store:
            loc_entry = data_store[selected_location]
            df_agg_raw = loc_entry.get('agg')
            df_buildings_raw = loc_entry.get('buildings')

        df_buildings = filter_by_occupancy(df_buildings_raw, selected_occupancy)
        
        # Drop the 2025 ("baseline") year from every downstream dataframe.
        # The bundle ships it as a 'Potential' label so users can see today's
        # damage in tables, but for this tool the relevant story is the
        # SLR-driven escalation across 2040 / 2055 / 2100 — surfacing 2025
        # adds a column/row that doesn't belong in any of the comparative
        # views and creates noise (e.g. damage trajectories anchored at the
        # baseline). Drop it ONCE here so every tab inherits the cleaner
        # year set automatically.
        BASELINE_YEAR_TO_DROP = 2025
        if df_buildings is not None and 'TargetYear' in df_buildings.columns:
            df_buildings = df_buildings[df_buildings['TargetYear'] != BASELINE_YEAR_TO_DROP].copy()
        if df_agg_raw is not None and 'TargetYear' in df_agg_raw.columns:
            df_agg_raw = df_agg_raw[df_agg_raw['TargetYear'] != BASELINE_YEAR_TO_DROP].copy()

        # When a new-format workbook is loaded we have a pre-computed,
        # MC-correct aggregate per occupancy bucket — pick the one that
        # matches the sidebar selection rather than resumming per-building
        # percentiles (which would be statistically wrong).
        preloaded_agg = None
        if loc_entry is not None and loc_entry.get('agg_by_occ'):
            preloaded_agg = loc_entry['agg_by_occ'].get(selected_occupancy)
            # Same baseline-year drop as above so the agg-driven Summary /
            # Trends / DFE-comparison panels don't include 2025 either.
            if preloaded_agg is not None and 'TargetYear' in preloaded_agg.columns:
                preloaded_agg = preloaded_agg[
                    preloaded_agg['TargetYear'] != BASELINE_YEAR_TO_DROP
                ].copy()
        
        st.divider()
        st.header("🎛️ Scenario Filters")
        
        # Fallback horizon set used only when no data is loaded yet; the
        # real options come from the bundle's TargetYear values just below.
        available_years = [2040, 2055, 2060, 2100]
        if df_buildings is not None and 'TargetYear' in df_buildings.columns:
            available_years = sorted(df_buildings['TargetYear'].unique())
        elif df_agg_raw is not None and 'TargetYear' in df_agg_raw.columns:
            available_years = sorted(df_agg_raw['TargetYear'].unique())
        
        # Per-bundle TargetYear label map ('Potential' for the 2025 baseline,
        # '2040'/'2055'/'2100' otherwise). Falls back to the raw int when a
        # location doesn't ship the labels (e.g. legacy data).
        target_year_labels = {}
        if loc_entry is not None and isinstance(loc_entry.get('target_year_labels'), dict):
            target_year_labels = loc_entry['target_year_labels']
        
        def _format_target_year(y):
            label = target_year_labels.get(int(y), str(int(y)))
            # Display rule: the bundle's 'Potential' label is an internal
            # convention for the 2025 baseline; users find a plain year
            # easier to reason about, so we show only the year here. Other
            # custom labels (anything that isn't 'Potential' and isn't just
            # the bare year) still get appended in parentheses.
            if label != str(int(y)) and label != 'Potential':
                return f"{label} ({int(y)})"
            return str(int(y))
        
        # Default to the first non-baseline horizon (typically 2040) — opening
        # the app on the 2025 baseline hides the SLR-driven escalation that's
        # the whole point of the tool.
        default_idx = 0
        if len(available_years) > 1:
            for i, y in enumerate(available_years):
                if target_year_labels.get(int(y)) not in (None, 'Potential'):
                    default_idx = i
                    break
        
        target_year = st.selectbox(
            "📅 Target Year",
            options=available_years,
            index=default_idx,
            format_func=_format_target_year,
        )
        
        available_scenarios = ['50th-percentile', '90th-percentile']
        if df_buildings is not None and 'SLR' in df_buildings.columns:
            available_scenarios = sorted(df_buildings['SLR'].unique())
        elif df_agg_raw is not None and 'SLR' in df_agg_raw.columns:
            available_scenarios = sorted(df_agg_raw['SLR'].unique())
        
        scenario = st.selectbox(
            "🌊 SLR Scenario",
            options=available_scenarios,
            format_func=lambda x: 'Median SLR (50th-percentile)' if x == '50th-percentile' else 'High-End SLR (90th-percentile)' if x == '90th-percentile' else x
        )
        
        st.divider()
        st.header("🗺️ Map Settings")
        
        if df_buildings is not None and 'DFE_Status' in df_buildings.columns:
            fp_options = df_buildings['DFE_Status'].dropna().unique().tolist()
            dfe_filter = st.multiselect(
                "DFE Status (BFE+2)",
                options=fp_options,
                default=fp_options
            )
        else:
            dfe_filter = None
        
        show_zero_damage = st.checkbox("Show buildings with $0 damage", value=True)
        
        if df_buildings is not None:
            st.divider()
            n_loaded = df_buildings['id'].nunique()
            st.caption(f"**Buildings loaded:** {n_loaded:,}")
            # When the bundle ships a skipped-buildings log, surface a small
            # transparency note so users know the inventory isn't claiming
            # 100% coverage. `loc_entry` may be None if no location is
            # selected yet.
            if loc_entry is not None:
                skipped_df = loc_entry.get('skipped')
                if isinstance(skipped_df, pd.DataFrame) and len(skipped_df) > 0:
                    n_skipped = len(skipped_df)
                    plural = 's' if n_skipped != 1 else ''
                    st.caption(
                        f"⚠️ {n_skipped} building{plural} excluded from the analysis "
                        f"(see *Data Notes* below the map)."
                    )
                bfe_ft = loc_entry.get('bfe_ft')
                if bfe_ft is not None:
                    st.caption(f"BFE: **{bfe_ft:g} ft NAVD88** (DFE = BFE+2)")
        
        # ====================================================================
        # 🖥️ Display
        # --------------------------------------------------------------------
        # Layout density toggle. Browsers don't expose a programmatic zoom
        # API for security/accessibility reasons, so an in-app "set my
        # zoom to 90%" isn't possible. The next-best thing is a CSS
        # `transform: scale(0.9)` applied to the main content area —
        # visually equivalent for most users, and the toggle leaves it
        # off by default so anyone with a small screen / large fonts
        # isn't surprised by an unrequested rescale.
        # ====================================================================
        st.divider()
        st.subheader("🖥️ Display")
        layout_density = st.radio(
            "Layout density",
            options=["Default", "Compact (90%)"],
            index=0,
            horizontal=True,
            key="layout_density",
            help=(
                "**Default** — content renders at the browser's actual size. "
                "**Compact** — visually shrinks the main page to ~90% via "
                "CSS scale, fitting more on screen. Equivalent to pressing "
                "Ctrl+– once in the browser, but only affects this app and "
                "leaves your global browser zoom alone. Note: hover hit-"
                "targets on the map may be slightly misaligned in Compact "
                "mode (Plotly computes them in unscaled pixel space)."
            ),
        )
    
    # ========================================================================
    # GLOBAL CSS — sidebar narrowing (always on) + optional compact density
    # ========================================================================
    # The sidebar narrowing is purely a layout fix: Streamlit's default
    # sidebar width (~336 px on desktop, ~21rem) eats into the main column
    # more than this dashboard needs. Pinning min/max keeps the controls
    # readable while reclaiming horizontal real estate for the maps and
    # box plots.
    #
    # The compact-mode rule scales the main content area to 90% and then
    # widens it by 1/0.9 to recover the visual width the scale would have
    # stolen. Anchoring to `top center` keeps content where the user
    # expects it. Applied only when the sidebar toggle is set to
    # "Compact (90%)".
    _css_blocks = [
        # Always-on: narrower sidebar
        """
        section[data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 280px !important;
        }
        """
    ]
    if layout_density == "Compact (90%)":
        _css_blocks.append("""
        .main .block-container {
            transform: scale(0.9);
            transform-origin: top center;
            width: 111.11%;
        }
        """)
    st.markdown(
        "<style>\n" + "\n".join(_css_blocks) + "\n</style>",
        unsafe_allow_html=True,
    )
    
    # ========================================================================
    # PAGE TITLE — centered, bold, above the tabs
    # ========================================================================
    location_name = selected_location if selected_location else ""
    occupancy_label = selected_occupancy if selected_occupancy != "All" else "All Buildings"
    
    if selected_location:
        page_title = f"Building-level flood damage assessment for {selected_location}"
        if selected_occupancy != "All":
            page_title += f" — {selected_occupancy}"
    else:
        page_title = "Building-level flood damage assessment under climate change scenarios"
    
    st.markdown(
        "<h1 style='text-align:center; color:#0f172a; font-weight:800; "
        "font-size:1.9rem; line-height:1.2; margin: 0.25rem 0 1.25rem 0;'>"
        f"{page_title}"
        "</h1>",
        unsafe_allow_html=True,
    )
    
    # ========================================================================
    # CHECK IF DATA IS LOADED
    # ========================================================================
    
    if len(available_locations) == 0:
        st.error("⚠️ No data files found. Please ensure `.xlsx` result files (e.g., `Shinnecock_Results_ALL.xlsx`) are in the same directory as `app.py`.")
        st.stop()
    
    if df_buildings is None or len(df_buildings) == 0:
        st.warning(f"No {selected_occupancy.lower()} buildings found in the data for {selected_location}.")
        st.stop()
    
    # ========================================================================
    # COMPUTE AGGREGATED DATA
    # ========================================================================
    
    df_agg = None
    if preloaded_agg is not None and not preloaded_agg.empty:
        # New-format: use the MC-correct aggregate (community percentiles
        # derived from the sum of MC realizations across buildings, not from
        # summing per-building percentiles).
        df_agg = preloaded_agg.copy()
    elif df_buildings is not None:
        agg_frames = []
        for yr in df_buildings['TargetYear'].unique():
            for scn in df_buildings['SLR'].unique():
                agg_df = aggregate_filtered_data(df_buildings, yr, scn)
                if agg_df is not None:
                    agg_frames.append(agg_df)
        if agg_frames:
            df_agg = pd.concat(agg_frames, ignore_index=True)
    
    # ========================================================================
    # MAIN CONTENT - TABS
    # ========================================================================
    
    tab1, tab_dist, tab2, tab3, tab4, tab_flood, tab_roads = st.tabs([
        "📊 Summary",
        "📦 Distributions",
        "🗺️ Map",
        "🏠 Details",
        "📈 Trends",
        "🌊 Flood Maps",
        "🛣️ Flooded Roads",
    ])

    # ========================================================================
    # TAB: FLOOD MAPS — bathtub inundation for user-specified water levels
    # ========================================================================
    with tab_flood:
        st.markdown(
            '<p class="tab-description">Bathtub flood-inundation maps for water levels you specify. '
            'Enter present-day flood levels (ft NAVD88); the app adds projected sea-level rise for each '
            'planning horizon and maps the resulting inundation depth. Terrain is the USGS 3DEP 1/3 arc-second '
            '(~10&nbsp;m) DEM, read on demand for this area only.</p>',
            unsafe_allow_html=True,
        )

        _fl_ok = (selected_location in data_store and df_buildings is not None)
        _has_xy = (
            df_buildings is not None
            and {'longitude', 'latitude'}.issubset(df_buildings.columns)
            and df_buildings['latitude'].notna().any()
            and df_buildings['longitude'].notna().any()
        )

        try:
            import rasterio as _rio  # noqa: F401
            _rio_ok = True
        except Exception:
            _rio_ok = False

        if not _fl_ok:
            st.info("Select a location with building data to generate flood maps.")
        elif not _has_xy:
            st.warning("This location has no per-building coordinates, so the map extent can't be determined.")
        elif not _rio_ok:
            st.error(
                "Flood maps need the **rasterio** package to read the terrain. "
                "Add `rasterio` to the app's requirements.txt and redeploy."
            )
        else:
            _wl = loc_entry.get('water_levels', {}) if loc_entry else {}
            # SLR scenario keys present in the bundle (exclude raw MC sheets)
            _scn_keys = [k for k in _wl.keys() if not k.endswith('_mc')]
            _scn_label = {
                '50th-percentile': 'Intermediate-High SLR (50th pct)',
                '90th-percentile': 'High SLR (90th pct)',
            }
            _scn_pretty = lambda k: _scn_label.get(k, k)
            # Present-day prefill comes from a reference scenario (prefer 50th).
            _ref = '50th-percentile' if '50th-percentile' in _wl else (_scn_keys[0] if _scn_keys else None)
            _ref_df = _wl.get(_ref)
            _base_year = int(_ref_df['Year'].min()) if (_ref_df is not None and not _ref_df.empty) else 2025

            def _lvl(df, year, col):
                if df is None or df.empty or col not in df.columns:
                    return None
                i = (df['Year'] - year).abs().idxmin()
                return float(df.loc[i, col])

            def _slr(scn_key, year):
                df = _wl.get(scn_key)
                a, b = _lvl(df, year, 'P50'), _lvl(df, _base_year, 'P50')
                return (a - b) if (a is not None and b is not None) else 0.0

            _pf = {
                'annual': _lvl(_ref_df, _base_year, 'P50'),
                'ten':    _lvl(_ref_df, _base_year, 'P90'),
                'one':    _lvl(_ref_df, _base_year, 'P99'),
            }
            if not _scn_keys:
                st.warning(
                    "No Monte-Carlo water-level data found for this location, so the three flood types "
                    "aren't prefilled and sea-level rise can't be derived (future levels will equal what "
                    "you enter). You can still enter levels manually."
                )

            st.markdown(
                f"**Present-day base levels** (ft NAVD88) — the same for both SLR scenarios; future "
                f"levels add each scenario's sea-level rise. Tick the levels to map and edit any value. "
                f"Annual / 10% / 1% are prefilled from the {_base_year} water-level percentiles where available."
            )

            _is_pamunkey = (selected_location == "Pamunkey")
            if _is_pamunkey:
                # Pamunkey: HTF / 10% / 1% are selected by default with project values.
                _defs = [
                    ("High-tide flood",         2.37,          True,
                     "High-tide flooding (HTF) level, ft NAVD88."),
                    ("Monthly flood",           None,          False,
                     "Optional. A level reached roughly monthly, ft NAVD88."),
                    ("Annual flood",            _pf['annual'], False,
                     "≈ the level reached most years (50th-percentile annual maximum)."),
                    ("10% annual-chance flood", 5.78,          True,
                     "1-in-10-year level (90th-percentile annual maximum)."),
                    ("1% annual-chance flood",  7.38,          True,
                     "1-in-100-year level (99th-percentile annual maximum)."),
                ]
            else:
                _defs = [
                    ("High-tide flood",         None,          False,
                     "Optional. e.g. the local NOAA minor/nuisance-flood threshold, ft NAVD88."),
                    ("Monthly flood",           None,          False,
                     "Optional. A level reached roughly monthly, ft NAVD88."),
                    ("Annual flood",            _pf['annual'], _pf['annual'] is not None,
                     "≈ the level reached most years (50th-percentile annual maximum)."),
                    ("10% annual-chance flood", _pf['ten'],    _pf['ten'] is not None,
                     "1-in-10-year level (90th-percentile annual maximum)."),
                    ("1% annual-chance flood",  _pf['one'],    _pf['one'] is not None,
                     "1-in-100-year level (99th-percentile annual maximum)."),
                ]

            _rows = []
            for _i, (_lbl, _val, _on, _help) in enumerate(_defs):
                _c0, _c1 = st.columns([0.34, 0.66])
                _inc = _c0.checkbox(_lbl, value=_on, key=f"fld_inc_{_lbl}", help=_help)
                _default = float(_val) if _val is not None else 0.0
                _lv = _c1.number_input(
                    f"{_lbl} level (ft NAVD88)", value=_default, step=0.5, format="%.2f",
                    key=f"fld_val_{_lbl}", label_visibility="collapsed",
                )
                if _inc:
                    _rows.append((_lbl, float(_lv)))

            _c_scn, _c_yr = st.columns(2)
            _scn_sel = _c_scn.multiselect(
                "SLR scenarios", options=_scn_keys, default=_scn_keys, format_func=_scn_pretty,
            )
            _years_sel = _c_yr.multiselect(
                "Planning horizons", options=available_years, default=available_years,
                format_func=lambda y: str(int(y)),
            )
            _cz, _cd, _cb = st.columns([0.36, 0.32, 0.32])
            _zoom_factor = _cz.slider(
                "Zoom (tighten)", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
                help="Higher = tighter crop around the community (more zoomed in).",
            )
            _res_label = _cd.selectbox(
                "Map detail", ["Standard", "Fine", "Finer"], index=1,
                help="Higher detail = sharper, larger images; a bit slower.",
            )
            _res_m, _target_px = {
                "Standard": (10.0, 700), "Fine": (5.0, 1000), "Finer": (3.0, 1300),
            }[_res_label]
            _base_label = _cb.selectbox("Basemap", ["OSM (color)", "Light", "Dark"], index=0)
            st.caption(
                f"A map is produced for every ticked level × horizon × scenario. For each scenario, SLR is "
                f"the rise in the median annual-maximum water level from {_base_year} to that horizon. "
                f"Open water (Z ≤ 0) is hidden. Maps render as static images, so there is no limit on how "
                f"many can be shown."
            )

            if not _rows:
                st.info("Tick at least one flood level to generate maps.")
            elif not _years_sel:
                st.info("Select at least one planning horizon.")
            elif not _scn_sel:
                st.info("Select at least one SLR scenario.")
            elif st.button("🌊 Generate flood maps", type="primary"):
                _lonA = df_buildings['longitude'].to_numpy(dtype=float)
                _latA = df_buildings['latitude'].to_numpy(dtype=float)
                _lonA, _latA, _swapped = fdem.maybe_swap_lonlat(_lonA, _latA)
                try:
                    _bbox = fdem.roi_from_lonlat(_lonA, _latA, buffer_m=350.0)
                    # Apply the zoom (tighten) factor: shrink the half-extent about the centre.
                    _cx = 0.5 * (_bbox[0] + _bbox[2])
                    _cy = 0.5 * (_bbox[1] + _bbox[3])
                    _hw = 0.5 * (_bbox[2] - _bbox[0]) / _zoom_factor
                    _hh = 0.5 * (_bbox[3] - _bbox[1]) / _zoom_factor
                    _bbox = (_cx - _hw, _cy - _hh, _cx + _hw, _cy + _hh)
                    _bbox_r = tuple(round(v, 5) for v in _bbox)
                except Exception as _e:
                    st.error(f"Could not determine the map area: {_e}")
                    _bbox_r = None

                _Zm = None
                _ext = None
                _base_img = None
                if _bbox_r is not None:
                    with st.spinner("Fetching terrain (USGS 3DEP) and basemap…"):
                        try:
                            _Zm, _ext = _cached_dem_roi(_bbox_r, _res_m)
                        except Exception as _e:
                            st.error(
                                "Could not load terrain for this area from USGS 3DEP "
                                "(this needs internet access at runtime). Details: %s" % _e
                            )
                        try:
                            _ztiles = fdem.tile_zoom_for_bbox(_bbox_r, target_px=_target_px)
                            _base_img, _ = _cached_basemap(_bbox_r, _ztiles, _base_label)
                        except Exception as _e:
                            st.warning(
                                "Could not load basemap tiles; maps will show flooding on a plain "
                                "background. Details: %s" % _e
                            )

                with st.expander("Map-area diagnostics"):
                    _fin = np.isfinite(_lonA) & np.isfinite(_latA)
                    st.write(
                        f"Buildings with coordinates: {int(_fin.sum())} of {len(_lonA)}"
                        + ("  •  longitude/latitude looked transposed and were auto-swapped" if _swapped else "")
                    )
                    if _fin.any():
                        st.write("lon  min / median / max: "
                                 f"{np.nanmin(_lonA[_fin]):.4f} / {np.nanmedian(_lonA[_fin]):.4f} / {np.nanmax(_lonA[_fin]):.4f}")
                        st.write("lat  min / median / max: "
                                 f"{np.nanmin(_latA[_fin]):.4f} / {np.nanmedian(_latA[_fin]):.4f} / {np.nanmax(_latA[_fin]):.4f}")
                    if _bbox_r is not None:
                        st.write(f"Region of interest (lon_min, lat_min, lon_max, lat_max): {_bbox_r}")
                    if _Zm is not None:
                        _zland = np.isfinite(_Zm) & (_Zm > 0)
                        st.write(
                            f"DEM grid: {_Zm.shape[0]} × {_Zm.shape[1]} cells (~{_res_m:.0f} m)  •  "
                            f"land cells: {int(_zland.sum())}  •  elevation "
                            f"{np.nanmin(_Zm):.1f} to {np.nanmax(_Zm):.1f} m NAVD88"
                        )

                if _Zm is not None and _ext is not None:
                    st.markdown(fdem.legend_html(), unsafe_allow_html=True)
                    _lat_m = (_ext[3] - _ext[1]) * 111320.0
                    _lon_m = (_ext[2] - _ext[0]) * 111320.0 * math.cos(math.radians(0.5 * (_ext[1] + _ext[3])))
                    _cell_m2 = (_lat_m / _Zm.shape[0]) * (_lon_m / _Zm.shape[1])
                    if _base_img is None:
                        _base_img = np.full((max(2, _Zm.shape[0]), max(2, _Zm.shape[1]), 3), 245, dtype=np.uint8)

                    for _scn in _scn_sel:
                        st.markdown(f"##### {_scn_pretty(_scn)}")
                        _cols = st.columns(2)
                        _k = 0
                        for _lbl, _base_lv in _rows:
                            for _yr in _years_sel:
                                _wl_ft = _base_lv + _slr(_scn, _yr)
                                _depth = fdem.bathtub_depth_ft(_Zm, _wl_ft, mask_water=True)
                                _nflood = int(np.isfinite(_depth).sum())
                                _flood_km2 = _nflood * _cell_m2 / 1e6
                                _note = (f"flooded ≈ {_flood_km2:.2f} km²" if _nflood > 0 else "no inundation")
                                _tgt = _cols[_k % 2]
                                # Title ABOVE the image (separate element — cannot overlap the figure).
                                _tgt.markdown(
                                    f"**{_lbl} — {int(_yr)}**<br>"
                                    f"<span style='font-size:0.95rem;color:#374151'>"
                                    f"WL ≈ {_wl_ft:.2f} ft NAVD88 &nbsp;•&nbsp; {_note}</span>",
                                    unsafe_allow_html=True,
                                )
                                _png = fdem.compose_flood_png(_base_img, _depth)
                                _tgt.image(_png, use_container_width=True)
                                _k += 1

                    st.caption(
                        f"Bathtub model (no hydraulic connectivity); open water (Z ≤ 0) hidden. Terrain: "
                        f"USGS 3DEP 1/3 arc-second (~10 m), displayed at ~{_res_m:.0f} m. Basemap: "
                        f"{_base_label} — © OpenStreetMap contributors / © CARTO."
                    )

    # ========================================================================
    # TAB: FLOODED ROADS — OSM road network classified by inundation
    # ========================================================================
    with tab_roads:
        st.markdown(
            '<p class="tab-description">OpenStreetMap roads classified against the same bathtub flood levels '
            'as the Flood Maps tab. Each road is sampled along its length, its ground elevation read from the '
            'terrain, and every segment flagged <b style="color:#dc1414">flooded</b> (surface below the water '
            'level), <b style="color:#ff8c00">proximate</b> (within the buffer of flooding), or '
            '<b style="color:#228b22">dry</b>.</p>',
            unsafe_allow_html=True,
        )

        _rfl_ok = (selected_location in data_store and df_buildings is not None)
        _rhas_xy = (
            df_buildings is not None
            and {'longitude', 'latitude'}.issubset(df_buildings.columns)
            and df_buildings['latitude'].notna().any()
            and df_buildings['longitude'].notna().any()
        )
        try:
            import rasterio as _rio2  # noqa: F401
            _rrio_ok = True
        except Exception:
            _rrio_ok = False

        if not _rfl_ok:
            st.info("Select a location with building data to map road flooding.")
        elif not _rhas_xy:
            st.warning("This location has no per-building coordinates, so the map extent can't be determined.")
        elif not _rrio_ok:
            st.error(
                "Road maps need the **rasterio** package to read the terrain. "
                "Add `rasterio` to the app's requirements.txt and redeploy."
            )
        else:
            _rwl = loc_entry.get('water_levels', {}) if loc_entry else {}
            _rscn_keys = [k for k in _rwl.keys() if not k.endswith('_mc')]
            _rscn_label = {
                '50th-percentile': 'Intermediate-High SLR (50th pct)',
                '90th-percentile': 'High SLR (90th pct)',
            }
            _rscn_pretty = lambda k: _rscn_label.get(k, k)
            _rref = '50th-percentile' if '50th-percentile' in _rwl else (_rscn_keys[0] if _rscn_keys else None)
            _rref_df = _rwl.get(_rref)
            _rbase_year = int(_rref_df['Year'].min()) if (_rref_df is not None and not _rref_df.empty) else 2025

            def _rlvl(df, year, col):
                if df is None or df.empty or col not in df.columns:
                    return None
                i = (df['Year'] - year).abs().idxmin()
                return float(df.loc[i, col])

            def _rslr(scn_key, year):
                df = _rwl.get(scn_key)
                a, b = _rlvl(df, year, 'P50'), _rlvl(df, _rbase_year, 'P50')
                return (a - b) if (a is not None and b is not None) else 0.0

            _rpf = {
                'annual': _rlvl(_rref_df, _rbase_year, 'P50'),
                'ten':    _rlvl(_rref_df, _rbase_year, 'P90'),
                'one':    _rlvl(_rref_df, _rbase_year, 'P99'),
            }

            st.markdown(
                f"**Present-day base levels** (ft NAVD88) — the same inputs as the Flood Maps tab; future "
                f"levels add each scenario's sea-level rise."
            )

            _ris_pam = (selected_location == "Pamunkey")
            if _ris_pam:
                _rdefs = [
                    ("High-tide flood",         2.37,           True,
                     "High-tide flooding (HTF) level, ft NAVD88."),
                    ("Monthly flood",           None,           False,
                     "Optional. A level reached roughly monthly, ft NAVD88."),
                    ("Annual flood",            _rpf['annual'], False,
                     "≈ the level reached most years (50th-percentile annual maximum)."),
                    ("10% annual-chance flood", 5.78,           True,
                     "1-in-10-year level (90th-percentile annual maximum)."),
                    ("1% annual-chance flood",  7.38,           True,
                     "1-in-100-year level (99th-percentile annual maximum)."),
                ]
            else:
                _rdefs = [
                    ("High-tide flood",         None,           False,
                     "Optional. e.g. the local NOAA minor/nuisance-flood threshold, ft NAVD88."),
                    ("Monthly flood",           None,           False,
                     "Optional. A level reached roughly monthly, ft NAVD88."),
                    ("Annual flood",            _rpf['annual'], _rpf['annual'] is not None,
                     "≈ the level reached most years (50th-percentile annual maximum)."),
                    ("10% annual-chance flood", _rpf['ten'],    _rpf['ten'] is not None,
                     "1-in-10-year level (90th-percentile annual maximum)."),
                    ("1% annual-chance flood",  _rpf['one'],    _rpf['one'] is not None,
                     "1-in-100-year level (99th-percentile annual maximum)."),
                ]

            _rrows = []
            for _lbl, _val, _on, _help in _rdefs:
                _c0, _c1 = st.columns([0.34, 0.66])
                _inc = _c0.checkbox(_lbl, value=_on, key=f"rdf_inc_{_lbl}", help=_help)
                _default = float(_val) if _val is not None else 0.0
                _lv = _c1.number_input(
                    f"{_lbl} level (ft NAVD88)", value=_default, step=0.5, format="%.2f",
                    key=f"rdf_val_{_lbl}", label_visibility="collapsed",
                )
                if _inc:
                    _rrows.append((_lbl, float(_lv)))

            _rc_scn, _rc_yr = st.columns(2)
            _rscn_sel = _rc_scn.multiselect(
                "SLR scenarios", options=_rscn_keys, default=_rscn_keys,
                format_func=_rscn_pretty, key="rd_scn",
            )
            _ryears_sel = _rc_yr.multiselect(
                "Planning horizons", options=available_years, default=available_years,
                format_func=lambda y: str(int(y)), key="rd_years",
            )
            _rcz, _rcd, _rcb = st.columns([0.36, 0.32, 0.32])
            _rzoom = _rcz.slider(
                "Zoom (tighten)", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
                key="rd_zoom", help="Higher = tighter crop around the community.",
            )
            _rres_label = _rcd.selectbox(
                "Map detail", ["Standard", "Fine", "Finer"], index=1, key="rd_detail",
                help="Higher detail = sharper, larger images; a bit slower.",
            )
            _rres_m, _rtarget_px = {
                "Standard": (10.0, 700), "Fine": (5.0, 1000), "Finer": (3.0, 1300),
            }[_rres_label]
            _rbase_label = _rcb.selectbox("Basemap", ["OSM (color)", "Light", "Dark"], index=0, key="rd_base")
            _rprox = st.slider(
                "Proximity buffer (m)", min_value=10, max_value=100, value=30, step=5, key="rd_prox",
                help="Roads within this distance of flooded ground are flagged 'proximate' "
                     "(Koks et al. 2019; Pregnolato et al. 2017 use ~30 m).",
            )
            st.caption(
                "A road map is produced for every ticked level × horizon × scenario. Maps render as static "
                "images. Roads come live from OpenStreetMap (Overpass) for the map area."
            )

            if not _rrows:
                st.info("Tick at least one flood level to generate maps.")
            elif not _ryears_sel:
                st.info("Select at least one planning horizon.")
            elif not _rscn_sel:
                st.info("Select at least one SLR scenario.")
            elif st.button("🛣️ Generate road maps", type="primary", key="rd_go"):
                _lonA = df_buildings['longitude'].to_numpy(dtype=float)
                _latA = df_buildings['latitude'].to_numpy(dtype=float)
                _lonA, _latA, _swap = fdem.maybe_swap_lonlat(_lonA, _latA)
                try:
                    _bb = fdem.roi_from_lonlat(_lonA, _latA, buffer_m=350.0)
                    _cx = 0.5 * (_bb[0] + _bb[2]); _cy = 0.5 * (_bb[1] + _bb[3])
                    _hw = 0.5 * (_bb[2] - _bb[0]) / _rzoom; _hh = 0.5 * (_bb[3] - _bb[1]) / _rzoom
                    _bb = (_cx - _hw, _cy - _hh, _cx + _hw, _cy + _hh)
                    _bbr = tuple(round(v, 5) for v in _bb)
                except Exception as _e:
                    st.error(f"Could not determine the map area: {_e}")
                    _bbr = None

                _Zm = None
                _ext = None
                _base_img = None
                _roads = None
                if _bbr is not None:
                    with st.spinner("Fetching terrain (3DEP), basemap, and OSM roads…"):
                        try:
                            _Zm, _ext = _cached_dem_roi(_bbr, _rres_m)
                        except Exception as _e:
                            st.error("Could not load terrain from USGS 3DEP (needs internet at runtime). %s" % _e)
                        try:
                            _zt = fdem.tile_zoom_for_bbox(_bbr, target_px=_rtarget_px)
                            _base_img, _ = _cached_basemap(_bbr, _zt, _rbase_label)
                        except Exception as _e:
                            st.warning("Basemap tiles unavailable; roads will draw on a plain background. %s" % _e)
                        try:
                            _roads = _cached_osm_roads(_bbr)
                        except Exception as _e:
                            st.error(
                                "Could not download OSM roads from Overpass (needs internet at runtime; the "
                                "public Overpass server may be busy — try again in a moment). %s" % _e
                            )

                if _Zm is not None and _ext is not None and _roads is not None:
                    st.markdown(
                        '<div style="margin:0.5rem 0 0.8rem;padding:0.55rem 0.8rem;background:#f8fafc;'
                        'border:1px solid #e2e8f0;border-radius:8px;font-size:1.05rem;">'
                        '<b style="font-size:1.2rem;">Roads</b>&nbsp;&nbsp;'
                        '<span style="color:#dc1414;font-weight:800;">━</span> flooded'
                        '&nbsp;&nbsp;&nbsp;<span style="color:#ff8c00;font-weight:800;">━</span> proximate'
                        '&nbsp;&nbsp;&nbsp;<span style="color:#228b22;font-weight:800;">━</span> dry'
                        '&nbsp;&nbsp;&nbsp;<span style="color:#3b6fb0;">▒▒</span> flood depth (blue shading)'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{len(_roads)} OpenStreetMap road way(s) in view.")
                    if _base_img is None:
                        _base_img = np.full((max(2, _Zm.shape[0]), max(2, _Zm.shape[1]), 3), 245, dtype=np.uint8)

                    with st.spinner("Classifying roads and building maps…"):
                        for _scn in _rscn_sel:
                            st.markdown(f"##### {_rscn_pretty(_scn)}")
                            _cols = st.columns(2)
                            _k = 0
                            for _lbl, _blv in _rrows:
                                for _yr in _ryears_sel:
                                    _wl = _blv + _rslr(_scn, _yr)
                                    _segs, _cnt = fdem.classify_roads(
                                        _Zm, _ext, _roads, _wl, prox_m=float(_rprox))
                                    _depth = fdem.bathtub_depth_ft(_Zm, _wl, mask_water=True)
                                    _tgt = _cols[_k % 2]
                                    _tgt.markdown(
                                        f"**{_lbl} — {int(_yr)}**<br>"
                                        f"<span style='font-size:0.95rem;color:#374151'>"
                                        f"WL ≈ {_wl:.2f} ft NAVD88 &nbsp;•&nbsp; "
                                        f"flooded {_cnt['pct_flood']:.0f}% · proximate {_cnt['pct_prox']:.0f}% · "
                                        f"dry {_cnt['pct_dry']:.0f}%</span>",
                                        unsafe_allow_html=True,
                                    )
                                    _png = fdem.compose_road_png(_base_img, _depth, _segs, _ext)
                                    _tgt.image(_png, use_container_width=True)
                                    _k += 1

                    st.caption(
                        f"Road segments: red = flooded (sampled surface below the water level), orange = within "
                        f"{int(_rprox)} m of flooding, green = dry. Percentages are by segment count. Road "
                        f"elevations and flood shading from USGS 3DEP (~10 m), displayed at ~{_rres_m:.0f} m; "
                        f"roads from OpenStreetMap. Open water (Z ≤ 0) is excluded from the flood mask."
                    )

    # ========================================================================
    # TAB: PER-BUILDING ANALYSIS — cross-building distributions + Plots 3/4/5
    # ========================================================================
    with tab_dist:
        st.markdown(
            '<p class="tab-description">Distribution of cumulative damage <b>across individual buildings</b> '
            'and counts of buildings by adaptation effectiveness. <b>Both SLR scenarios</b> are shown '
            'side-by-side for the selected target year, regardless of the SLR Scenario chosen in the sidebar. '
            'For the community-aggregated distribution (community totals across Monte Carlo realizations), '
            'see the Community Summary tab.</p>',
            unsafe_allow_html=True
        )
        
        if df_buildings is None or df_buildings.empty:
            st.warning("No per-building data available for this location.")
        else:
            df_b_year = df_buildings[df_buildings['TargetYear'] == target_year].copy()
            
            # Honor the sidebar DFE filter here too. Previously this tab
            # always ran on the full inventory regardless of the sidebar
            # selection, which made the per-building distribution panels
            # look insensitive to a control that visibly drives the Map.
            # The filter is in scope because the sidebar is rendered before
            # the tabs.
            if dfe_filter and 'DFE_Status' in df_b_year.columns:
                df_b_year = df_b_year[df_b_year['DFE_Status'].isin(dfe_filter)]
            
            if df_b_year.empty:
                st.warning(f"No per-building data for year {target_year}.")
            else:
                st.subheader(
                    f"Damage Distribution Across Buildings — {location_name} "
                    f"({occupancy_label}) — Year {target_year}"
                )
                
                # ----- Strategy ordering & labels -----
                action_order  = ['No mitigation', 'Raise Utilities', 'WFP B', 'Elevate', 'WFP 1st']
                action_labels_plain = {
                    'No mitigation':   'No Mitigation',
                    'Raise Utilities': 'Raise Utilities',
                    'WFP B':           'WFP Basement',
                    'WFP 1st':         'WFP 1st Floor',
                    'Elevate':         'Elevate',
                }
                actions_present = [a for a in action_order if a in df_b_year['Action'].unique()]
                
                # When the user has restricted the view to ONLY Above-DFE
                # buildings (DFE_Status == 'Above DFE'), drop
                # Elevate from every Distributions chart on this tab. The
                # data generator treats Elevate as a no-op for those buildings
                # (Elevate damage = baseline damage by construction), so an
                # Elevate box would just mirror the No-Mitigation box and
                # mislead readers into thinking elevation does nothing.
                # The opposite case (only Under-DFE selected) keeps Elevate.
                only_above_dfe = (
                    dfe_filter
                    and len(dfe_filter) == 1
                    and 'DFE_Status' in df_b_year.columns
                    and {str(v).strip().lower() for v in dfe_filter} <= {
                        'out of floodplain', 'out_of_floodplain', 'above dfe'
                    }
                )
                if only_above_dfe and 'Elevate' in actions_present:
                    actions_present = [a for a in actions_present if a != 'Elevate']
                
                thr = ZERO_THRESH_DISPLAY
                
                # =============================================================
                # CROSS-BUILDING DISTRIBUTION PANELS
                # Two panels: left = distribution of per-building MEDIAN (P50)
                #             right = distribution of per-building UPPER-TAIL (P95)
                # Each box summarizes one statistic across damaged buildings.
                # Mirrors the workshop's make_perbldg_bw / pb_p50, pb_p90.
                # =============================================================
                
                # Filter to "damaged" buildings: those with No-Mitigation P50 > thr
                # under at least one SLR (so the same building-set is used for
                # all strategies, matching the workshop).
                d_nomit = df_b_year[df_b_year['Action'] == 'No mitigation']
                damaged_ids = set()
                n_total = 0
                for slr_key, _, _, _ in SCENARIO_SPECS:
                    ds = d_nomit[d_nomit['SLR'] == slr_key]
                    if ds.empty:
                        continue
                    n_total = max(n_total, int(ds['id'].nunique()))
                    damaged_ids.update(ds.loc[ds['CumEAD_P50'] > thr, 'id'].tolist())
                n_aff = len(damaged_ids)
                
                if n_aff == 0:
                    st.info(
                        f"No buildings exceed the median-damage threshold (≥ ${thr:,}) "
                        f"under either SLR scenario for {target_year}. "
                        "Cross-building distribution panels are not shown."
                    )
                else:
                    st.markdown(
                        f"**{n_aff:,} of {n_total:,} buildings damaged** "
                        f"(median cumulative damage > ${thr:,} under at least one SLR scenario)"
                    )
                    
                    # Pre-pivot building-level damages for fast access
                    # Indexed by building id, columns = (Action, SLR) → P50 / P95
                    pb = df_b_year[df_b_year['id'].isin(damaged_ids)]
                    
                    def _cross_bldg_stats(stat_col):
                        """Build scenario_data for the box-whisker helper, where
                        each 'group' is a strategy and the box summarizes the
                        distribution of `stat_col` across damaged buildings.

                        Returns 5-tuples ``(p05, p25, p50, p75, p95)`` per
                        group: whiskers reach to the 5th/95th percentiles of
                        damage across buildings, and box edges sit at the
                        25th/75th percentiles of the same distribution.
                        Computing P25 and P75 directly from the building
                        values means the box edges are real quartiles of the
                        cross-building distribution rather than CDF-linear
                        approximations between P05 and P95. The wider
                        5th/95th window (vs the earlier 10th/90th) surfaces
                        residual damage on Elevate that a tighter window
                        would clip out — many buildings only see non-zero
                        post-elevation damage in the upper tail of the
                        cross-building spread.
                        """
                        out = {slr_key: [] for slr_key, *_ in SCENARIO_SPECS}
                        for action in actions_present:
                            for slr_key, *_ in SCENARIO_SPECS:
                                vals = pb[(pb['Action'] == action) &
                                          (pb['SLR'] == slr_key)][stat_col].values
                                if len(vals) == 0:
                                    out[slr_key].append(None)
                                    continue
                                p05 = float(np.percentile(vals, 5))
                                p25 = float(np.percentile(vals, 25))
                                p50 = float(np.percentile(vals, 50))
                                p75 = float(np.percentile(vals, 75))
                                p95 = float(np.percentile(vals, 95))
                                # build_box_whisker_panel reads a 5-tuple as
                                # (lower-whisker, Q1, median, Q3, upper-whisker)
                                # — exactly the cross-building 5/25/50/75/95
                                # we just computed. No interpolation needed.
                                out[slr_key].append((p05, p25, p50, p75, p95))
                        return out
                    
                    sd_p50 = _cross_bldg_stats('CumEAD_P50')
                    sd_p95 = _cross_bldg_stats('CumEAD_P95')
                    
                    fig_pb_left = build_box_whisker_panel(
                        group_labels=[action_labels_plain[a] for a in actions_present],
                        scenario_data=sd_p50,
                        panel_title=(
                            "Median per-building damage: distribution across "
                            f"{n_aff:,} damaged buildings"
                        ),
                        y_label="Per-Building Cumulative Damage",
                        lower_label="P05", upper_label="P95",
                        lower_pct=0.05, upper_pct=0.95,
                    )
                    fig_pb_right = build_box_whisker_panel(
                        group_labels=[action_labels_plain[a] for a in actions_present],
                        scenario_data=sd_p95,
                        panel_title=(
                            "Upper-tail (P95) per-building damage: distribution across "
                            f"{n_aff:,} damaged buildings"
                        ),
                        y_label="Per-Building Cumulative Damage",
                        lower_label="P05", upper_label="P95",
                        lower_pct=0.05, upper_pct=0.95,
                    )
                    
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.plotly_chart(fig_pb_left, use_container_width=True)
                    with col_r:
                        st.plotly_chart(fig_pb_right, use_container_width=True)
                    
                    st.caption(
                        "Each box summarizes how a single damage statistic varies across the "
                        "damaged buildings in this community. The **left panel** shows the "
                        "distribution of each building's *median* cumulative damage; the "
                        "**right panel** shows the distribution of each building's *upper-tail "
                        "(P95)* cumulative damage. Box edges show the 25th and 75th percentiles "
                        "across buildings, the white center line is the median across buildings, "
                        "and whiskers extend to the 5th and 95th percentiles across buildings."
                    )
                
                st.divider()
                
                # =============================================================
                # PLOTS 3, 4, 5 — Per-building damage classification
                # Ported from VVV_Visualization_for_workshop_MasticBeach.py
                # Uses per-building P90 as upper-tail proxy (matches the
                # workshop convention; P95 was tested earlier but the user
                # asked for P90 because it's less sensitive to the tail's
                # smallest realizations and reads more conservatively).
                # =============================================================
                st.subheader(f"Building Counts by Adaptation Effectiveness — Year {target_year}")
                
                # Compute per-scenario stats (analog of compute_bldg_stats)
                per_scen_stats = {}
                for slr_key, slr_label, line_clr, _fill in SCENARIO_SPECS:
                    ds = df_b_year[df_b_year['SLR'] == slr_key]
                    if ds.empty:
                        per_scen_stats[slr_key] = None
                        continue
                    
                    d_nomit_s  = ds[ds['Action'] == 'No mitigation'].set_index('id')
                    d_wfpb_s   = ds[ds['Action'] == 'WFP B'].set_index('id')
                    d_elev_s   = ds[ds['Action'] == 'Elevate'].set_index('id')
                    # Raise Utilities only matters for the Pamunkey variant of
                    # fig4 below — we still always compute it so the stats
                    # dict has a stable schema across locations.
                    d_raiseu_s = ds[ds['Action'] == 'Raise Utilities'].set_index('id')
                    
                    n_tot_s = int(d_nomit_s.index.nunique()) if not d_nomit_s.empty else 0
                    if n_tot_s == 0:
                        per_scen_stats[slr_key] = None
                        continue
                    
                    ids_s = d_nomit_s.index
                    no_p50 = d_nomit_s['CumEAD_P50'].reindex(ids_s).fillna(0).values
                    no_p90 = d_nomit_s['CumEAD_P90'].reindex(ids_s).fillna(0).values
                    wb_p90 = (d_wfpb_s['CumEAD_P90'].reindex(ids_s).fillna(np.nan).values
                              if not d_wfpb_s.empty else None)
                    el_p90 = (d_elev_s['CumEAD_P90'].reindex(ids_s).fillna(np.nan).values
                              if not d_elev_s.empty else None)
                    ru_p90 = (d_raiseu_s['CumEAD_P90'].reindex(ids_s).fillna(np.nan).values
                              if not d_raiseu_s.empty else None)
                    
                    any_damage = no_p90 > thr
                    mask_p50   = no_p50 > thr
                    mask_sev   = no_p90 > thr
                    
                    # --- WFP-Basement bucket: same threshold rule as Map ---
                    # A damaged building counts here iff WFP B brings P90 ≤ $1k.
                    # WFP B is the cheapest fix that "eliminates" damage, so the
                    # threshold reading still makes sense for this bucket.
                    mask_wfpb = np.zeros(n_tot_s, dtype=bool)
                    if wb_p90 is not None:
                        wb_arr = np.where(np.isnan(wb_p90), no_p90, wb_p90)
                        mask_wfpb = any_damage & (wb_arr <= thr)
                    
                    # --- Raise-Utilities bucket: same threshold rule as WFP B ---
                    # A damaged building counts here iff Raise Utilities brings
                    # P90 ≤ $1k. Used by the Pamunkey variant of fig4 where
                    # basement floodproofing isn't a relevant retrofit
                    # (manufactured housing / RES2-dominant inventory).
                    mask_raiseu = np.zeros(n_tot_s, dtype=bool)
                    if ru_p90 is not None:
                        ru_arr = np.where(np.isnan(ru_p90), no_p90, ru_p90)
                        mask_raiseu = any_damage & (ru_arr <= thr)
                    
                    # --- Elevation bucket: dominance rule (matches the Map) ---
                    # An "Elevation" building is one that is damaged at baseline,
                    # WFP Basement is NOT enough on its own, AND Elevation
                    # strictly outperforms WFP Basement on P90 — i.e., elevation
                    # provides meaningful additional protection beyond WFP B.
                    # This replaces the earlier strict ≤$1k rule, which was too
                    # demanding under high MC realizations: most damaged
                    # Under-DFE buildings can't get P90 below $1k with
                    # elevation either, even though Elevate is substantially
                    # better than WFP B for them (Mastic Beach 2055-50th: median
                    # P90 ~$27k under Elevate vs ~$41k under WFP B). The
                    # dominance rule also auto-handles the MATLAB convention
                    # where Above-DFE buildings have Elevate = baseline (those
                    # have Elevate ≥ WFP B by construction → they fall through
                    # to "Residual" rather than getting credited as Elevation
                    # successes).
                    mask_elev_works = np.zeros(n_tot_s, dtype=bool)
                    if el_p90 is not None and wb_p90 is not None:
                        el_arr = np.where(np.isnan(el_p90), no_p90, el_p90)
                        wb_arr = np.where(np.isnan(wb_p90), no_p90, wb_p90)
                        wfpb_to_thr = wb_arr <= thr
                        elev_dom    = el_arr < wb_arr
                        mask_elev_works = (any_damage
                                           & ~wfpb_to_thr
                                           & elev_dom)
                    
                    per_scen_stats[slr_key] = {
                        'label':     slr_label,
                        'color':     line_clr,
                        'n_tot':     n_tot_s,
                        'n_p50_dmg': int(mask_p50.sum()),
                        'n_sev_dmg': int(mask_sev.sum()),
                        'n_damaged': int(any_damage.sum()),
                        'n_wfpb':    int(mask_wfpb.sum()),
                        'n_raiseu':  int(mask_raiseu.sum()),
                        'n_elev':    int(mask_elev_works.sum()),
                    }
                
                valid_stats = {k: v for k, v in per_scen_stats.items() if v is not None}
                if not valid_stats:
                    st.warning("No per-building data available for the selected year.")
                else:
                    n_tot_max = max(s['n_tot'] for s in valid_stats.values())
                    
                    def _make_paired_bar(title, value_fn, count_fn,
                                         x_left='Median damage > $0',
                                         x_right='P90 damage > $0',
                                         single_group=False):
                        fig = go.Figure()
                        if single_group:
                            cat_labels = [x_left]
                        else:
                            cat_labels = [x_left, x_right]
                        y_max = 0.0
                        for slr_key, slr_label, line_clr, _fill in SCENARIO_SPECS:
                            if slr_key not in valid_stats:
                                continue
                            stats = valid_stats[slr_key]
                            pcts = value_fn(stats); cnts = count_fn(stats)
                            fig.add_trace(go.Bar(
                                name=slr_label,
                                x=cat_labels,                # CATEGORICAL x
                                y=pcts,
                                offsetgroup=slr_key,         # side-by-side pair
                                alignmentgroup='slr_pair',
                                marker=dict(color=line_clr, opacity=0.88,
                                            line=dict(color='white', width=1.5)),
                                text=[f"<b>{p:.1f}%</b><br>({n:,})"
                                      for p, n in zip(pcts, cnts)],
                                textposition='outside',
                                textfont=dict(size=13, color=line_clr),
                                cliponaxis=False,
                                customdata=list(zip(cat_labels, cnts)),
                                hovertemplate=(
                                    f"<b>{slr_label}</b><br>"
                                    "%{customdata[0]}<br>"
                                    "Share: %{y:.1f}%<br>"
                                    "Buildings: %{customdata[1]:,}"
                                    "<extra></extra>"
                                ),
                            ))
                            y_max = max(y_max, max(pcts) if len(pcts) > 0 else 0)
                        y_head = min(100.0, max(y_max + 18, 25))
                        fig.update_layout(
                            title=dict(text=title, x=0.02, xanchor='left',
                                       font=dict(size=15)),
                            height=420,
                            plot_bgcolor='white', paper_bgcolor='white',
                            showlegend=True,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                        xanchor='right', x=1,
                                        bgcolor='rgba(255,255,255,0.85)'),
                            margin=dict(l=60, r=20, t=70, b=70),
                            barmode='group',
                            bargap=0.30,
                            bargroupgap=0.10,
                            xaxis=dict(
                                type='category',
                                categoryorder='array',
                                categoryarray=cat_labels,
                                showgrid=False, showline=True, linecolor='#cbd5e1',
                                zeroline=False,
                            ),
                            yaxis=dict(
                                title="Share of buildings (%)",
                                showgrid=True, gridcolor='#e5e7eb',
                                showline=True, linecolor='#cbd5e1', zeroline=False,
                                range=[0, y_head], ticksuffix='%',
                            ),
                        )
                        return fig
                    
                    # Plot 3
                    def _v3(s): return [
                        100.0 * s['n_p50_dmg'] / s['n_tot'] if s['n_tot'] else 0,
                        100.0 * s['n_sev_dmg'] / s['n_tot'] if s['n_tot'] else 0,
                    ]
                    def _c3(s): return [s['n_p50_dmg'], s['n_sev_dmg']]
                    fig3 = _make_paired_bar(
                        f"Buildings experiencing flood damage by {target_year} "
                        f"(of {n_tot_max:,} buildings)",
                        _v3, _c3,
                        x_left='Median damage > $0',
                        x_right='Upper-tail (P90) damage > $0',
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Plots 4 & 5
                    # For Pamunkey, basement floodproofing isn't a relevant
                    # retrofit (the inventory is RES2-heavy / mostly without
                    # basements), so we swap the "WFP Basement eliminates P90"
                    # bucket for the analogous "Raise Utilities eliminates
                    # P90" bucket. Everything downstream (caption, summary
                    # table column) follows the same swap so the labels stay
                    # internally consistent.
                    _use_raiseu_for_fig4 = (location_name == "Pamunkey")
                    if _use_raiseu_for_fig4:
                        def _v4(s):
                            nd = s['n_damaged']
                            return [100.0 * s['n_raiseu'] / nd if nd > 0 else 0]
                        def _c4(s): return [s['n_raiseu']]
                        fig4 = _make_paired_bar(
                            f"Damaged buildings where Raise Utilities eliminates "
                            f"upper-tail damage by {target_year}",
                            _v4, _c4,
                            x_left='Raise Utilities eliminates P90 damage',
                            single_group=True,
                        )
                    else:
                        def _v4(s):
                            nd = s['n_damaged']
                            return [100.0 * s['n_wfpb'] / nd if nd > 0 else 0]
                        def _c4(s): return [s['n_wfpb']]
                        fig4 = _make_paired_bar(
                            f"Damaged buildings where WFP Basement eliminates "
                            f"upper-tail damage by {target_year}",
                            _v4, _c4,
                            x_left='WFP Basement eliminates P90 damage',
                            single_group=True,
                        )
                    
                    def _v5(s):
                        nd = s['n_damaged']
                        return [100.0 * s['n_elev'] / nd if nd > 0 else 0]
                    def _c5(s): return [s['n_elev']]
                    fig5 = _make_paired_bar(
                        f"Damaged buildings where Elevation outperforms "
                        f"WFP Basement on P90 damage by {target_year}",
                        _v5, _c5,
                        x_left='Elevation strictly beats WFP Basement on P90',
                        single_group=True,
                    )
                    
                    # When only Above-DFE buildings are selected, Elevate is
                    # a no-op in the data generator (Elevate damage = baseline
                    # by construction), so the "Elevation outperforms WFP B"
                    # chart would be 0 by definition and the corresponding
                    # column in the summary table would mislead. Hide both.
                    if only_above_dfe:
                        col_p4, = st.columns(1)
                        with col_p4:
                            st.plotly_chart(fig4, use_container_width=True)
                    else:
                        col_p4, col_p5 = st.columns(2)
                        with col_p4:
                            st.plotly_chart(fig4, use_container_width=True)
                        with col_p5:
                            st.plotly_chart(fig5, use_container_width=True)
                    
                    # Per-scenario summary table
                    tbl_rows = []
                    for slr_key in ['50th-percentile', '90th-percentile']:
                        if slr_key not in valid_stats:
                            continue
                        s = valid_stats[slr_key]
                        nd = s['n_damaged']
                        # Match the table's eliminator column to the strategy
                        # used by fig4 so the figure and the table never tell
                        # different stories.
                        if _use_raiseu_for_fig4:
                            elim_col_label = 'Raise Utilities eliminates P90'
                            elim_count = s['n_raiseu']
                        else:
                            elim_col_label = 'WFP Basement eliminates P90'
                            elim_count = s['n_wfpb']
                        row = {
                            'SLR Scenario':              s['label'],
                            'Buildings':                 f"{s['n_tot']:,}",
                            'Damaged (P90 > $0)':        f"{s['n_sev_dmg']:,}  ({100*s['n_sev_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "—",
                            'Damaged (median > $0)':     f"{s['n_p50_dmg']:,}  ({100*s['n_p50_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "—",
                            elim_col_label:              f"{elim_count:,}  ({100*elim_count/nd:.1f}%)" if nd > 0 else "—",
                        }
                        if not only_above_dfe:
                            row['Elevation > WFP B (where WFP B fails)'] = (
                                f"{s['n_elev']:,}  ({100*s['n_elev']/nd:.1f}%)" if nd > 0 else "—"
                            )
                        tbl_rows.append(row)
                    if tbl_rows:
                        st.markdown("**Per-scenario summary**")
                        st.dataframe(pd.DataFrame(tbl_rows),
                                     use_container_width=True, hide_index=True)
                    
                    if _use_raiseu_for_fig4:
                        st.caption(
                            "Per-building counts use the **P90** of the per-building cumulative damage as "
                            "the upper-tail proxy (matching the workshop visualization convention). "
                            "The **damaged-buildings chart** shows the share of buildings with median "
                            "damage greater than zero and the share with P90 damage greater than zero. "
                            "The **Raise Utilities chart** shows, among buildings that experience any "
                            "damage, the share for which raising utilities above BFE+2 ft brings P90 "
                            "damage to ≤ $1k. The **Elevation chart** shows, among damaged buildings "
                            "where WFP Basement is **not** sufficient on its own, the share for which "
                            "elevation strictly outperforms WFP Basement on P90 — i.e. elevation "
                            "provides meaningful additional protection beyond what basement "
                            "floodproofing achieves. This dominance rule matches the Map tab's "
                            "Adaptation Effectiveness classifier "
                            "(No Damage → WFP Basement → Elevation → Residual)."
                        )
                    else:
                        st.caption(
                            "Per-building counts use the **P90** of the per-building cumulative damage as "
                            "the upper-tail proxy (matching the workshop visualization convention). "
                            "The **damaged-buildings chart** shows the share of buildings with median "
                            "damage greater than zero and the share with P90 damage greater than zero. "
                            "The **WFP Basement chart** shows, among buildings that experience any "
                            "damage, the share for which wet-floodproofing the basement brings P90 "
                            "damage to ≤ $1k. The **Elevation chart** shows, among damaged buildings "
                            "where WFP Basement is **not** sufficient on its own, the share for which "
                            "elevation strictly outperforms WFP Basement on P90 — i.e. elevation "
                            "provides meaningful additional protection beyond what basement "
                            "floodproofing achieves. This dominance rule matches the Map tab's "
                            "Adaptation Effectiveness classifier "
                            "(No Damage → WFP Basement → Elevation → Residual)."
                        )
                
    
    # ========================================================================
    # TAB 2: BUILDING MAP
    # ========================================================================
    with tab2:
        st.markdown(
            '<p class="tab-description">Interactive map showing building-level flood risk. '
            'Use the <b>Map View</b> selector to switch between damage intensity, adaptation '
            'effectiveness, and binned damage maps. Hover any building to compare baseline '
            'damage with all adaptation strategies.</p>',
            unsafe_allow_html=True
        )
        
        # The map requires per-building longitude/latitude. The bundle
        # format guarantees these columns are present (they live in
        # bldg_lookup.csv). The guard remains as a safety net for any
        # future loader that might not populate them.
        _has_coords = (
            df_buildings is not None
            and {'longitude', 'latitude'}.issubset(df_buildings.columns)
            and df_buildings['latitude'].notna().any()
            and df_buildings['longitude'].notna().any()
        )

        if df_buildings is not None and not _has_coords:
            st.info(
                "🗺️ Map view is unavailable for this dataset — building "
                "coordinates are missing. All other tabs remain fully functional."
            )
        elif df_buildings is not None:
            st.subheader(f"Building Risk Map — {location_name} ({occupancy_label}) — {target_year}, {scenario}")

            map_view = st.radio(
                "Map View",
                options=["Damage Heatmap", "Damage Bins", "Adaptation Effectiveness"],
                horizontal=True,
                key="map_view_selector",
                help=(
                    "**Damage Heatmap**: continuous color by No-Mitigation P50 cumulative damage. "
                    "**Damage Bins**: discrete bins of upper-tail damage with breakpoints fixed across years. "
                    "**Adaptation Effectiveness**: classifies each building by which retrofit eliminates upper-tail damage."
                ),
            )

            # Basemap choice for the live map. "Streets" is the original
            # OpenStreetMap default; "Aerial" overlays ESRI World Imagery
            # tiles, which is useful for locating individual buildings,
            # spotting parking lots, vegetation, and shoreline detail
            # that an abstract street map doesn't show. The choice is
            # remembered globally (one key, not per-location) since it's
            # a UI preference rather than data.
            basemap_choice = st.radio(
                "Basemap",
                options=["Streets", "Aerial"],
                horizontal=True,
                key="map_basemap_choice",
                help=(
                    "**Streets**: OpenStreetMap road network and place labels. "
                    "**Aerial**: satellite imagery (ESRI World Imagery) — useful for "
                    "spotting individual buildings, parking lots, vegetation, and "
                    "shoreline detail. Requires internet access at render time; if "
                    "the tile server is unreachable the map will fall back to a "
                    "white background."
                ),
            )
            
            # ----------------------------------------------------------
            # Building-ID search — drops a temporary highlight ring on
            # the map. The ring is plotted from the same figure as the
            # rest of the markers (so it survives pan/zoom and exports
            # correctly), but it auto-expires after a few seconds so the
            # user can take a clean screenshot without manually undoing
            # anything. Implementation:
            #   * The search input writes (building_id, timestamp) to
            #     session state.
            #   * Each time the map renders, we check whether the
            #     timestamp is still "fresh" (within HIGHLIGHT_TTL_SEC).
            #     If it is, we add the ring trace; if it isn't, we don't.
            #   * A small "Clear" button next to the input lets the user
            #     dismiss the ring instantly without waiting for the TTL.
            #   * Streamlit re-renders any time the user interacts with
            #     the page (zoom, pan, scroll, etc.), and on each
            #     re-render the TTL check runs again — so the ring
            #     disappears on its own without any explicit timer.
            HIGHLIGHT_TTL_SEC = 8.0
            # Four columns: text input | Find+Clear buttons | Point-size slider
            # | spacer. The buttons themselves are made shorter (less vertical
            # padding) via the global `.stButton > button` CSS rule at the top
            # of the file — that's how we get a compact button row WITHOUT
            # squashing the labels onto two lines. Width here is just wide
            # enough that "🔍 Find" and "✖ Clear" each fit on a single line.
            search_col1, search_col2, search_col3, search_col4 = st.columns([3, 2, 3, 2])
            with search_col1:
                search_id_text = st.text_input(
                    "Find building by ID",
                    value="",
                    key="map_search_bldg_id",
                    placeholder="e.g. 8466717",
                    help=(
                        "Enter a Building ID and press Enter (or click 🔍). "
                        f"A magenta ring will flash on that building for "
                        f"~{int(HIGHLIGHT_TTL_SEC)} seconds, then disappear "
                        "automatically — no extra marks left on the map for "
                        "screenshots. Click ✖ to clear immediately."
                    ),
                )
            with search_col2:
                # Spacer above the buttons so they vertically line up with
                # the text input baseline (Streamlit reserves label space
                # above the input). Slightly larger than the original 1.85em
                # because the buttons are now shorter (compact CSS rule), so
                # they need to be pushed down a hair further to stay flush.
                st.markdown("<div style='height:2.05em'></div>", unsafe_allow_html=True)
                btn_a, btn_b = st.columns(2)
                with btn_a:
                    find_clicked = st.button("🔍 Find", key="map_search_find",
                                             use_container_width=True)
                with btn_b:
                    clear_clicked = st.button("✖ Clear", key="map_search_clear",
                                              use_container_width=True)
            with search_col3:
                # Point-size slider. Each location remembers its own setting
                # via a location-scoped key, so switching study sites doesn't
                # carry Pamunkey's bumped-up size over to Mastic Beach (and
                # vice versa). Pamunkey defaults to 2× because its small
                # building count makes the default markers read as tiny dots
                # at the natural map zoom; other locations default to 1×.
                _default_scale = 2.0 if location_name == "Pamunkey" else 1.0
                _point_scale = st.slider(
                    "Point size",
                    min_value=0.5, max_value=4.0,
                    value=_default_scale, step=0.25,
                    format="%.2fx",
                    key=f"map_point_scale_{location_name}",
                    help=(
                        "Scale all building markers up or down on the map. "
                        "Useful at locations with few buildings (e.g. Pamunkey) "
                        "where the default dots look small at the natural zoom."
                    ),
                )
            with search_col4:
                # Spacer column; deliberately empty.
                pass
            
            # Resolve the click → session-state interaction
            import time as _time_mod
            if 'map_highlight_id' not in st.session_state:
                st.session_state['map_highlight_id']    = None
                st.session_state['map_highlight_until'] = 0.0
                st.session_state['map_highlight_msg']   = None
            
            if clear_clicked:
                st.session_state['map_highlight_id']    = None
                st.session_state['map_highlight_until'] = 0.0
                st.session_state['map_highlight_msg']   = None
            elif find_clicked and search_id_text.strip():
                # Try to coerce the user's input to the same dtype as the
                # 'id' column. The bundle stores ids as ints, but a user
                # might paste them with whitespace, leading zeros, or a
                # stray decimal point — be forgiving.
                raw = search_id_text.strip()
                try:
                    bid_target = int(float(raw))
                except (TypeError, ValueError):
                    bid_target = None
                
                if bid_target is None:
                    st.session_state['map_highlight_msg'] = (
                        'warning', f"'{raw}' isn't a valid building ID."
                    )
                    st.session_state['map_highlight_id']    = None
                    st.session_state['map_highlight_until'] = 0.0
                else:
                    found_row = df_buildings[df_buildings['id'] == bid_target]
                    if found_row.empty:
                        st.session_state['map_highlight_msg'] = (
                            'warning',
                            f"Building #{bid_target} isn't in the current "
                            f"selection. Try clearing the DFE / occupancy "
                            f"filters or check the ID."
                        )
                        st.session_state['map_highlight_id']    = None
                        st.session_state['map_highlight_until'] = 0.0
                    else:
                        st.session_state['map_highlight_id']    = bid_target
                        st.session_state['map_highlight_until'] = (
                            _time_mod.time() + HIGHLIGHT_TTL_SEC
                        )
                        st.session_state['map_highlight_msg'] = (
                            'success',
                            f"Highlighting #{bid_target} for "
                            f"~{int(HIGHLIGHT_TTL_SEC)}s."
                        )
            
            # Surface any feedback message from the resolution above.
            _hl_msg = st.session_state.get('map_highlight_msg')
            if _hl_msg:
                _kind, _text = _hl_msg
                if _kind == 'warning':
                    st.warning(_text, icon='⚠️')
                elif _kind == 'success':
                    st.caption(f"✓ {_text}")
            
            df_map = prepare_map_data(df_buildings, target_year, scenario)
            
            if df_map is None or len(df_map) == 0:
                st.warning("No buildings match the current filters.")
            else:
                if dfe_filter and 'DFE_Status' in df_map.columns:
                    df_map = df_map[df_map['DFE_Status'].isin(dfe_filter)]
                
                # ----------------------------------------------------------
                # "Hide $0-damage buildings" — pick the right damage metric
                # ----------------------------------------------------------
                # The hide filter must look at the SAME statistic the active
                # map view colors by, otherwise we hide buildings the user
                # would expect to see:
                #   * Damage Heatmap        → P50 (what the heatmap colors)
                #   * Damage Bins           → P90 (the bins are upper-tail)
                #   * Adaptation Effective. → P90 (categories are P90-based)
                # In particular, for the bins/effectiveness views, hiding by
                # P50 silently drops every building with P50 = 0 but
                # P90 > $1k — the very buildings that drive tail-risk
                # planning.
                if map_view == "Damage Heatmap":
                    zero_filter_col = 'No mitigation_P50' if 'No mitigation_P50' in df_map.columns else None
                else:
                    # Upper-tail view — fall back to P50 only if P90 isn't loaded
                    zero_filter_col = (
                        'No mitigation_P90' if 'No mitigation_P90' in df_map.columns
                        else 'No mitigation_P50' if 'No mitigation_P50' in df_map.columns
                        else None
                    )
                
                # The downstream colorbar/hover code still keys off P50
                # (it expects the heatmap's coloring metric), so keep
                # baseline_col aligned with the heatmap convention.
                baseline_col = 'No mitigation_P50' if 'No mitigation_P50' in df_map.columns else None
                
                if zero_filter_col and not show_zero_damage:
                    df_map = df_map[df_map[zero_filter_col] > 0]
                
                if len(df_map) == 0:
                    st.warning("No buildings match the current filters.")
                else:
                    if baseline_col:
                        non_zero_damages = df_map[df_map[baseline_col] > 0][baseline_col]
                        if len(non_zero_damages) > 0:
                            # Absolute max damage (used for hover context)
                            max_damage_raw = float(non_zero_damages.max())
                            # Cap the colorbar at the 90th percentile of nonzero
                            # damages so the gradient is readable across the bulk
                            # of buildings instead of being compressed by the
                            # single worst outlier. Buildings above the cap still
                            # render at the top red (their true damage shows in
                            # the hover tooltip).
                            p90_damage = float(non_zero_damages.quantile(0.90))
                            # Snap to a nice rounded value for the legend
                            color_cap = nice_round_up(p90_damage)
                            # If the data is very flat (cap collapses to 0), fall
                            # back to the true max so we don't render a dead scale.
                            if color_cap <= 0 or color_cap >= max_damage_raw:
                                color_cap = max_damage_raw
                                cap_is_clipped = False
                            else:
                                cap_is_clipped = True
                        else:
                            max_damage_raw = 1.0
                            color_cap = 1.0
                            cap_is_clipped = False
                        max_damage = color_cap   # legacy alias used elsewhere
                    else:
                        max_damage_raw = 1.0
                        color_cap = 1.0
                        cap_is_clipped = False
                        max_damage = 1.0
                    
                    action_cols_p50 = [col for col in df_map.columns if col.endswith('_P50')]
                    
                    # ---- Mark non-residential buildings (used by categorical maps) ----
                    if 'occupancy_type' in df_map.columns:
                        is_nonres_series = ~df_map['occupancy_type'].apply(is_residential)
                    else:
                        is_nonres_series = pd.Series(False, index=df_map.index)
                    df_map = df_map.copy()
                    df_map['_is_nonres'] = is_nonres_series.values
                    
                    # ---- Build the standard hover text (shared across all views) ----
                    # Hover text is duplicated across every map trace × every
                    # building, so each byte multiplies into the JSON payload.
                    # We keep the new NSI fields (address, year built, etc.)
                    # but render them compactly so even ~5,000-building
                    # locations stay well under Streamlit's per-chart JSON
                    # limit (~3 MB before the frontend starts truncating).
                    hover_texts = []
                    for idx, row in df_map.iterrows():
                        addr = row.get('address') if 'address' in row else None
                        if pd.notna(addr) and str(addr).strip():
                            text = f"<b>{addr}</b> <span style='color:#94a3b8'>#{row['id']}</span><br>"
                        else:
                            text = f"<b>Building #{row['id']}</b><br>"
                        
                        # One compact attribute line: type · stories · sqft · foundation · year
                        attr_bits = []
                        if 'occupancy_type' in row and pd.notna(row.get('occupancy_type')):
                            attr_bits.append(str(row['occupancy_type']))
                        if 'number_of_stories' in row and pd.notna(row.get('number_of_stories')):
                            attr_bits.append(f"{int(row['number_of_stories'])}-story")
                        if 'area' in row and pd.notna(row.get('area')):
                            attr_bits.append(f"{int(row['area']):,} sqft")
                        if 'foundation_type' in row and pd.notna(row.get('foundation_type')):
                            attr_bits.append(f"fnd {row['foundation_type']}")
                        if 'year_built' in row and pd.notna(row.get('year_built')):
                            attr_bits.append(f"{int(row['year_built'])}")
                        if attr_bits:
                            text += " · ".join(attr_bits) + "<br>"
                        
                        # One compact value line: structure value + DFE status
                        val_bits = []
                        if 'structure_value' in row and pd.notna(row['structure_value']):
                            val_bits.append(f"{format_currency(row['structure_value'])}")
                        if 'DFE_Status' in row and pd.notna(row.get('DFE_Status')):
                            val_bits.append(str(row['DFE_Status']))
                        if val_bits:
                            text += " · ".join(val_bits) + "<br>"
                        
                        text += "<br><b>Cumulative Damage by strategy</b><br>"

                        # No Mitigation gets two values because the three map
                        # views use TWO different statistics to color the
                        # buildings:
                        #   • Damage Heatmap         → P50 (median)
                        #   • Damage Bins            → P90 (upper tail)
                        #   • Adaptation Effectiveness → P90 (upper tail)
                        # Showing only the P50 here used to confuse users on
                        # the P90-based views: a building with P50 = $14.6k
                        # and P90 = $250k would land in a "$100k–$500k" bin
                        # and look misclassified, because the user couldn't
                        # see the P90 value that put it there. Surface both
                        # so the bin / category placement is always traceable
                        # from the hover.
                        baseline_p50 = row.get('No mitigation_P50', 0)
                        baseline_p90 = row.get('No mitigation_P90', np.nan)
                        baseline_val = baseline_p50  # the rest of the loop uses P50 for savings %

                        if (pd.notna(baseline_p90)
                            and pd.notna(baseline_p50)
                            and not np.isclose(baseline_p50, baseline_p90)):
                            text += (
                                f"<b>No Mitigation</b>: "
                                f"{format_currency(baseline_p50)} (median) · "
                                f"{format_currency(baseline_p90)} (P90)<br>"
                            )
                            _no_mit_shown_inline = True
                        else:
                            _no_mit_shown_inline = False

                        for col in action_cols_p50:
                            action_name = col.replace('_P50', '')
                            val = row.get(col, 0)

                            # Skip retrofits that don't physically apply to this
                            # building. Those rows were dropped in load_bundle's
                            # applicability filter, which means the wide-form
                            # pivot in prepare_map_data() left this cell NaN.
                            # Rendering a NaN here would either show "$nan" or,
                            # via the format_currency rounding, a misleading $0.
                            # Skipping the line entirely is what "hide it
                            # completely" actually looks like in the hover.
                            if action_name != 'No mitigation' and pd.isna(val):
                                continue

                            # Skip the No-mitigation column here if we already
                            # rendered it inline with both percentiles above.
                            if action_name == 'No mitigation' and _no_mit_shown_inline:
                                continue

                            display_name = action_name
                            if action_name == 'WFP B':
                                display_name = 'WFP Basement'
                            elif action_name == 'WFP 1st':
                                display_name = 'WFP 1st Floor'
                            
                            if action_name == 'No mitigation':
                                text += f"<b>No Mitigation</b>: {format_currency(val)}<br>"
                            else:
                                savings = baseline_val - val if baseline_val > 0 else 0
                                pct = (savings / baseline_val * 100) if baseline_val > 0 else 0
                                if savings > 0:
                                    text += f"{display_name}: {format_currency(val)} (-{pct:.0f}%)<br>"
                                else:
                                    text += f"{display_name}: {format_currency(val)}<br>"
                        
                        hover_texts.append(text)
                    df_map['hover_text'] = hover_texts
                    # Two-element customdata: [hover_html, building_id]
                    # The id rides along so click events can identify the building.
                    df_map['hover_data'] = [
                        [t, int(i)] for t, i in zip(hover_texts, df_map['id'])
                    ]
                    
                    # ---- Common map center ----
                    center_lat = df_map['latitude'].mean()
                    center_lon = df_map['longitude'].mean()

                    # `_point_scale` is set by the "Point size" slider at the
                    # top of the map tab (see search_col3 above). All map
                    # markers below multiply their literal sizes by this
                    # factor so the user can dial dot size up/down on the fly.
                    # Defaults to 2× for Pamunkey, 1× for other locations.

                    # =====================================================
                    # Helper: add a non-residential "ring" trace beneath
                    # any colored category trace
                    # =====================================================
                    def _add_nonres_ring(fig, df_subset, ring_size=13):
                        nr = df_subset[df_subset['_is_nonres']]
                        if len(nr) > 0:
                            fig.add_trace(go.Scattermapbox(
                                lat=nr['latitude'], lon=nr['longitude'],
                                mode='markers',
                                marker=dict(size=ring_size, color='black', opacity=0.85),
                                hoverinfo='skip',
                                showlegend=False,
                                name='_nonres_ring',
                            ))
                    
                    fig_map = go.Figure()
                    bin_caption_extra = ""  # for the Damage Bins view
                    
                    # =====================================================
                    # VIEW 1 — Damage Heatmap (existing; continuous color)
                    # =====================================================
                    if map_view == "Damage Heatmap":
                        if baseline_col:
                            df_zero = df_map[df_map[baseline_col] == 0]
                            df_nonzero = df_map[df_map[baseline_col] > 0]
                        else:
                            df_zero = pd.DataFrame()
                            df_nonzero = df_map
                        
                        if len(df_zero) > 0:
                            fig_map.add_trace(go.Scattermapbox(
                                lat=df_zero['latitude'], lon=df_zero['longitude'],
                                mode='markers',
                                marker=dict(size=8 * _point_scale, color='#22c55e', opacity=0.85),
                                hovertemplate='%{customdata[0]}<extra></extra>',
                                customdata=list(df_zero['hover_data']),
                                name='No Damage ($0)'
                            ))
                        
                        if len(df_nonzero) > 0 and baseline_col:
                            cb_ticks, cb_labels = smart_money_ticks(color_cap, target_n=5)
                            # If we're clipping at the 90th-percentile cap, mark the
                            # top tick as "≥ X" so the user knows a worse building
                            # would still render at the top red.
                            if cap_is_clipped and len(cb_labels) > 0:
                                cb_labels = list(cb_labels)
                                cb_labels[-1] = f"≥ {cb_labels[-1]}"
                            fig_map.add_trace(go.Scattermapbox(
                                lat=df_nonzero['latitude'], lon=df_nonzero['longitude'],
                                mode='markers',
                                marker=dict(
                                    size=10 * _point_scale,
                                    color=df_nonzero[baseline_col],
                                    colorscale=[
                                        [0,    '#facc15'],
                                        [0.25, '#f59e0b'],
                                        [0.50, '#f97316'],
                                        [0.75, '#ef4444'],
                                        [1.0,  '#b91c1c'],
                                    ],
                                    cmin=0, cmax=color_cap,
                                    colorbar=dict(
                                        title="No Mitigation<br>Cumulative Damage",
                                        tickmode='array',
                                        tickvals=cb_ticks, ticktext=cb_labels,
                                        len=0.7, y=0.5,
                                    ),
                                    opacity=0.85
                                ),
                                hovertemplate='%{customdata[0]}<extra></extra>',
                                customdata=list(df_nonzero['hover_data']),
                                name='At Risk'
                            ))
                    
                    # =====================================================
                    # VIEW 2 — Adaptation Effectiveness (4 categories)
                    # Ported from generate_action_animation.m
                    # Uses upper-tail (P95) cumulative damage as a proxy
                    # for the MATLAB script's P90.
                    # =====================================================
                    elif map_view == "Adaptation Effectiveness":
                        # Required columns — P90 is the upper-tail proxy
                        # (matches the Distributions tab and the workshop
                        # convention).
                        #
                        # Per-location action swap: for Pamunkey, WFP Basement
                        # is non-applicable (RES2 / pier inventory, no
                        # basements — the applicability filter in load_bundle
                        # drops those rows entirely). The cheapest retrofit
                        # that actually eliminates upper-tail damage there is
                        # Raise Utilities, so we plug it into the "cheapest
                        # retrofit that eliminates damage" slot of the
                        # classifier. The internal variable names below keep
                        # the `wfpb*` spelling for code economy — they hold
                        # whichever retrofit is playing that role for the
                        # current location.
                        col_nomit = 'No mitigation_P90'
                        if location_name == "Pamunkey":
                            col_wfpb = 'Raise Utilities_P90'
                            cheap_retrofit_label = 'Raise Utilities'
                        else:
                            col_wfpb = 'WFP B_P90'
                            cheap_retrofit_label = 'WFP Basement'
                        col_elev  = 'Elevate_P90'
                        
                        missing = [c for c in (col_nomit, col_wfpb, col_elev)
                                   if c not in df_map.columns]
                        if missing:
                            st.warning(
                                f"This view needs P90 columns for No mitigation, "
                                f"{cheap_retrofit_label}, and Elevate. "
                                f"Missing: {', '.join(missing)}"
                            )
                        else:
                            thr = ZERO_THRESH_DISPLAY  # treat damages below $1k as zero
                            
                            # Preserve NaN so missing retrofit values don't silently
                            # count as effective (fillna(0) would do that).
                            no_mit_raw = df_map[col_nomit].values.astype(float)
                            wfpb_raw   = df_map[col_wfpb].values.astype(float)
                            elev_raw   = df_map[col_elev].values.astype(float)
                            
                            no_mit = np.where(np.isnan(no_mit_raw), 0.0, no_mit_raw)
                            # NaN in a retrofit column ==> "retrofit not applied" ==> baseline
                            wfpb   = np.where(np.isnan(wfpb_raw), no_mit, wfpb_raw)
                            elev   = np.where(np.isnan(elev_raw), no_mit, elev_raw)
                            
                            # --- MAP classifier (threshold rule) ---
                            # Five buckets ordered by the cheapest adaptation
                            # that brings upper-tail damage below the $1k
                            # "no damage" threshold. Reading the colors top
                            # to bottom gives a decision pyramid: do nothing →
                            # cheap retrofit → expensive retrofit (elevation) →
                            # nothing works.
                            #
                            #   1 = No Damage     baseline P90 ≤ thr
                            #   2 = Cheap fix     baseline > thr AND the
                            #                     cheap retrofit (WFP Basement
                            #                     elsewhere, Raise Utilities
                            #                     in Pamunkey) brings P90 ≤ thr
                            #   3 = Elevation     baseline > thr, cheap retrofit
                            #                     doesn't reach thr, but elevation
                            #                     DOES bring P90 ≤ thr
                            #   4 = Residual      Under-DFE, damaged, neither
                            #                     retrofit (cheap nor elevation)
                            #                     brings P90 ≤ thr — even the
                            #                     strongest adaptation in scope
                            #                     leaves residual damage, so the
                            #                     conversation has to move to
                            #                     buyout / relocation / managed
                            #                     retreat. Drawn in red.
                            #   5 = Out of scope  Above-DFE buildings that fall
                            #                     through (their retrofit options
                            #                     are different from those in the
                            #                     three-bucket pyramid above —
                            #                     e.g., WFP 1st Floor, dry
                            #                     floodproofing, content-only).
                            #                     Not plotted on this view, since
                            #                     coloring them red would
                            #                     misrepresent "the menu shown
                            #                     here doesn't cover this case"
                            #                     as "this building is doomed".
                            #
                            # NB: the earlier version used a *dominance* rule
                            # for cat=3 (Elevate < WFP B, even if neither hit
                            # thr). That was retired when the Residual bucket
                            # came back: keeping dominance would have meant
                            # damaged Under-DFE buildings where elevation
                            # merely beat WFP B (but still left tens of thousands
                            # of dollars of damage) got colored orange
                            # "Elevation works", contradicting the new red
                            # "even elevation leaves residual" bucket. The
                            # threshold rule makes the orange/red split honest:
                            # orange means elevation eliminates damage, red
                            # means it doesn't.
                            wfpb_to_thr = wfpb <= thr
                            elev_to_thr = elev <= thr

                            # Under-DFE membership mask (Above-DFE buildings
                            # are NEVER classified as Residual — they fall
                            # through to cat=5 / omitted instead).
                            if 'DFE_Status' in df_map.columns:
                                _dfe_lower = (df_map['DFE_Status']
                                              .fillna('').astype(str)
                                              .str.strip().str.lower())
                                is_under_dfe = _dfe_lower.str.contains('under').values
                            else:
                                is_under_dfe = np.zeros(len(df_map), dtype=bool)

                            # Priority classification. Default = 5 (out of
                            # scope / omitted) so any building not affirmatively
                            # placed in 1–4 drops off the map quietly.
                            cat = np.full(len(df_map), 5, dtype=int)
                            cat[no_mit <= thr] = 1
                            cat[(no_mit > thr) & wfpb_to_thr] = 2
                            cat[(no_mit > thr) & ~wfpb_to_thr & elev_to_thr] = 3
                            cat[(no_mit > thr) & ~wfpb_to_thr & ~elev_to_thr & is_under_dfe] = 4

                            df_map['_cat_action'] = cat

                            # Legend counts — each building appears in exactly
                            # one bucket, so these add up to (buildings shown).
                            n_no_damage  = int((cat == 1).sum())
                            n_wfpb_works = int((cat == 2).sum())
                            n_elev_works = int((cat == 3).sum())
                            n_residual   = int((cat == 4).sum())
                            cat_legend_counts = {
                                1: n_no_damage,
                                2: n_wfpb_works,
                                3: n_elev_works,
                                4: n_residual,
                            }

                            # Workshop palette + red residual.
                            # cat=5 (Above-DFE fall-through) is intentionally
                            # absent from cat_specs and therefore not plotted.
                            cat_specs = [
                                (1, 'No Damage',            '#22c55e'),  # green
                                (2, cheap_retrofit_label,   '#facc15'),  # yellow
                                (3, 'Elevation',            '#f97316'),  # orange
                                (4, 'Residual Damage',      '#dc2626'),  # red
                            ]
                            
                            for ci, label, color in cat_specs:
                                df_c = df_map[df_map['_cat_action'] == ci]
                                # Legend label uses the INDEPENDENT count
                                # (e.g. Elevation includes buildings where WFP-B
                                # also works); the colored markers on the map
                                # still follow priority order, so each building
                                # appears in only one color.
                                legend_count = cat_legend_counts.get(ci, len(df_c))
                                if len(df_c) == 0:
                                    # No markers in this priority bucket — still
                                    # show a legend stub with the independent count.
                                    fig_map.add_trace(go.Scattermapbox(
                                        lat=[None], lon=[None],
                                        mode='markers',
                                        marker=dict(size=8 * _point_scale, color=color, opacity=0.92),
                                        name=f"{label} ({legend_count})",
                                        showlegend=True, hoverinfo='skip',
                                    ))
                                    continue
                                _add_nonres_ring(fig_map, df_c, ring_size=13 * _point_scale)
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=df_c['latitude'], lon=df_c['longitude'],
                                    mode='markers',
                                    marker=dict(size=8 * _point_scale, color=color, opacity=0.92),
                                    hovertemplate='%{customdata[0]}<extra></extra>',
                                    customdata=list(df_c['hover_data']),
                                    name=f"{label} ({legend_count})",
                                ))
                            
                            # Add a legend-only trace for the non-residential ring marker
                            if df_map['_is_nonres'].any():
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=[None], lon=[None],
                                    mode='markers',
                                    marker=dict(size=10 * _point_scale, color='black', opacity=0.85),
                                    name='Non-Residential (ringed)',
                                    showlegend=True, hoverinfo='skip',
                                ))
                    
                    # =====================================================
                    # VIEW 3 — Damage Bins (5 categories with dynamic breaks)
                    # Ported from generate_damage_animation_v3.m
                    # =====================================================
                    elif map_view == "Damage Bins":
                        # P90 is the upper-tail proxy here, matching
                        # compute_damage_bin_breaks() above and the
                        # Distributions tab's classifier.
                        col_nomit = 'No mitigation_P90'
                        if col_nomit not in df_map.columns:
                            st.warning(f"This view needs the '{col_nomit}' column.")
                        else:
                            dmg = df_map[col_nomit].fillna(0).values
                            thr = ZERO_THRESH_DISPLAY
                            
                            # Separate "no damage" (<= $1k) from damaged buildings.
                            no_damage_mask = dmg <= thr
                            df_no_dmg = df_map[no_damage_mask]
                            n_no_dmg  = len(df_no_dmg)
                            
                            nonzero = dmg[~no_damage_mask]
                            
                            # ---- Plot the green "No Damage" layer first, so the
                            # damaged-building markers draw on top of it. ----
                            if n_no_dmg > 0:
                                _add_nonres_ring(fig_map, df_no_dmg, ring_size=13 * _point_scale)
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=df_no_dmg['latitude'], lon=df_no_dmg['longitude'],
                                    mode='markers',
                                    marker=dict(size=8 * _point_scale, color='#22c55e', opacity=0.85),
                                    hovertemplate='%{customdata[0]}<extra></extra>',
                                    customdata=list(df_no_dmg['hover_data']),
                                    name=f"No Damage ($0)  ({n_no_dmg})",
                                ))
                            
                            if len(nonzero) == 0:
                                if n_no_dmg == 0:
                                    st.info("No buildings match the current filters.")
                                bin_caption_extra = ""
                            else:
                                # ---- Stable bin breakpoints across all years ----
                                # Computed once from the pooled distribution of
                                # nonzero No-Mit P90 damages over every year for
                                # the selected SLR scenario, so the same building
                                # gets the same bin color regardless of which
                                # year is active. Bins recompute when the user
                                # switches SLR scenario (because the underlying
                                # damage distribution shifts).
                                breaks = compute_damage_bin_breaks(df_buildings, scenario)
                                if not breaks:
                                    # Degenerate case (no nonzero damages anywhere
                                    # in this SLR scenario) — fall back to the
                                    # current year's data
                                    raw_breaks = np.quantile(nonzero, [0.20, 0.40, 0.60, 0.80])
                                    breaks = sorted({nice_round_up(v) for v in raw_breaks if v > 0})[:4]
                                # bin edges for damaged buildings: thr, b1, …, bk, +inf
                                # where k = len(breaks) ∈ [1..4], giving 2..5 bins.
                                # The bin count is data-driven: locations with
                                # tightly-clustered damages collapse adjacent
                                # quantile breaks during nice-number rounding, and
                                # we now ACCEPT the lower bin count rather than
                                # extrapolating breaks beyond the actual data
                                # range (which used to push typical buildings
                                # into a misleadingly-red bin at low-damage
                                # locations like Pamunkey).
                                edges = [float(thr)] + breaks + [np.inf]
                                n_bins = len(edges) - 1
                                
                                # MATLAB workshop palette (5 bins for damaged buildings)
                                bin_palette = [
                                    '#f0d200',   # yellow
                                    '#d25f00',   # dark orange
                                    '#dc2626',   # red
                                    '#8b0000',   # dark red
                                    '#6a1b9a',   # purple
                                ]
                                bin_colors = bin_palette[:n_bins]
                                
                                def _bin_label(lo, hi):
                                    if hi == np.inf:
                                        return f"> {fmt_money_short(lo)}"
                                    # Lower-edge label: "≤ hi" for the first bin (lo is $1k threshold)
                                    if lo == float(thr):
                                        return f"≤ {fmt_money_short(hi)}"
                                    return f"{fmt_money_short(lo)} – {fmt_money_short(hi)}"
                                
                                bin_labels = [_bin_label(edges[i], edges[i+1]) for i in range(n_bins)]
                                
                                # Classify damaged buildings into the 5 bins. np.digitize
                                # returns 0..n-1 where bin 0 = (thr, b1], bin n-1 = (b4, inf).
                                damaged_mask = ~no_damage_mask
                                df_damaged   = df_map[damaged_mask].copy()
                                dmg_damaged  = dmg[damaged_mask]
                                bin_idx = np.digitize(dmg_damaged, edges[1:-1], right=False)
                                df_damaged['_bin_idx'] = bin_idx
                                
                                for ci in range(n_bins):
                                    df_c = df_damaged[df_damaged['_bin_idx'] == ci]
                                    if len(df_c) == 0:
                                        continue
                                    count = len(df_c)
                                    _add_nonres_ring(fig_map, df_c, ring_size=13 * _point_scale)
                                    fig_map.add_trace(go.Scattermapbox(
                                        lat=df_c['latitude'], lon=df_c['longitude'],
                                        mode='markers',
                                        marker=dict(size=8 * _point_scale, color=bin_colors[ci], opacity=0.92),
                                        hovertemplate='%{customdata[0]}<extra></extra>',
                                        customdata=list(df_c['hover_data']),
                                        name=f"{bin_labels[ci]} ({count})",
                                    ))
                                
                                bin_caption_extra = (
                                    "Damaged buildings are binned at the 20/40/60/80th percentiles "
                                    "of pooled nonzero damages **across all years for the selected SLR scenario**, "
                                    "snapped to nice rounded values "
                                    f"({', '.join(fmt_money_short(b) for b in breaks)}). "
                                    "Bins stay constant when you change the year (so the same building "
                                    "keeps the same color across 2040 / 2055 / 2060 / 2100), but recompute when "
                                    "you switch SLR scenario. Buildings with no damage (≤ $1k) are shown in green."
                                )
                            
                            if df_map['_is_nonres'].any():
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=[None], lon=[None],
                                    mode='markers',
                                    marker=dict(size=10 * _point_scale, color='black', opacity=0.85),
                                    name='Non-Residential (ringed)',
                                    showlegend=True, hoverinfo='skip',
                                ))
                    
                    # ---- Search highlight (auto-expires) ----
                    # If the user just searched for a Building ID and the
                    # highlight is still fresh, drop a magenta ring on top
                    # of all other traces. Two layered markers give a clear
                    # visual flag: an outer translucent halo + an inner
                    # solid ring. Both vanish on the next rerun once the
                    # TTL has elapsed.
                    #
                    # Streamlit doesn't rerun on a wall-clock timer, so we
                    # schedule a one-shot rerun right after the TTL using
                    # st_autorefresh — without it, the ring would persist
                    # on screen until the user moved the mouse / resized
                    # the page, which defeats the "clean screenshot"
                    # goal.
                    _hl_id    = st.session_state.get('map_highlight_id')
                    _hl_until = float(st.session_state.get('map_highlight_until', 0.0))
                    _hl_remaining = _hl_until - _time_mod.time()
                    if _hl_id is not None and _hl_remaining > 0:
                        # Look up the building's coordinates in the same
                        # df_map the rest of the figure was built from, so
                        # the ring lands on the same dot the user sees.
                        _hl_row = df_map[df_map['id'] == _hl_id]
                        if not _hl_row.empty:
                            _hl_lat = float(_hl_row['latitude'].iloc[0])
                            _hl_lon = float(_hl_row['longitude'].iloc[0])
                            _addr = _hl_row.get('address', pd.Series([None])).iloc[0]
                            _hl_label = (
                                f"Building #{_hl_id}"
                                + (f" — {_addr}" if pd.notna(_addr) and str(_addr).strip() else "")
                            )
                            # Outer halo — large translucent magenta circle
                            fig_map.add_trace(go.Scattermapbox(
                                lat=[_hl_lat], lon=[_hl_lon],
                                mode='markers',
                                marker=dict(size=44 * _point_scale, color='rgba(217, 70, 239, 0.32)'),
                                hoverinfo='skip',
                                showlegend=False,
                                name='_search_halo',
                            ))
                            # Inner solid ring
                            fig_map.add_trace(go.Scattermapbox(
                                lat=[_hl_lat], lon=[_hl_lon],
                                mode='markers',
                                marker=dict(size=22 * _point_scale, color='rgba(217, 70, 239, 0.95)'),
                                hovertemplate=f'<b>{_hl_label}</b><extra></extra>',
                                showlegend=True,
                                name=f'🔍 {_hl_label}',
                            ))
                            # Schedule a single rerun ~200ms after the TTL
                            # so the ring auto-disappears without manual
                            # interaction. Falls back gracefully if the
                            # st_autorefresh package isn't installed —
                            # the ring still vanishes on the next user
                            # interaction (pan, zoom, click, etc.).
                            try:
                                from streamlit_autorefresh import st_autorefresh
                                st_autorefresh(
                                    interval=int(_hl_remaining * 1000) + 200,
                                    limit=1,
                                    key=f"map_hl_expire_{_hl_id}_{int(_hl_until)}",
                                )
                            except ImportError:
                                # Optional dependency — silently skip the
                                # auto-clear. The user can hit "✖ Clear"
                                # or trigger any UI event to clear it.
                                pass
                    
                    # ---- Common layout ----
                    # Resolve the user's basemap choice (Streets / Aerial) into
                    # the (style, layers) pair plotly needs. Aerial is a
                    # white-bg style with an ESRI raster tile layer drawn
                    # under the data traces; Streets is plain OSM.
                    _bm_style, _bm_layers = _basemap_config(basemap_choice)
                    fig_map.update_layout(
                        mapbox=dict(
                            style=_bm_style,
                            layers=_bm_layers,
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=12
                        ),
                        margin={"r":0,"t":0,"l":0,"b":0},
                        height=600,
                        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                    bgcolor="rgba(255,255,255,0.85)",
                                    bordercolor="rgba(0,0,0,0.15)", borderwidth=1),
                    )
                    
                    # Click capture: tag the chart with a stable key + on_select="rerun"
                    # so clicks bubble back to Python. selection_mode="points" limits
                    # the interaction to simple clicks (no lasso/box select), which also
                    # prevents Plotly from dimming unselected markers.
                    # The clicked building's id rides along in customdata[1] (set when
                    # df_map['hover_data'] was built) — we extract it below.
                    map_event = st.plotly_chart(
                        fig_map,
                        use_container_width=True,
                        key="bldg_map",
                        on_select="rerun",
                        selection_mode="points",
                    )
                    
                    # --- Extract clicked building id (defensive against several
                    #     return-value shapes Streamlit can emit). If the event
                    #     comes back with an empty points array, treat that as a
                    #     deselection (user clicked an empty area of the map).
                    clicked_id = None
                    map_selection_seen = False   # True iff event payload reached us
                    pts_nonempty = False
                    try:
                        sel = None
                        if map_event is not None:
                            sel = getattr(map_event, 'selection', None) or \
                                  (map_event.get('selection') if isinstance(map_event, dict) else None)
                        pts = None
                        if sel is not None:
                            pts = getattr(sel, 'points', None)
                            if pts is None and isinstance(sel, dict):
                                pts = sel.get('points')
                            # `pts` may be None (no event), [] (empty click), or non-empty.
                            map_selection_seen = pts is not None
                            pts_nonempty = bool(pts)
                        if pts_nonempty:
                            p0 = pts[0]
                            cd = p0.get('customdata') if isinstance(p0, dict) else getattr(p0, 'customdata', None)
                            # cd may be [text, id]  OR  [[text, id]] (nested) — unwrap as needed
                            if isinstance(cd, (list, tuple)):
                                if len(cd) >= 2 and not isinstance(cd[1], (list, tuple)):
                                    clicked_id = int(cd[1])
                                elif len(cd) >= 1 and isinstance(cd[0], (list, tuple)) and len(cd[0]) >= 2:
                                    clicked_id = int(cd[0][1])
                    except (TypeError, ValueError, AttributeError, IndexError):
                        clicked_id = None
                    
                    if clicked_id is not None:
                        st.session_state['selected_building_id'] = clicked_id
                    elif map_selection_seen and not pts_nonempty:
                        # User clicked an empty area of the map — clear the
                        # selection everywhere (map caption, Building Details tab).
                        # NOTE: Plotly on Scattermapbox doesn't always emit an
                        # event for empty-area clicks, so this path is a
                        # best-effort fallback. The Clear button below is the
                        # reliable way to deselect.
                        st.session_state.pop('selected_building_id', None)
                    
                    if 'selected_building_id' in st.session_state:
                        col_sel, col_btn = st.columns([4, 1])
                        with col_sel:
                            st.caption(
                                f"📍 **Selected building:** #{st.session_state['selected_building_id']} "
                                "— open the **Details** tab to see its full profile."
                            )
                        with col_btn:
                            if st.button("✖ Clear selection", key="clear_map_selection",
                                         use_container_width=True):
                                st.session_state.pop('selected_building_id', None)
                                # Also clear the widget-state key used by the
                                # Building Details selectbox so its value
                                # resets on the next render.
                                st.session_state.pop('bldg_details_selectbox', None)
                                st.session_state.pop('bldg_search_input', None)
                                st.rerun()
                    
                    # ---- View-specific caption ----
                    if map_view == "Damage Heatmap":
                        cap_note = ""
                        if cap_is_clipped:
                            cap_note = (
                                f" The color scale is capped at the 90th percentile of nonzero "
                                f"damages (about {fmt_money_short(color_cap)}) so the gradient is "
                                f"readable across the bulk of buildings. Buildings above the cap "
                                f"(max in this view: {fmt_money_short(max_damage_raw)}) still render "
                                f"at the top red and show their true damage in the hover tooltip."
                            )
                        st.caption(
                            "Each at-risk building is colored by its No-Mitigation P50 cumulative "
                            "damage." + cap_note
                        )
                    elif map_view == "Adaptation Effectiveness":
                        # Caption uses the same per-location action swap as
                        # the classifier above: for Pamunkey, "Raise Utilities"
                        # plays the role of "WFP Basement" everywhere else.
                        if location_name == "Pamunkey":
                            _cheap_lbl = "Raise Utilities"
                            _cheap_desc = "raising at-risk utilities"
                        else:
                            _cheap_lbl = "WFP Basement"
                            _cheap_desc = "basement floodproofing"
                        st.caption(
                            "Each building is colored by the **cheapest adaptation that "
                            "eliminates** its upper-tail (P90) cumulative damage under "
                            "the selected year and SLR scenario. Buckets are checked in "
                            "priority order: "
                            "**No Damage** (baseline P90 ≤ $1k — no intervention needed) → "
                            f"**{_cheap_lbl}** ({_cheap_desc} brings P90 ≤ $1k) → "
                            f"**Elevation** ({_cheap_lbl} doesn't reach the threshold but "
                            "elevation does) → "
                            "**Residual Damage** (Under-DFE buildings where neither "
                            f"{_cheap_lbl.lower()} nor elevation can bring P90 ≤ $1k — "
                            "the conversation has to move beyond retrofits, to buyout, "
                            "relocation, or larger community-scale interventions). "
                            "Above-DFE buildings whose damage isn't eliminated by the "
                            f"two retrofits shown ({_cheap_lbl} or elevation) are not "
                            "plotted on this view — their relevant adaptation options "
                            "(wet floodproofing the first floor, content-only measures, "
                            "etc.) aren't represented in the three-bucket pyramid above. "
                            "Each building appears in exactly one color, and the legend "
                            "counts partition the buildings shown. Non-residential "
                            "buildings are marked with a black ring."
                        )
                    elif map_view == "Damage Bins":
                        st.caption(
                            "Each building is colored by its No-Mitigation P90 cumulative damage. " +
                            bin_caption_extra +
                            " Non-residential buildings are marked with a black ring."
                        )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Buildings Shown", f"{len(df_map):,}")
                    with col2:
                        total_baseline = df_map[baseline_col].sum() if baseline_col else 0
                        st.metric("Total No Mitigation Cumulative Damage", format_currency(total_baseline))
                    
                    # ========================================================
                    # HIGH-RESOLUTION MAP EXPORT
                    # ========================================================
                    # Generates a publication-quality PNG/PDF/SVG of the
                    # current map view: title, scenario subtitle, scale bar
                    # (geographically correct at the chosen zoom), north
                    # arrow, and credits footer with data sources.
                    # Output is sized for ~300 DPI print at the chosen
                    # paper aspect (8×6 in by default).
                    with st.expander("🖨️ Export Map (High-Resolution, Publication-Ready)", expanded=False):
                        st.markdown(
                            "Generate a high-resolution image of the current map view, with a "
                            "title, scale bar, north arrow, and full credits footer (data sources, "
                            "basemap attribution, and date). Formats below are sized for a journal "
                            "figure at ~300 DPI print quality."
                        )
                        exp_col1, exp_col2, exp_col3 = st.columns([1.1, 1.1, 1])
                        with exp_col1:
                            export_format = st.selectbox(
                                "Format",
                                options=["PNG (raster)", "PDF (vector)", "SVG (vector)"],
                                index=0,
                                key="map_export_format",
                                help="PNG: best for slides and web. PDF/SVG: best for "
                                     "journal submission since they remain crisp at any zoom."
                            )
                        with exp_col2:
                            export_size = st.selectbox(
                                "Size",
                                options=[
                                    "Single column (1600×1200)",
                                    "1.5 column (2000×1500)",
                                    "Double column (2400×1800)",
                                    "Full page (3200×2400)",
                                ],
                                index=2,
                                key="map_export_size",
                                help="Approximate journal figure widths. Larger = sharper "
                                     "print at the cost of file size."
                            )
                        with exp_col3:
                            export_scale = st.selectbox(
                                "DPI",
                                options=[1, 2, 3, 4],
                                index=1,
                                key="map_export_scale",
                                format_func=lambda x: f"{x}× (~{x*150} DPI)",
                                help="Multiplier on rendered pixel density. 2× ≈ 300 DPI; "
                                     "3× ≈ 450 DPI. SVG/PDF ignore this (always vector)."
                            )
                        # Second row: basemap choice. We deliberately omit
                        # "open-street-map" here because OSM's tile servers
                        # block headless render requests (kaleido's bundled
                        # browser doesn't send the Referer header OSM's tile
                        # usage policy requires), so an OSM export consistently
                        # comes back peppered with "Access blocked" tiles.
                        # The live map can still use OSM — your real browser
                        # supplies the Referer there. For exports use Carto
                        # (which has no such restriction) or the white-bg
                        # option.
                        export_basemap = st.selectbox(
                            "Basemap (printed export only)",
                            options=[
                                "carto-positron",
                                "carto-darkmatter",
                                "Aerial",
                                "white-bg (no basemap)",
                            ],
                            index=0,   # carto-positron is the cleanest print default
                            key="map_export_basemap",
                            help="Tile-based basemaps need internet access at export time; "
                                 "if a tile server refuses the request the export will "
                                 "auto-fall-back to a white background. `carto-positron` "
                                 "(light, neutral) and `carto-darkmatter` (dark) are the "
                                 "preferred print styles. `Aerial` uses ESRI World "
                                 "Imagery satellite tiles — great for showing physical "
                                 "context (shorelines, vegetation, building footprints) "
                                 "but heavier and lower-contrast in print. Pick "
                                 "`white-bg` for a guaranteed-working export with no "
                                 "roads/labels. OpenStreetMap is intentionally excluded "
                                 "— its tile servers block headless render requests, so "
                                 "OSM exports come back covered in 'Access blocked' tiles."
                        )
                        
                        # Third row: framing override.
                        # Default behavior is auto-fit-to-data; "Custom" lets
                        # the user dial in center/zoom to match their live
                        # map view (Streamlit doesn't surface Plotly pan/zoom
                        # state back to Python, so a manual override is the
                        # most reliable WYSIWYG path).
                        st.markdown("**Map framing**")
                        # Bbox-fit defaults so the user can tweak from there
                        # rather than starting from scratch.
                        _df_lat = df_map['latitude'].dropna()
                        _df_lon = df_map['longitude'].dropna()
                        _bbox_lat = float(_df_lat.mean()) if len(_df_lat) else 40.86
                        _bbox_lon = float(_df_lon.mean()) if len(_df_lon) else -72.49
                        frame_mode = st.radio(
                            "Frame",
                            options=["Auto-fit data (default)", "Custom center & zoom"],
                            index=0, horizontal=True,
                            key="map_export_frame_mode",
                            help="Auto-fit centers on the data bbox and chooses a zoom "
                                 "that shows everything. Custom lets you set the center "
                                 "and zoom — useful when you've panned/zoomed in the live "
                                 "map and want the export to match. (Streamlit doesn't "
                                 "expose the live map's current pan/zoom back to Python, "
                                 "so this is the most reliable way to get WYSIWYG.)"
                        )
                        custom_center_lat = None
                        custom_center_lon = None
                        custom_zoom = None
                        if frame_mode == "Custom center & zoom":
                            fc1, fc2, fc3 = st.columns(3)
                            with fc1:
                                custom_center_lat = st.number_input(
                                    "Center latitude (°N)",
                                    value=_bbox_lat, format="%.4f", step=0.001,
                                    key="map_export_center_lat",
                                )
                            with fc2:
                                custom_center_lon = st.number_input(
                                    "Center longitude (°E)",
                                    value=_bbox_lon, format="%.4f", step=0.001,
                                    key="map_export_center_lon",
                                )
                            with fc3:
                                custom_zoom = st.number_input(
                                    "Zoom level",
                                    min_value=1.0, max_value=20.0,
                                    value=13.0, step=0.5,
                                    key="map_export_zoom",
                                    help="Web Mercator zoom: 11≈town, 13≈neighborhood, "
                                         "15≈street level. Increase to zoom in.",
                                )
                            st.caption(
                                "💡 *To match your current map view: hover the live map "
                                "to read coordinates near the center, copy them here, and "
                                "pick a zoom that frames the same area. Generate, inspect, "
                                "adjust if needed.*"
                            )
                        
                        # Parse selections
                        _fmt_map = {"PNG (raster)": "png", "PDF (vector)": "pdf", "SVG (vector)": "svg"}
                        _size_map = {
                            "Single column (1600×1200)":   (1600, 1200),
                            "1.5 column (2000×1500)":      (2000, 1500),
                            "Double column (2400×1800)":   (2400, 1800),
                            "Full page (3200×2400)":       (3200, 2400),
                        }
                        fmt = _fmt_map[export_format]
                        ex_w, ex_h = _size_map[export_size]
                        # Strip the trailing description for white-bg
                        export_basemap_value = export_basemap.split(' ')[0]
                        
                        if st.button("📤 Generate Export", key="generate_map_export",
                                     type="primary", use_container_width=False):
                            with st.spinner("Rendering high-resolution map…"):
                                _scenario_label = ('Median SLR (P50)' if scenario == '50th-percentile'
                                                   else 'High-End SLR (P90)' if scenario == '90th-percentile'
                                                   else scenario)
                                pub_fig = build_publication_map_figure(
                                    fig_map,
                                    location=location_name,
                                    occupancy=occupancy_label,
                                    target_year=int(target_year),
                                    scenario_label=_scenario_label,
                                    map_view=map_view,
                                    df_map=df_map,
                                    width_px=ex_w, height_px=ex_h,
                                    mapbox_style=export_basemap_value,
                                    center_lat_override=custom_center_lat,
                                    center_lon_override=custom_center_lon,
                                    zoom_override=custom_zoom,
                                )
                                img_bytes, mime, ext, info = export_map_image(
                                    pub_fig, fmt=fmt, scale=int(export_scale),
                                )
                            if img_bytes is None:
                                # Hard failure (kaleido missing, or fallback also failed)
                                st.error(f"❌ Export failed: {info}")
                                st.info(
                                    "If you're running this app yourself, install the export "
                                    "engine with `pip install -U kaleido` and run "
                                    "`plotly_get_chrome` once to set up the headless browser."
                                )
                            else:
                                # Filename: location_year_scenario_view.ext (sanitized)
                                _safe = lambda s: ''.join(
                                    c if c.isalnum() or c in '-_' else '_' for c in str(s)
                                )
                                _scn_short = ('P50' if scenario == '50th-percentile'
                                              else 'P90' if scenario == '90th-percentile'
                                              else _safe(scenario))
                                _occ_short = (_safe(occupancy_label).replace('Buildings','').strip('_')
                                              or 'All')
                                fname = (f"ADAPT_{_safe(location_name)}_{int(target_year)}_"
                                         f"{_scn_short}_{_safe(map_view)}_{_occ_short}.{ext}")
                                st.success(
                                    f"✅ Map ready — {ex_w}×{ex_h} px"
                                    f"{' (×' + str(int(export_scale)) + ')' if fmt == 'png' else ''}, "
                                    f"{len(img_bytes)/1024:.0f} KB"
                                )
                                if info == "basemap_fallback":
                                    st.warning(
                                        f"⚠ The chosen basemap (`{export_basemap_value}`) "
                                        "couldn't be rendered — its tile server refused the "
                                        "request from the headless browser. The export was "
                                        "generated on a clean white background instead. "
                                        "Try `carto-positron` or `white-bg` for a more "
                                        "reliable export, or supply a Mapbox API token if "
                                        "you need branded tiles."
                                    )
                                st.download_button(
                                    label=f"⬇ Download {fmt.upper()}",
                                    data=img_bytes,
                                    file_name=fname,
                                    mime=mime,
                                    key="download_map_export",
                                    use_container_width=False,
                                )
                                st.caption(
                                    "Includes: title, scenario subtitle, geographically-correct "
                                    "scale bar, north arrow, and credits footer (data sources, "
                                    "basemap attribution, generation date). Cite as: "
                                    "*Generated with ADAPT, Center for Climate Systems Research, "
                                    "The Climate School, Columbia University.*"
                                )
                    
                    st.subheader(f"🔴 Top 10 Highest Risk Buildings (No Mitigation)")
                    
                    display_cols = ['id']
                    if 'occupancy_type' in df_map.columns:
                        display_cols.append('occupancy_type')
                    if 'structure_value' in df_map.columns:
                        display_cols.append('structure_value')
                    if 'DFE_Status' in df_map.columns:
                        display_cols.append('DFE_Status')
                    display_cols.extend(action_cols_p50)
                    
                    if baseline_col:
                        top10 = df_map.nlargest(10, baseline_col)[display_cols].copy()
                    else:
                        top10 = df_map.head(10)[display_cols].copy()
                    
                    if 'structure_value' in top10.columns:
                        top10['structure_value'] = top10['structure_value'].apply(format_currency)
                    for col in action_cols_p50:
                        top10[col] = top10[col].apply(format_currency)
                    
                    rename_map = {col: col.replace('_P50', '') for col in action_cols_p50}
                    rename_map['DFE_Status'] = 'DFE Status'
                    top10 = top10.rename(columns=rename_map)
                    
                    st.dataframe(top10, use_container_width=True, hide_index=True)
        else:
            st.warning("No per-building data available for this location.")
        
        # --------------------------------------------------------------
        # Data Notes — provenance / coverage information for the bundle
        # --------------------------------------------------------------
        # Surfaced below the map so users have a one-click path to
        # check what's in the inventory, what got skipped, and which
        # hazard inputs the analysis used. The expander stays collapsed
        # by default to keep the main view clean.
        if loc_entry is not None:
            with st.expander("ℹ️ Data Notes — coverage, exclusions, and bundle metadata", expanded=False):
                meta = loc_entry.get('metadata') or {}
                bfe_ft_local = loc_entry.get('bfe_ft')
                
                # Coverage line
                bldg_attrs = loc_entry.get('bldg_attrs')
                n_total = int(len(bldg_attrs)) if bldg_attrs is not None else None
                skipped_df = loc_entry.get('skipped')
                n_skipped = len(skipped_df) if isinstance(skipped_df, pd.DataFrame) else 0
                
                cov_col1, cov_col2, cov_col3 = st.columns(3)
                with cov_col1:
                    if n_total is not None:
                        st.markdown(f"**Buildings analyzed**  \n{n_total:,}")
                with cov_col2:
                    if bfe_ft_local is not None:
                        st.markdown(f"**Base Flood Elevation (BFE)**  \n{bfe_ft_local:g} ft NAVD88")
                        st.caption(f"DFE = BFE + 2 = {bfe_ft_local + 2:g} ft NAVD88")
                with cov_col3:
                    target_year_labels_local = loc_entry.get('target_year_labels') or {}
                    if target_year_labels_local:
                        years_str = ", ".join(
                            f"{lab} ({y})" if lab != str(y) else str(y)
                            for y, lab in sorted(target_year_labels_local.items())
                        )
                        st.markdown(f"**Evaluation horizons**  \n{years_str}")
                
                # Skipped buildings table — only shown when there are any.
                if n_skipped > 0:
                    st.markdown("---")
                    st.markdown(f"**Excluded buildings ({n_skipped})**")
                    st.caption(
                        "These buildings appear in the National Structure Inventory for "
                        "this location but were excluded from the damage analysis "
                        "(typically due to invalid or incomplete attributes)."
                    )
                    # Show the relevant columns; the CSV ships
                    # BuildingID / NSI_row / OccupancyType / SOID / Reason
                    show_cols = [c for c in ['BuildingID', 'OccupancyType', 'Reason']
                                 if c in skipped_df.columns]
                    st.dataframe(
                        skipped_df[show_cols] if show_cols else skipped_df,
                        use_container_width=True, hide_index=True,
                    )
                
                # Bundle metadata — small, gray, for the technically curious.
                if meta:
                    st.markdown("---")
                    st.caption("**Bundle metadata**")
                    # Format key fields nicely; dump the rest as a list.
                    pretty_lines = []
                    if 'LOCATION' in meta:
                        pretty_lines.append(f"Location: **{meta['LOCATION']}**")
                    if 'n_res' in meta and 'n_nonres' in meta:
                        pretty_lines.append(
                            f"Residential: **{meta['n_res']}** · "
                            f"Non-Residential: **{meta['n_nonres']}**"
                        )
                    actions_meta = meta.get('ACTION_NAMES')
                    if isinstance(actions_meta, list):
                        pretty_lines.append("Adaptation actions: " + ", ".join(f"`{a}`" for a in actions_meta))
                    pcts_meta = meta.get('PERCENTILES')
                    if isinstance(pcts_meta, list):
                        pretty_lines.append(
                            f"Percentiles available: **{len(pcts_meta)}** "
                            f"({pcts_meta[0]}–{pcts_meta[-1]}, dense at tails plus quartiles)"
                        )
                    # Water-level MC realizations
                    wl = loc_entry.get('water_levels') or {}
                    mc_keys = [k for k in wl.keys() if k.endswith('_mc')]
                    if mc_keys:
                        # Use the first MC sheet to count realizations
                        first_mc = wl[mc_keys[0]]
                        n_mc = sum(1 for c in first_mc.columns if c.startswith('MC_'))
                        n_years = len(first_mc)
                        pretty_lines.append(
                            f"Water-level Monte Carlo: **{n_mc:,}** annual-max realizations × "
                            f"**{n_years}** years × {len(mc_keys)} SLR scenarios"
                        )
                    if pretty_lines:
                        st.markdown("\n".join(f"- {ln}" for ln in pretty_lines))
    
    # ========================================================================
    # TAB 1: COMMUNITY SUMMARY
    # ========================================================================
    with tab1:
        st.markdown('<p class="tab-description">Aggregated community-wide damage statistics comparing all adaptation strategies, separated by buildings Under DFE and Above DFE.</p>', unsafe_allow_html=True)
        
        if df_agg is not None:
            st.subheader(f"Community-Wide Damage Summary — {location_name} ({occupancy_label}) — {target_year}, {scenario}")
            
            df_current = df_agg[
                (df_agg['TargetYear'] == target_year) & 
                (df_agg['SLR'] == scenario)
            ]
            
            col1, col2, col3, col4 = st.columns(4)
            
            baseline_row = df_current[df_current['Action'] == 'No mitigation']
            
            if not baseline_row.empty:
                baseline_p50 = baseline_row['Total_CumEAD_P50'].values[0]
                baseline_p05 = baseline_row['Total_CumEAD_P05'].values[0]
                baseline_p95 = baseline_row['Total_CumEAD_P95'].values[0]
                num_buildings = baseline_row['Num_Buildings'].values[0]
                
                infp_baseline = baseline_row['InFP_CumEAD_P50'].values[0] if 'InFP_CumEAD_P50' in baseline_row.columns else 0
                outfp_baseline = baseline_row['OutFP_CumEAD_P50'].values[0] if 'OutFP_CumEAD_P50' in baseline_row.columns else 0
                
                with col1:
                    st.metric(label=f"Total Buildings", value=f"{int(num_buildings):,}")
                with col2:
                    st.metric(label="No Mitigation Cumulative Damage (P50)", value=format_currency(baseline_p50),
                              help=f"Range: {format_currency(baseline_p05)} - {format_currency(baseline_p95)}")
                with col3:
                    st.metric(label="Under DFE (No Mitigation)", value=format_currency(infp_baseline),
                              help="Buildings with FFE below Design Flood Elevation (BFE+2)")
                with col4:
                    st.metric(label="Above DFE (No Mitigation)", value=format_currency(outfp_baseline),
                              help="Buildings with FFE above Design Flood Elevation (BFE+2)")
            
            st.divider()
            
            # ================================================================
            # AGGREGATED DAMAGE DISTRIBUTION — Both SLR scenarios side-by-side
            # Box edges = Q1/Q3 (interpolated from CDF); whiskers = P05/P95;
            # white center line = P50. Boxes summarize the distribution of
            # COMMUNITY-TOTAL damage across Monte Carlo realizations.
            # ================================================================
            st.subheader(f"Aggregated Damage Distribution — Year {target_year}")
            st.markdown(
                f"<p style='color:#64748b;font-size:0.95rem;margin-top:-0.5rem;'>"
                "Distribution of <b>community-total</b> cumulative damage across Monte Carlo "
                "realizations. Both SLR scenarios are shown side-by-side, regardless of the "
                "SLR Scenario chosen in the sidebar.</p>",
                unsafe_allow_html=True,
            )
            
            df_year_agg = df_agg[df_agg['TargetYear'] == target_year].copy()
            
            if df_year_agg.empty:
                st.info(f"No aggregated data for year {target_year}.")
            else:
                action_order_cs = ['No mitigation', 'Raise Utilities', 'WFP B', 'Elevate', 'WFP 1st']
                action_labels_cs = {
                    'No mitigation':   'No Mitigation',
                    'Raise Utilities': 'Raise Utilities',
                    'WFP B':           'WFP Basement',
                    'WFP 1st':         'WFP 1st Floor',
                    'Elevate':         'Elevate',
                }
                actions_present_cs = [a for a in action_order_cs
                                      if a in df_year_agg['Action'].unique()]
                
                # Pivots indexed by (Action, SLR). We pull P05/P25/P50/P75/P95
                # so the box edges are taken straight from the workbook's
                # stored quartiles (rather than interpolated between P05/P50/P95).
                piv_p05 = df_year_agg.pivot_table(index='Action', columns='SLR',
                                                  values='Total_CumEAD_P05')
                piv_p25 = (df_year_agg.pivot_table(index='Action', columns='SLR',
                                                   values='Total_CumEAD_P25')
                           if 'Total_CumEAD_P25' in df_year_agg.columns else None)
                piv_p50 = df_year_agg.pivot_table(index='Action', columns='SLR',
                                                  values='Total_CumEAD_P50')
                piv_p75 = (df_year_agg.pivot_table(index='Action', columns='SLR',
                                                   values='Total_CumEAD_P75')
                           if 'Total_CumEAD_P75' in df_year_agg.columns else None)
                piv_p95 = df_year_agg.pivot_table(index='Action', columns='SLR',
                                                  values='Total_CumEAD_P95')

                # 5-tuple (P05, P25, P50, P75, P95) when stored quartiles are
                # available, otherwise fall back to the legacy 3-tuple and let
                # build_box_whisker_panel interpolate Q1/Q3.
                _have_quartiles = piv_p25 is not None and piv_p75 is not None
                
                def _agg_stats(action, slr_key):
                    if (action not in piv_p50.index) or (slr_key not in piv_p50.columns):
                        return None
                    if _have_quartiles:
                        return (
                            float(piv_p05.loc[action, slr_key]),
                            float(piv_p25.loc[action, slr_key]),
                            float(piv_p50.loc[action, slr_key]),
                            float(piv_p75.loc[action, slr_key]),
                            float(piv_p95.loc[action, slr_key]),
                        )
                    return (
                        float(piv_p05.loc[action, slr_key]),
                        float(piv_p50.loc[action, slr_key]),
                        float(piv_p95.loc[action, slr_key]),
                    )
                
                def _agg_red_stats(action, slr_key):
                    if 'No mitigation' not in piv_p50.index:
                        return None
                    if (action not in piv_p50.index) or (slr_key not in piv_p50.columns):
                        return None
                    # Reduction = baseline_PX − strategy_PX, computed at each
                    # percentile rank. With real P25/P75 we can give a true
                    # 5-number summary of the reduction; otherwise we fall
                    # back to a 3-number summary at P05/P50/P95.
                    if _have_quartiles:
                        return (
                            float(piv_p05.loc['No mitigation', slr_key]) - float(piv_p05.loc[action, slr_key]),
                            float(piv_p25.loc['No mitigation', slr_key]) - float(piv_p25.loc[action, slr_key]),
                            float(piv_p50.loc['No mitigation', slr_key]) - float(piv_p50.loc[action, slr_key]),
                            float(piv_p75.loc['No mitigation', slr_key]) - float(piv_p75.loc[action, slr_key]),
                            float(piv_p95.loc['No mitigation', slr_key]) - float(piv_p95.loc[action, slr_key]),
                        )
                    return (
                        float(piv_p05.loc['No mitigation', slr_key]) - float(piv_p05.loc[action, slr_key]),
                        float(piv_p50.loc['No mitigation', slr_key]) - float(piv_p50.loc[action, slr_key]),
                        float(piv_p95.loc['No mitigation', slr_key]) - float(piv_p95.loc[action, slr_key]),
                    )
                
                # Cumulative-damage panel (all strategies)
                sd_cum = {slr: [_agg_stats(a, slr) for a in actions_present_cs]
                          for slr, *_ in SCENARIO_SPECS}
                fig_agg_cum = build_box_whisker_panel(
                    group_labels=[action_labels_cs[a] for a in actions_present_cs],
                    scenario_data=sd_cum,
                    panel_title="Cumulative Damage",
                    y_label="Community-Total Cumulative Damage",
                )
                
                # Reduction panel (no baseline)
                actions_no_base = [a for a in actions_present_cs if a != 'No mitigation']
                if actions_no_base:
                    sd_red = {slr: [_agg_red_stats(a, slr) for a in actions_no_base]
                              for slr, *_ in SCENARIO_SPECS}
                    fig_agg_red = build_box_whisker_panel(
                        group_labels=[action_labels_cs[a] for a in actions_no_base],
                        scenario_data=sd_red,
                        panel_title="Damage Reduction vs. No Mitigation",
                        y_label="Community-Total Damage Reduction",
                        bg_color='#f3f4f6',
                    )
                else:
                    fig_agg_red = None
                
                col_aggL, col_aggR = st.columns(2)
                with col_aggL:
                    st.plotly_chart(fig_agg_cum, use_container_width=True)
                with col_aggR:
                    if fig_agg_red is not None:
                        st.plotly_chart(fig_agg_red, use_container_width=True)
                    else:
                        st.info("No retrofit strategies available for the reduction panel.")
                
                st.caption(
                    "Each box summarizes the distribution of the **community-total** cumulative "
                    "damage across Monte Carlo realizations. Box edges show the 25th and 75th "
                    "percentiles, the white center line is the median (P50), and whiskers extend "
                    "to the 5th and 95th percentiles. All five percentiles are taken directly "
                    "from the aggregated Monte Carlo results stored in the workbook — no "
                    "interpolation between bounds is performed. The reduction panel is computed "
                    "percentile-by-percentile against the same-scenario No-Mitigation baseline."
                )
            
            st.divider()
            
            st.subheader("Adaptation Strategy Comparison by DFE Status")
            
            col_under, col_above = st.columns(2)
            
            with col_under:
                st.markdown("### 🔴 Under DFE (Below BFE+2)")
                
                if 'InFP_CumEAD_P50' in df_current.columns:
                    # Canonical action order: severity-of-intervention, with
                    # Elevate placed BEFORE WFP 1st Floor (per workshop
                    # convention; both the bar chart x-axis and the table
                    # rows below follow it).
                    dfe_action_order = ['No mitigation', 'Raise Utilities',
                                        'WFP B', 'Elevate', 'WFP 1st']
                    df_current_sorted = df_current.copy()
                    df_current_sorted['_ord'] = df_current_sorted['Action'].apply(
                        lambda a: dfe_action_order.index(a)
                                  if a in dfe_action_order else 999
                    )
                    df_current_sorted = df_current_sorted.sort_values('_ord')
                    
                    under_dfe_data = []
                    baseline_infp = df_current[df_current['Action'] == 'No mitigation']['InFP_CumEAD_P50'].values
                    baseline_infp = baseline_infp[0] if len(baseline_infp) > 0 else 0
                    
                    for _, row in df_current_sorted.iterrows():
                        action = row['Action']
                        val = row['InFP_CumEAD_P50']
                        savings = baseline_infp - val
                        pct = (savings / baseline_infp * 100) if baseline_infp > 0 else 0
                        under_dfe_data.append({
                            'Action': action,
                            'Cumulative Damage ($)': val,
                            'Savings': savings,
                            'Reduction (%)': pct
                        })
                    
                    df_under = pd.DataFrame(under_dfe_data)
                    
                    fig_under = px.bar(df_under, x='Action', y='Cumulative Damage ($)', color='Action',
                        color_discrete_map={'No mitigation': '#ef4444', 'Raise Utilities': '#f97316',
                            'WFP B': '#eab308', 'Elevate': '#22c55e', 'WFP 1st': '#3b82f6'},
                        category_orders={'Action': dfe_action_order},
                        title="Under DFE — All Strategies")
                    # Smart $k/$M/$B y-axis ticks
                    _u_max = df_under['Cumulative Damage ($)'].max()
                    u_ticks, u_labels = smart_money_ticks(_u_max, target_n=5)
                    fig_under.update_layout(
                        showlegend=False, height=300,
                        yaxis_title="Cumulative Damage",
                    )
                    fig_under.update_yaxes(tickmode='array', tickvals=u_ticks, ticktext=u_labels)
                    st.plotly_chart(fig_under, use_container_width=True)
                    
                    df_under_display = df_under.copy()
                    df_under_display['Cumulative Damage ($)'] = df_under_display['Cumulative Damage ($)'].apply(format_currency)
                    df_under_display['Savings'] = df_under_display['Savings'].apply(format_currency)
                    df_under_display['Reduction (%)'] = df_under_display['Reduction (%)'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(df_under_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No Under DFE data available.")
            
            with col_above:
                st.markdown("### 🟢 Above DFE (Above BFE+2)")
                
                if 'OutFP_CumEAD_P50' in df_current.columns:
                    # Same canonical order; Elevate is omitted from the
                    # Above-DFE chart (already at/above BFE+2 by definition).
                    dfe_action_order_above = ['No mitigation', 'Raise Utilities',
                                              'WFP B', 'WFP 1st']
                    df_current_sorted_a = df_current.copy()
                    df_current_sorted_a['_ord'] = df_current_sorted_a['Action'].apply(
                        lambda a: dfe_action_order_above.index(a)
                                  if a in dfe_action_order_above else 999
                    )
                    df_current_sorted_a = df_current_sorted_a.sort_values('_ord')
                    
                    above_dfe_data = []
                    baseline_outfp = df_current[df_current['Action'] == 'No mitigation']['OutFP_CumEAD_P50'].values
                    baseline_outfp = baseline_outfp[0] if len(baseline_outfp) > 0 else 0
                    
                    for _, row in df_current_sorted_a.iterrows():
                        action = row['Action']
                        if action == 'Elevate':
                            continue
                        val = row['OutFP_CumEAD_P50']
                        savings = baseline_outfp - val
                        pct = (savings / baseline_outfp * 100) if baseline_outfp > 0 else 0
                        above_dfe_data.append({
                            'Action': action,
                            'Cumulative Damage ($)': val,
                            'Savings': savings,
                            'Reduction (%)': pct
                        })
                    
                    df_above = pd.DataFrame(above_dfe_data)
                    
                    if not df_above.empty:
                        fig_above = px.bar(df_above, x='Action', y='Cumulative Damage ($)', color='Action',
                            color_discrete_map={'No mitigation': '#ef4444', 'Raise Utilities': '#f97316',
                                'WFP B': '#eab308', 'WFP 1st': '#3b82f6'},
                            category_orders={'Action': dfe_action_order_above},
                            title="Above DFE — Strategies (excl. Elevate)")
                        _a_max = df_above['Cumulative Damage ($)'].max()
                        a_ticks, a_labels = smart_money_ticks(_a_max, target_n=5)
                        fig_above.update_layout(
                            showlegend=False, height=300,
                            yaxis_title="Cumulative Damage",
                        )
                        fig_above.update_yaxes(tickmode='array', tickvals=a_ticks, ticktext=a_labels)
                        st.plotly_chart(fig_above, use_container_width=True)
                        
                        df_above_display = df_above.copy()
                        df_above_display['Cumulative Damage ($)'] = df_above_display['Cumulative Damage ($)'].apply(format_currency)
                        df_above_display['Savings'] = df_above_display['Savings'].apply(format_currency)
                        df_above_display['Reduction (%)'] = df_above_display['Reduction (%)'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(df_above_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No Above DFE data available.")
            
            st.divider()
            
            st.subheader("Damage Trajectory Over Time")
            
            # Include every adaptation strategy that's present in the data
            # (matching the Trends tab). The previous version omitted WFP 1st
            # despite it being available in the workbook, which biased the
            # visual story by hiding the first-floor wet-floodproof curve.
            traj_action_order = ['No mitigation', 'Raise Utilities',
                                 'WFP B', 'Elevate', 'WFP 1st']
            traj_action_labels = {
                'No mitigation':   'No Mitigation',
                'Raise Utilities': 'Raise Utilities',
                'WFP B':           'WFP Basement',
                'WFP 1st':         'WFP 1st Floor',
                'Elevate':         'Elevate',
            }
            traj_color_map = {
                'No Mitigation':   '#ef4444',   # red
                'Raise Utilities': '#f97316',   # orange
                'WFP Basement':    '#eab308',   # yellow
                'WFP 1st Floor':   '#3b82f6',   # blue
                'Elevate':         '#22c55e',   # green
            }
            
            df_timeline = df_agg[
                (df_agg['SLR'] == scenario) &
                (df_agg['Action'].isin(traj_action_order))
            ].copy()
            # Apply the consistent action labels so the legend reads cleanly
            df_timeline['Strategy'] = df_timeline['Action'].map(traj_action_labels)
            traj_present = [traj_action_labels[a] for a in traj_action_order
                            if a in df_timeline['Action'].unique()]
            
            if not df_timeline.empty:
                fig_line = px.line(
                    df_timeline, x='TargetYear', y='Total_CumEAD_P50',
                    color='Strategy',
                    category_orders={'Strategy': traj_present},
                    markers=True, color_discrete_map=traj_color_map,
                    title=f"Cumulative Damage Projection — {occupancy_label} ({scenario} SLR Scenario)",
                )
                _l_max = df_timeline['Total_CumEAD_P50'].max()
                l_ticks, l_labels = smart_money_ticks(_l_max, target_n=6)
                fig_line.update_layout(yaxis_title="Cumulative Damage",
                                       xaxis_title="Year", height=400)
                fig_line.update_yaxes(tickmode='array', tickvals=l_ticks, ticktext=l_labels)
                # All x-axis ticks render as plain integer years.
                _xs = sorted(df_timeline['TargetYear'].unique())
                _x_labels = [str(int(y)) for y in _xs]
                # Constrain the x-range to the actual data span (with a small
                # symmetric pad so markers don't sit on the axes). Without
                # this, Plotly's auto-range extends back ~5 years from the
                # first point, suggesting we have data we don't.
                if _xs:
                    _xpad = max(1.0, 0.04 * (max(_xs) - min(_xs)))
                    _xrange = [min(_xs) - _xpad, max(_xs) + _xpad]
                else:
                    _xrange = None
                fig_line.update_xaxes(tickmode='array', tickvals=_xs, ticktext=_x_labels,
                                      range=_xrange)
                st.plotly_chart(fig_line, use_container_width=True)
            
            # ----------------------------------------------------------------
            # Damage breakdown by building category (replaces DFE pie chart)
            # ----------------------------------------------------------------
            if df_buildings is not None:
                # Filter to current year + scenario + No mitigation baseline
                df_bd = df_buildings[
                    (df_buildings['TargetYear'] == target_year) &
                    (df_buildings['SLR'] == scenario) &
                    (df_buildings['Action'] == 'No mitigation')
                ].copy()
                
                # Pick the best available grouping column, preferring occupancy
                group_col = None
                group_label = None
                if 'occupancy_type' in df_bd.columns and df_bd['occupancy_type'].notna().sum() > 0:
                    group_col = 'occupancy_type'
                    group_label = 'Occupancy Type'
                elif 'foundation_type' in df_bd.columns and df_bd['foundation_type'].notna().sum() > 0:
                    group_col = 'foundation_type'
                    group_label = 'Foundation Type'
                
                if group_col is not None and not df_bd.empty:
                    st.subheader(f"Damage Breakdown by {group_label}")
                    
                    by_grp = (df_bd.groupby(group_col, dropna=False)
                              .agg(total_damage=('CumEAD_P50', 'sum'),
                                   n_buildings=('id', 'nunique'),
                                   avg_damage=('CumEAD_P50', 'mean'))
                              .reset_index())
                    by_grp[group_col] = by_grp[group_col].fillna('Unknown').astype(str)
                    by_grp = by_grp[by_grp['total_damage'] > 0].sort_values('total_damage', ascending=True)
                    
                    if not by_grp.empty:
                        total_damage_all = by_grp['total_damage'].sum()
                        by_grp['share'] = by_grp['total_damage'] / total_damage_all * 100
                        
                        # Bar labels: compact $ + count + share
                        bar_text = [
                            f"{fmt_money_short(d)}  •  {int(n):,} bldgs  •  {s:.1f}%"
                            for d, n, s in zip(by_grp['total_damage'], by_grp['n_buildings'], by_grp['share'])
                        ]
                        
                        _bd_max = by_grp['total_damage'].max()
                        bd_ticks, bd_labels = smart_money_ticks(_bd_max, target_n=5)
                        
                        fig_bd = go.Figure()
                        fig_bd.add_trace(go.Bar(
                            x=by_grp['total_damage'],
                            y=by_grp[group_col],
                            orientation='h',
                            marker=dict(
                                color=by_grp['total_damage'],
                                colorscale=[
                                    [0.0, '#fee2e2'],
                                    [0.5, '#f87171'],
                                    [1.0, '#b91c1c'],
                                ],
                                line=dict(color='white', width=1),
                            ),
                            text=bar_text,
                            textposition='outside',
                            cliponaxis=False,
                            hovertemplate=(
                                f"<b>%{{y}}</b><br>"
                                f"Total damage: %{{customdata[0]}}<br>"
                                f"Buildings: %{{customdata[1]:,}}<br>"
                                f"Avg per building: %{{customdata[2]}}<br>"
                                f"Share of total: %{{customdata[3]:.1f}}%"
                                "<extra></extra>"
                            ),
                            customdata=np.stack([
                                [fmt_money_short(v) for v in by_grp['total_damage']],
                                by_grp['n_buildings'].values,
                                [fmt_money_short(v) for v in by_grp['avg_damage']],
                                by_grp['share'].values,
                            ], axis=-1),
                        ))
                        
                        # Headroom so the outside text labels don't get clipped
                        x_head = _bd_max * 1.30 if _bd_max > 0 else 1.0
                        
                        fig_bd.update_layout(
                            title=dict(
                                text=(f"No-Mitigation Cumulative Damage by {group_label} — "
                                      f"{occupancy_label} ({target_year}, {scenario})"),
                                x=0.02, xanchor='left', font=dict(size=15),
                            ),
                            height=max(300, 55 * len(by_grp) + 120),
                            plot_bgcolor='white',
                            showlegend=False,
                            xaxis=dict(
                                title="Cumulative Damage",
                                showgrid=True, gridcolor='#e5e7eb',
                                showline=True, linecolor='#cbd5e1',
                                tickmode='array', tickvals=bd_ticks, ticktext=bd_labels,
                                range=[0, x_head],
                            ),
                            yaxis=dict(
                                title=group_label,
                                showgrid=False,
                                showline=True, linecolor='#cbd5e1',
                            ),
                            margin=dict(l=20, r=30, t=60, b=50),
                        )
                        
                        st.plotly_chart(fig_bd, use_container_width=True)
                        st.caption(
                            f"Each bar shows the total No-Mitigation cumulative damage for buildings in that "
                            f"{group_label.lower()}, with the number of buildings and share of total damage. "
                            "This helps identify which building categories drive community-wide damage."
                        )
        else:
            st.warning("No data available for this location.")
    
    # ========================================================================
    # TAB 3: BUILDING DETAILS
    # ========================================================================
    with tab3:
        st.markdown('<p class="tab-description">Select an individual building to view detailed damage projections across time horizons and compare adaptation options.</p>', unsafe_allow_html=True)
        
        if df_buildings is not None:
            st.subheader(f"🏠 Individual Building Analysis — {location_name} ({occupancy_label})")
            
            building_ids = df_buildings['id'].unique()
            sorted_ids = sorted(building_ids)
            
            # Cross-tab sync: if the user just clicked a building on the map, the
            # click-handler stored its id in session_state['selected_building_id'].
            # Streamlit widgets with a `key` restore their value from session_state
            # on every rerun, which would OVERRIDE a new default index. To force
            # the selectbox to pick up the clicked building, we write directly to
            # the widget's own state key BEFORE the widget is created.
            stored_id = st.session_state.get('selected_building_id')
            if stored_id is not None:
                try:
                    stored_id_int = int(stored_id)
                    if stored_id_int in building_ids:
                        st.session_state['bldg_details_selectbox'] = stored_id_int
                except (ValueError, TypeError):
                    pass
            
            try:
                default_idx = sorted_ids.index(int(stored_id)) if stored_id is not None else 0
            except (ValueError, TypeError):
                default_idx = 0
            
            came_from_map = (stored_id is not None and default_idx > 0
                             and st.session_state.get('bldg_details_selectbox')
                             == int(stored_id) if stored_id is not None else False)
            if came_from_map:
                st.caption(
                    f"📍 Showing the building you selected on the **Building Map** "
                    f"(#{stored_id}). Use the controls below to choose a different one."
                )
            
            # --- Search-by-ID + selectbox, side-by-side ---
            col_search, col_pick = st.columns([1, 2])
            
            with col_search:
                search_raw = st.text_input(
                    "🔍 Search by Building ID",
                    value="",
                    placeholder="e.g. 12345",
                    key='bldg_search_input',
                    help="Type a building ID and press Enter to jump to it.",
                )
                # If the user typed a valid id, make it take precedence over the
                # selectbox by writing to the selectbox's own state key BEFORE it
                # is instantiated (same trick as for the map-click handoff).
                search_hit = None
                if search_raw.strip():
                    try:
                        sid = int(search_raw.strip())
                        if sid in building_ids:
                            search_hit = sid
                            st.session_state['bldg_details_selectbox'] = sid
                            # Re-resolve the default index so the selectbox opens on it
                            default_idx = sorted_ids.index(sid)
                        else:
                            st.caption(
                                f"⚠️ Building **#{sid}** not found in this dataset "
                                f"(filtered by {occupancy_label})."
                            )
                    except ValueError:
                        st.caption("⚠️ Please enter a numeric building ID.")
            
            with col_pick:
                selected_id = st.selectbox(
                    "Or pick from the list",
                    options=sorted_ids,
                    index=default_idx,
                    format_func=lambda x: f"Building #{x}",
                    key='bldg_details_selectbox',
                )
            
            if search_hit is not None:
                st.caption(f"✅ Jumped to **Building #{search_hit}**.")
            
            # Keep the cross-tab pointer in sync with the current dropdown value.
            if selected_id is not None:
                st.session_state['selected_building_id'] = int(selected_id)
            
            if selected_id:
                df_building = df_buildings[df_buildings['id'] == selected_id]
                building_info = df_building.iloc[0]
                
                building_dfe_status = building_info.get('DFE_Status', 'Unknown')
                is_above_dfe = building_dfe_status == 'Above DFE'
                
                # Header line: lead with the address when we have one (NSI
                # ships them for ~75% of buildings); fall back to the bare
                # ID otherwise. Either way, the ID stays visible.
                addr = building_info.get('address') if 'address' in building_info else None
                if pd.notna(addr) and str(addr).strip():
                    st.subheader(f"{addr}")
                    st.caption(f"Building #{selected_id}")
                else:
                    st.subheader(f"Building #{selected_id}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Type**")
                    st.write(building_info.get('occupancy_type', 'N/A'))
                    if 'number_of_stories' in building_info:
                        st.markdown("**Stories**")
                        st.write(building_info.get('number_of_stories', 'N/A'))
                
                with col2:
                    if 'year_built' in building_info:
                        st.markdown("**Year Built**")
                        yr = building_info.get('year_built')
                        st.write(f"{int(yr)}" if pd.notna(yr) else 'N/A')
                    if 'area' in building_info:
                        st.markdown("**Area (sf)**")
                        area = building_info.get('area', 0)
                        st.write(f"{area:,.0f}" if pd.notna(area) else 'N/A')
                
                with col3:
                    if 'foundation_type' in building_info:
                        st.markdown("**Foundation**")
                        st.write(building_info.get('foundation_type', 'N/A'))
                    if 'FFE_ft' in building_info:
                        st.markdown("**FFE (ft)**")
                        ffe = building_info.get('FFE_ft', 0)
                        st.write(f"{ffe:.2f}" if pd.notna(ffe) else 'N/A')
                
                with col4:
                    if 'structure_value' in building_info:
                        st.markdown("**Structure Value**")
                        st.write(format_currency(building_info.get('structure_value', 0)))
                    if 'DFE_Status' in building_info:
                        st.markdown("**DFE Status**")
                        fp_status = building_info.get('DFE_Status', 'N/A')
                        if fp_status == 'Under DFE':
                            st.error(fp_status)
                        else:
                            st.success(fp_status)
                
                # Building-type and SOID provenance row — small, gray,
                # optional. Only renders when the bundle ships these fields.
                provenance_bits = []
                if pd.notna(building_info.get('building_type')):
                    provenance_bits.append(f"Building type: **{building_info['building_type']}**")
                if pd.notna(building_info.get('SOID')):
                    provenance_bits.append(f"Structural Occupancy ID: **{building_info['SOID']}**")
                if 'foundation_height' in building_info and pd.notna(building_info.get('foundation_height')):
                    provenance_bits.append(f"Foundation height: **{building_info['foundation_height']:.1f} ft**")
                if 'ground_elevation' in building_info and pd.notna(building_info.get('ground_elevation')):
                    provenance_bits.append(f"Ground elevation: **{building_info['ground_elevation']:.2f} ft NAVD88**")
                if provenance_bits:
                    st.caption(" · ".join(provenance_bits))
                
                # ----------------------------------------------------------
                # Annual-max water level exposure panel
                # ----------------------------------------------------------
                # Direct decision-support metric the bundle's raw MC
                # ensemble unlocks: P(annual-max stillwater ≥ FFE) for this
                # building, evaluated year-by-year under both SLR scenarios.
                # We compute against the raw 1,000-realization MC sheets
                # (water_levels[<slr>_mc]); when those aren't available we
                # silently skip — no fallback to percentile interpolation
                # since those would be misleading for a binary threshold.
                ffe_val = building_info.get('FFE_ft')
                bfe_local = (loc_entry or {}).get('bfe_ft')
                wl_data = (loc_entry or {}).get('water_levels') or {}
                if (pd.notna(ffe_val) and ffe_val
                        and ('50th-percentile_mc' in wl_data
                             or '90th-percentile_mc' in wl_data)):
                    st.divider()
                    st.subheader("Annual flood exposure")
                    st.caption(
                        "Probability that the annual-maximum stillwater level reaches or "
                        f"exceeds this building's first-floor elevation "
                        f"(**{ffe_val:.2f} ft NAVD88**) in each evaluation year, "
                        "computed directly from the 1,000-realization Monte Carlo ensemble."
                    )
                    
                    # Available years from the per-building damage table.
                    # We use df_building (defined above) rather than df_traj
                    # (which is defined later in the trajectory section);
                    # both have the same TargetYear values, but df_traj
                    # isn't in scope here.
                    target_years_local = sorted(df_building['TargetYear'].unique())
                    
                    exp_rows = []
                    for slr_key, slr_label in (('50th-percentile', 'Median SLR (P50)'),
                                                ('90th-percentile', 'High-End SLR (P90)')):
                        mc_df = wl_data.get(f'{slr_key}_mc')
                        if mc_df is None or 'Year' not in mc_df.columns:
                            continue
                        mc_cols = [c for c in mc_df.columns if c.startswith('MC_')]
                        if not mc_cols:
                            continue
                        for yr in target_years_local:
                            yr_mc = mc_df[mc_df['Year'] == int(yr)]
                            if yr_mc.empty:
                                continue
                            arr = yr_mc[mc_cols].to_numpy(dtype=float).flatten()
                            n = arr.size
                            if n == 0:
                                continue
                            p_ffe = float((arr >= ffe_val).sum()) / n
                            p_bfe = (float((arr >= bfe_local).sum()) / n
                                     if bfe_local is not None else float('nan'))
                            # Plain integer year; the 2025 baseline used to
                            # render as 'Potential' but a bare year is clearer.
                            label = str(int(yr))
                            exp_rows.append({
                                'SLR Scenario':         slr_label,
                                'Year':                 label,
                                f'P(WL ≥ FFE = {ffe_val:.1f} ft)':
                                    f"{p_ffe*100:.1f}%",
                                **({f'P(WL ≥ BFE = {bfe_local:g} ft)':
                                    f"{p_bfe*100:.1f}%"}
                                   if bfe_local is not None else {}),
                                'Median annual max (ft)': f"{float(np.median(arr)):.2f}",
                                'P95 annual max (ft)':    f"{float(np.percentile(arr, 95)):.2f}",
                            })
                    
                    if exp_rows:
                        st.dataframe(pd.DataFrame(exp_rows),
                                     use_container_width=True, hide_index=True)
                        st.caption(
                            "**P(WL ≥ FFE)** is the simulated annual exceedance "
                            "probability for this building's first floor — a direct, "
                            "decision-relevant counterpart to the cumulative-damage "
                            "estimates below. Years and SLR trajectories use the "
                            "same MC realizations that fed the damage chain."
                        )
                
                # ----------------------------------------------------------
                # Flood depth at building location
                # ----------------------------------------------------------
                # Companion to the exposure panel, but framed in depth (ft)
                # rather than probability. We sample three percentiles of
                # the simulated annual-maximum stillwater distribution
                # under the SIDEBAR-SELECTED SLR scenario and convert each
                # to a flood depth at the building by subtracting the
                # ground elevation. Percentiles match the user's spec:
                #   * Yearly      → P01 of annual max  (level the property
                #                   sees in ~99 % of years)
                #   * 10 % chance → P10 of annual max
                #   * 1  % chance → P99 of annual max  (≈ 100-year flood)
                # Negative values are clipped to zero (site stays dry).
                ground_elev = building_info.get('ground_elevation')
                mc_for_scenario = wl_data.get(f'{scenario}_mc')
                target_years_local = sorted(df_building['TargetYear'].unique())
                if (pd.notna(ground_elev)
                        and mc_for_scenario is not None
                        and 'Year' in mc_for_scenario.columns
                        and target_years_local):
                    st.divider()
                    scen_pretty = {
                        '50th-percentile': 'Median SLR (P50)',
                        '90th-percentile': 'High-End SLR (P90)',
                    }.get(scenario, scenario)
                    st.subheader(f"Flood depth at this building — {scen_pretty}")
                    st.caption(
                        f"Annual-maximum stillwater **flood depth above ground** "
                        f"at this property (ground elevation: "
                        f"**{float(ground_elev):.2f} ft NAVD88**) under the "
                        f"{scen_pretty.lower()} trajectory. The three depth "
                        f"columns sample the simulated annual-maximum "
                        f"distribution at three return-period proxies. "
                        f"Negative values are clipped to zero (site stays dry)."
                    )
                    
                    # (column-label, percentile)
                    pct_specs = [
                        ('Yearly (P01)',     1),
                        ('10% chance (P10)', 10),
                        ('1% chance (P99)',  99),
                    ]
                    
                    fd_rows = []
                    mc_cols_fd = [c for c in mc_for_scenario.columns
                                  if c.startswith('MC_')]
                    for yr in target_years_local:
                        yr_mc = mc_for_scenario[mc_for_scenario['Year'] == int(yr)]
                        if yr_mc.empty or not mc_cols_fd:
                            continue
                        arr = yr_mc[mc_cols_fd].to_numpy(dtype=float).flatten()
                        if arr.size == 0:
                            continue
                        # Plain integer year for the row label. The bundle
                        # ships a 'Potential' label for the 2025 baseline,
                        # but the bare year is less ambiguous to readers.
                        yr_int = int(yr)
                        year_display = str(yr_int)
                        row = {'Year': year_display}
                        for col_label, pct in pct_specs:
                            wl = float(np.percentile(arr, pct))
                            depth = wl - float(ground_elev)
                            wl_cell = f"{wl:.2f}"
                            if depth <= 0:
                                row[col_label] = f"Dry  (WL {wl_cell})"
                            else:
                                row[col_label] = f"{depth:.2f} ft  (WL {wl_cell})"
                        fd_rows.append(row)
                    
                    if fd_rows:
                        st.dataframe(pd.DataFrame(fd_rows),
                                     use_container_width=True, hide_index=True)
                        st.caption(
                            "Each cell shows **flood depth above ground (ft)** "
                            "with the corresponding **stillwater level (ft NAVD88)** "
                            "in parentheses. Depths are computed from this "
                            "building's ground elevation and the simulated "
                            "annual-maximum water-level percentile under the "
                            "selected SLR scenario; switch SLR in the sidebar "
                            "to compare. "
                            "*Note:* the three return-period proxies use "
                            "P01 / P10 / P99 of the annual-maximum distribution "
                            "as you specified — let me know if you'd prefer "
                            "P50 / P90 / P99 (the standard engineering reading "
                            "of \"yearly / 10-year / 100-year\")."
                        )
                
                st.divider()
                
                st.subheader("Damage Trajectory: SLR Scenario Comparison")
                st.caption(
                    "Cumulative damage under the No-Mitigation baseline for both sea-level rise scenarios. "
                    "Solid lines are the median estimate (P50); shaded bands show the 90% confidence interval (P05–P95)."
                )
                
                df_traj = df_building[df_building['Action'] == 'No mitigation'].sort_values(['SLR', 'TargetYear'])
                
                if not df_traj.empty:
                    fig_building = go.Figure()
                    
                    scenario_styles = {
                        '50th-percentile': {
                            'label': 'Median SLR (P50)',
                            'line_color': '#3b82f6',                  # blue, matches Tab 4
                            'fill_color': 'rgba(59, 130, 246, 0.18)',
                        },
                        '90th-percentile': {
                            'label': 'High-End SLR (P90)',
                            'line_color': '#ef4444',                  # red, matches Tab 4
                            'fill_color': 'rgba(239, 68, 68, 0.18)',
                        },
                    }
                    
                    for slr_key, style in scenario_styles.items():
                        df_s = df_traj[df_traj['SLR'] == slr_key]
                        if df_s.empty:
                            continue
                        # 90% CI band first, so the central line draws on top
                        fig_building.add_trace(go.Scatter(
                            x=list(df_s['TargetYear']) + list(df_s['TargetYear'])[::-1],
                            y=list(df_s['CumEAD_P95']) + list(df_s['CumEAD_P05'])[::-1],
                            fill='toself', fillcolor=style['fill_color'],
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f"{style['label']} — 90% CI",
                            hoverinfo='skip', showlegend=True))
                        # Central P50 line
                        fig_building.add_trace(go.Scatter(
                            x=df_s['TargetYear'], y=df_s['CumEAD_P50'],
                            mode='lines+markers',
                            name=style['label'],
                            line=dict(color=style['line_color'], width=3),
                            marker=dict(size=11, line=dict(color='white', width=1.5)),
                            hovertemplate='Year %{x}<br>P50: $%{y:,.0f}<extra></extra>'))
                    
                    # Smart y-axis ticks for the trajectory chart
                    _t_max = df_traj['CumEAD_P95'].max() if 'CumEAD_P95' in df_traj.columns else df_traj['CumEAD_P50'].max()
                    t_ticks, t_labels = smart_money_ticks(_t_max, target_n=6)
                    
                    fig_building.update_layout(
                        title=dict(
                            text=f"Cumulative Damage Trajectory — Building #{selected_id}",
                            x=0.02, xanchor='left', font=dict(size=16)),
                        xaxis_title="Year",
                        yaxis_title="Cumulative Damage",
                        height=460,
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                    xanchor='right', x=1, bgcolor='rgba(255,255,255,0.85)'),
                        plot_bgcolor='white',
                        margin=dict(l=60, r=20, t=70, b=50),
                    )
                    _bd_xs = sorted(df_traj['TargetYear'].unique())
                    _bd_x_labels = [str(int(y)) for y in _bd_xs]
                    # Constrain x-range to actual data span with a small pad
                    # (otherwise Plotly auto-extends back several years).
                    if _bd_xs:
                        _bd_xpad = max(1.0, 0.04 * (max(_bd_xs) - min(_bd_xs)))
                        _bd_xrange = [min(_bd_xs) - _bd_xpad, max(_bd_xs) + _bd_xpad]
                    else:
                        _bd_xrange = None
                    fig_building.update_xaxes(
                        showgrid=True, gridcolor='#e5e7eb',
                        tickmode='array', tickvals=_bd_xs, ticktext=_bd_x_labels,
                        range=_bd_xrange,
                        showline=True, linecolor='#cbd5e1')
                    fig_building.update_yaxes(
                        showgrid=True, gridcolor='#e5e7eb',
                        showline=True, linecolor='#cbd5e1', zeroline=False,
                        tickmode='array', tickvals=t_ticks, ticktext=t_labels)
                    
                    st.plotly_chart(fig_building, use_container_width=True)
                    
                    # ---- Side-by-side numeric comparison ----
                    pivot_p50 = df_traj.pivot_table(index='TargetYear', columns='SLR', values='CumEAD_P50')
                    pivot_p05 = df_traj.pivot_table(index='TargetYear', columns='SLR', values='CumEAD_P05')
                    pivot_p95 = df_traj.pivot_table(index='TargetYear', columns='SLR', values='CumEAD_P95')
                    
                    def _fmt_range(lo, hi):
                        if pd.isna(lo) or pd.isna(hi):
                            return "—"
                        return f"{format_currency(lo)} – {format_currency(hi)}"
                    
                    def _fmt_val(v):
                        return "—" if pd.isna(v) else format_currency(v)
                    
                    cmp_rows = []
                    for yr in sorted(pivot_p50.index):
                        med = pivot_p50.loc[yr].get('50th-percentile', float('nan'))
                        high = pivot_p50.loc[yr].get('90th-percentile', float('nan'))
                        med_lo = pivot_p05.loc[yr].get('50th-percentile', float('nan'))
                        med_hi = pivot_p95.loc[yr].get('50th-percentile', float('nan'))
                        high_lo = pivot_p05.loc[yr].get('90th-percentile', float('nan'))
                        high_hi = pivot_p95.loc[yr].get('90th-percentile', float('nan'))
                        if pd.notna(med) and pd.notna(high):
                            delta = high - med
                            pct = (delta / med * 100) if med > 0 else float('nan')
                        else:
                            delta, pct = float('nan'), float('nan')
                        cmp_rows.append({
                            'Year': int(yr),
                            'Median SLR — P50': _fmt_val(med),
                            'Median SLR — 90% CI': _fmt_range(med_lo, med_hi),
                            'High-End SLR — P50': _fmt_val(high),
                            'High-End SLR — 90% CI': _fmt_range(high_lo, high_hi),
                            'High-End vs Median (Δ)': _fmt_val(delta),
                            'Increase (%)': f"{pct:.1f}%" if pd.notna(pct) else "—",
                        })
                    
                    st.markdown("**Side-by-side comparison across planning horizons**")
                    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                
                st.subheader(f"Adaptation Strategy Comparison — Both SLR Scenarios ({target_year})")
                st.caption(
                    "How each adaptation strategy performs under both sea-level rise scenarios at the selected planning horizon. "
                    "Box edges show the 25th and 75th percentiles, the white center line is the median (P50), "
                    "and whiskers extend to the 5th and 95th percentiles of this building's cumulative damage distribution."
                )
                
                # Both scenarios for the selected target year
                df_building_year = df_building[df_building['TargetYear'] == target_year].copy()
                
                if is_above_dfe:
                    df_building_year = df_building_year[df_building_year['Action'] != 'Elevate']
                
                if not df_building_year.empty:
                    # Consistent ordering: baseline first, then strategies from least → most invasive
                    bd_action_order = ['No mitigation', 'Raise Utilities', 'WFP B', 'Elevate', 'WFP 1st']
                    bd_action_labels = {
                        'No mitigation':   'No Mitigation',
                        'Raise Utilities': 'Raise Utilities',
                        'WFP B':           'WFP Basement',
                        'WFP 1st':         'WFP 1st Floor',
                        'Elevate':         'Elevate',
                    }
                    # For RES2 (manufactured housing) and RES4 (small mixed-use /
                    # temporary lodging) buildings, basement wet-floodproofing
                    # and 1st-floor wet-floodproofing aren't viable retrofits —
                    # these structures typically have no basement and no
                    # conditioned 1st-floor envelope to dry out. Restrict the
                    # benefit chart (and the strategy axis below) to the
                    # retrofits that actually apply: Raise Utilities and
                    # Elevate. Baseline (No Mitigation) is kept here for the
                    # benefit-stats reference; the chart still drops it via
                    # `bd_actions_for_chart` below.
                    _occ_for_strategy_filter = str(
                        building_info.get('occupancy_type', '')
                    ).upper()
                    _is_res2_or_res4 = (
                        _occ_for_strategy_filter.startswith('RES2')
                        or _occ_for_strategy_filter.startswith('RES4')
                    )
                    if _is_res2_or_res4:
                        bd_action_order = [
                            a for a in bd_action_order
                            if a not in ('WFP B', 'WFP 1st')
                        ]
                    bd_actions_present = [a for a in bd_action_order
                                          if a in df_building_year['Action'].unique()]
                    
                    # Pivots over (Action, SLR) — values are this building's P05/P50/P95
                    bd_piv = df_building_year.pivot_table(
                        index='Action', columns='SLR',
                        values=['CumEAD_P05', 'CumEAD_P50', 'CumEAD_P95'],
                    )
                    
                    def _pct(action, slr_key, col):
                        try:
                            v = float(bd_piv.loc[action, (col, slr_key)])
                            return v if pd.notna(v) else float('nan')
                        except (KeyError, ValueError):
                            return float('nan')
                    
                    def _benefit_stats(action, slr_key):
                        """Return a 5-tuple (P05, P25, P50, P75, P95) of
                        avoided-damage benefit for a retrofit strategy,
                        clamped so the median is always between the bounds.

                        Benefit is the avoided damage relative to No
                        Mitigation under the same SLR scenario, computed
                        percentile-by-percentile. Returns ``None`` for the
                        No-Mitigation row (no sensible self-benefit).

                        We pull P25 and P75 straight from the workbook when
                        available (the ALL format stores them), so the box
                        edges of the benefit chart are real percentile
                        differences rather than CDF-linear approximations
                        between P05 and P95.
                        """
                        if action == 'No mitigation':
                            return None
                        # Pull every percentile we have. P25/P75 will be NaN
                        # for older workbooks that don't store them; in that
                        # case we fall back to a 3-tuple.
                        b05 = _pct('No mitigation', slr_key, 'CumEAD_P05')
                        b25 = _pct('No mitigation', slr_key, 'CumEAD_P25')
                        b50 = _pct('No mitigation', slr_key, 'CumEAD_P50')
                        b75 = _pct('No mitigation', slr_key, 'CumEAD_P75')
                        b95 = _pct('No mitigation', slr_key, 'CumEAD_P95')
                        s05 = _pct(action, slr_key, 'CumEAD_P05')
                        s25 = _pct(action, slr_key, 'CumEAD_P25')
                        s50 = _pct(action, slr_key, 'CumEAD_P50')
                        s75 = _pct(action, slr_key, 'CumEAD_P75')
                        s95 = _pct(action, slr_key, 'CumEAD_P95')
                        if any(pd.isna(v) for v in (b05, b50, b95, s05, s50, s95)):
                            return None
                        raw_ben05 = b05 - s05
                        raw_ben50 = b50 - s50
                        raw_ben95 = b95 - s95
                        have_quartiles = not any(
                            pd.isna(v) for v in (b25, b75, s25, s75)
                        )
                        if have_quartiles:
                            raw_ben25 = b25 - s25
                            raw_ben75 = b75 - s75
                            # Clamp the lower whisker, Q1, Q3, and upper
                            # whisker so the box and whisker geometry is
                            # monotone — even if rank-correlation between
                            # baseline and strategy realizations briefly
                            # inverts, the plotted geometry stays sensible.
                            vals = sorted([raw_ben05, raw_ben25, raw_ben50,
                                           raw_ben75, raw_ben95])
                            lo, q1, _, q3, hi = vals
                            return (lo, q1, raw_ben50, q3, hi)
                        # Fallback: 3-tuple (whisker, median, whisker), with
                        # build_box_whisker_panel interpolating Q1/Q3.
                        lo = min(raw_ben05, raw_ben50, raw_ben95)
                        hi = max(raw_ben05, raw_ben50, raw_ben95)
                        return (lo, raw_ben50, hi)
                    
                    # Strategies shown on the chart exclude No Mitigation (no
                    # self-benefit), matching the Benefit columns in the table.
                    bd_actions_for_chart = [a for a in bd_actions_present
                                             if a != 'No mitigation']
                    
                    bd_scenario_data = {
                        slr_key: [_benefit_stats(a, slr_key) for a in bd_actions_for_chart]
                        for slr_key, *_ in SCENARIO_SPECS
                    }
                    
                    if bd_actions_for_chart and any(
                        any(v is not None for v in vals)
                        for vals in bd_scenario_data.values()
                    ):
                        fig_strat = build_box_whisker_panel(
                            group_labels=[bd_action_labels[a] for a in bd_actions_for_chart],
                            scenario_data=bd_scenario_data,
                            panel_title=(
                                f"Benefit (avoided damage) by Strategy and SLR Scenario — "
                                f"Building #{selected_id}, Year {target_year}"
                            ),
                            y_label="Benefit — avoided cumulative damage vs No Mitigation",
                            height=500,
                        )
                        
                        st.plotly_chart(fig_strat, use_container_width=True)
                        st.caption(
                            "Each box shows the distribution of this building's **avoided damage** "
                            "(No-Mitigation damage minus the strategy's remaining damage) at the same "
                            "percentile rank, under the selected target year and for both SLR scenarios. "
                            "The white center line is the median benefit (matches the *Benefit — Median* "
                            "column below); the whiskers reach to the 5th and 95th percentile benefits "
                            "(matching the *Benefit — 5th pctile* and *Benefit — 95th pctile* columns). "
                            "No Mitigation is omitted because it has no self-benefit."
                        )
                    
                    # ============================================================
                    # Strategy performance tables — Damage, Benefit, Remaining damage
                    # Separate Median / Min / Max columns (not single "range" cells)
                    # ============================================================
                    def _fmt_v(v):
                        return "—" if pd.isna(v) else format_currency(v)
                    def _fmt_pct(v):
                        return "—" if pd.isna(v) else f"{v:.1f}%"
                    
                    # Clamp (min, median, max) so the median is always bracketed by
                    # the printed min and max. Under perfect positive rank-correlation
                    # between baseline and strategy realizations, the benefit's
                    # implied P05/P95 are baseline_PX - strategy_PX; those can
                    # occasionally straddle the median tightly, so we clamp to keep
                    # the display self-consistent.
                    def _bracket(lo, med, hi):
                        vals = [v for v in (lo, med, hi) if pd.notna(v)]
                        if not vals:
                            return float('nan'), float('nan'), float('nan')
                        return min(vals), med, max(vals)
                    
                    # Mapping from sidebar's scenario label -> SLR key in the data
                    _sidebar_slr_key = {
                        '50th-percentile': '50th-percentile',
                        '90th-percentile': '90th-percentile',
                    }.get(scenario, scenario)
                    
                    def _build_strategy_rows_for_slr(year, slr_key):
                        """Per-strategy rows for a given (year, SLR) — split columns.
                        Returns list[dict] with Remaining damage (Med/Min/Max),
                        Benefit (Med/Min/Max), and Reduction %."""
                        df_y = df_building[df_building['TargetYear'] == year].copy()
                        if is_above_dfe:
                            df_y = df_y[df_y['Action'] != 'Elevate']
                        if df_y.empty:
                            return []
                        df_ys = df_y[df_y['SLR'] == slr_key]
                        if df_ys.empty:
                            return []
                        
                        actions_here = [a for a in bd_action_order
                                        if a in df_ys['Action'].unique()]
                        
                        ps = df_ys.set_index('Action')
                        
                        def _get(action, col):
                            try:
                                v = ps.loc[action, col]
                                # If duplicate, pandas may return a Series; take first
                                if hasattr(v, 'iloc'):
                                    v = v.iloc[0]
                                return float(v) if pd.notna(v) else float('nan')
                            except (KeyError, ValueError):
                                return float('nan')
                        
                        base05 = _get('No mitigation', 'CumEAD_P05')
                        base50 = _get('No mitigation', 'CumEAD_P50')
                        base95 = _get('No mitigation', 'CumEAD_P95')
                        
                        rows = []
                        for action in actions_here:
                            s05 = _get(action, 'CumEAD_P05')
                            s50 = _get(action, 'CumEAD_P50')
                            s95 = _get(action, 'CumEAD_P95')
                            
                            # Remaining damage bounds (clamped so median is inside)
                            dmg_lo, dmg_med, dmg_hi = _bracket(s05, s50, s95)
                            
                            if action == 'No mitigation':
                                ben_lo = ben_med = ben_hi = float('nan')
                                red_pct = float('nan')
                            else:
                                raw_ben05 = base05 - s05 if pd.notna(base05) and pd.notna(s05) else float('nan')
                                raw_ben50 = base50 - s50 if pd.notna(base50) and pd.notna(s50) else float('nan')
                                raw_ben95 = base95 - s95 if pd.notna(base95) and pd.notna(s95) else float('nan')
                                ben_lo, ben_med, ben_hi = _bracket(raw_ben05, raw_ben50, raw_ben95)
                                red_pct = (raw_ben50 / base50 * 100) if pd.notna(raw_ben50) and base50 > 0 else float('nan')
                            
                            rows.append({
                                'Strategy':                             bd_action_labels[action],
                                'Remaining damage — 5th pctile':        _fmt_v(dmg_lo),
                                'Remaining damage — Median':            _fmt_v(dmg_med),
                                'Remaining damage — 95th pctile':       _fmt_v(dmg_hi),
                                'Benefit — 5th pctile':                 _fmt_v(ben_lo) if action != 'No mitigation' else "—",
                                'Benefit — Median':                     _fmt_v(ben_med) if action != 'No mitigation' else "—",
                                'Benefit — 95th pctile':                _fmt_v(ben_hi) if action != 'No mitigation' else "—",
                                'Reduction (median)':                   _fmt_pct(red_pct) if action != 'No mitigation' else "—",
                            })
                        return rows
                    
                    # ---- Selected-year tables, one per SLR scenario ----
                    st.markdown(f"**Strategy performance — Year {target_year}**")
                    for slr_key, slr_label in [
                        ('50th-percentile', 'Median SLR (P50)'),
                        ('90th-percentile', 'High-End SLR (P90)'),
                    ]:
                        rows = _build_strategy_rows_for_slr(target_year, slr_key)
                        if not rows:
                            continue
                        st.markdown(f"*{slr_label}*")
                        st.dataframe(pd.DataFrame(rows),
                                     use_container_width=True, hide_index=True)
                    
                    st.caption(
                        "**Remaining damage** is the cumulative damage left under each strategy, "
                        "reported at the 5th percentile, median (50th), and 95th percentile of the "
                        "Monte Carlo damage realizations. **Benefit** is the avoided damage compared "
                        "to No Mitigation under the same SLR scenario, shown the same way. "
                        "**Reduction (median)** is the percent drop in median damage relative to "
                        "No Mitigation. The printed 5th and 95th percentiles are clamped so the "
                        "median always falls between them."
                    )
                    if is_above_dfe:
                        st.info(
                            "ℹ️ The **Elevate Structure** option is not shown for this building because its first-floor "
                            "elevation already meets or exceeds the Design Flood Elevation (BFE + 2 ft)."
                        )
                    
                    # ---- Cross-year comparison table (selected SLR only) ----
                    years_all = sorted(df_building['TargetYear'].unique())
                    if len(years_all) > 1:
                        st.markdown("---")
                        scenario_label = (
                            "Median SLR (P50)" if _sidebar_slr_key == '50th-percentile'
                            else "High-End SLR (P90)" if _sidebar_slr_key == '90th-percentile'
                            else scenario
                        )
                        st.markdown(
                            "**Strategy performance across all planning horizons**  \n"
                            f"<span style='color:#64748b;font-size:0.9rem;'>"
                            f"Building #{selected_id} — {scenario_label} "
                            "(change the SLR Scenario in the sidebar to see the other)</span>",
                            unsafe_allow_html=True,
                        )
                        
                        multi_rows = []
                        for yr in years_all:
                            yr_rows = _build_strategy_rows_for_slr(yr, _sidebar_slr_key)
                            if not yr_rows:
                                continue
                            for r in yr_rows:
                                multi_rows.append({'Year': yr, **r})
                        
                        if multi_rows:
                            df_multi = pd.DataFrame(multi_rows)
                            st.dataframe(df_multi, use_container_width=True, hide_index=True)
                            st.caption(
                                "Same building, same metrics as above, broken out by planning horizon "
                                f"under the **{scenario_label}** scenario. Use this to compare how a "
                                "given strategy performs in 2040 vs 2055 vs 2060 vs 2100 — and how its benefit "
                                "and remaining damage evolve as sea level rises."
                            )
        else:
            st.warning("No per-building data available for this location.")
    
    # ========================================================================
    # TAB 4: SCENARIO COMPARISON
    # ========================================================================
    with tab4:
        st.markdown('<p class="tab-description">Compare cumulative damage projections between Median (50th-percentile) and High-End (90th-percentile) sea level rise scenarios across all time horizons.</p>', unsafe_allow_html=True)
        
        if df_agg is not None:
            st.subheader(f"📈 Scenario Comparison — {location_name} ({occupancy_label})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Median SLR (P50)")
                df_p50 = df_agg[(df_agg['SLR'] == '50th-percentile') & (df_agg['Action'] == 'No mitigation')].sort_values('TargetYear')
                for _, row in df_p50.iterrows():
                    st.metric(label=f"Year {int(row['TargetYear'])}", value=format_currency(row['Total_CumEAD_P50']))
            
            with col2:
                st.markdown("### High-End SLR (P90)")
                df_p90 = df_agg[(df_agg['SLR'] == '90th-percentile') & (df_agg['Action'] == 'No mitigation')].sort_values('TargetYear')
                for _, row in df_p90.iterrows():
                    p50_val = df_p50[df_p50['TargetYear'] == row['TargetYear']]['Total_CumEAD_P50'].values
                    delta = row['Total_CumEAD_P50'] - p50_val[0] if len(p50_val) > 0 else 0
                    st.metric(label=f"Year {int(row['TargetYear'])}", value=format_currency(row['Total_CumEAD_P50']),
                        delta=f"+{format_currency(delta)} vs Median", delta_color="inverse")
            
            st.divider()
            
            df_comparison = df_agg[df_agg['Action'] == 'No mitigation'].copy()
            df_comparison['Label'] = df_comparison['SLR'].map({'50th-percentile': 'Median SLR', '90th-percentile': 'High-End SLR'})
            
            if not df_comparison.empty:
                fig_comp = px.line(df_comparison, x='TargetYear', y='Total_CumEAD_P50', color='Label',
                    markers=True, color_discrete_map={'Median SLR': '#3b82f6', 'High-End SLR': '#ef4444'},
                    title=f"No Mitigation Damage: Median vs High-End SLR Scenarios — {occupancy_label}")
                _c_max = df_comparison['Total_CumEAD_P50'].max()
                c_ticks, c_labels = smart_money_ticks(_c_max, target_n=6)
                fig_comp.update_layout(height=450, yaxis_title="Cumulative Damage", xaxis_title="Year")
                fig_comp.update_yaxes(tickmode='array', tickvals=c_ticks, ticktext=c_labels)
                # Constrain x-axis to actual data span (otherwise Plotly's
                # auto-range pads backward, suggesting data we don't have).
                _comp_xs = sorted(df_comparison['TargetYear'].unique())
                if _comp_xs:
                    _comp_xpad = max(1.0, 0.04 * (max(_comp_xs) - min(_comp_xs)))
                    _comp_xrange = [min(_comp_xs) - _comp_xpad,
                                    max(_comp_xs) + _comp_xpad]
                else:
                    _comp_xrange = None
                fig_comp.update_xaxes(
                    tickmode='array',
                    tickvals=_comp_xs,
                    ticktext=[str(int(y)) for y in _comp_xs],
                    range=_comp_xrange,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            # ================================================================
            # Per-SLR-scenario trajectories — all mitigation actions together
            # Separate plot per SLR scenario so each strategy's trend over time
            # can be compared directly to the No-Mitigation baseline.
            # ================================================================
            st.divider()
            st.subheader("📉 Mitigation Strategies Over Time — by SLR Scenario")
            st.caption(
                "Cumulative community damage trajectory for every adaptation strategy "
                "(including No Mitigation), plotted separately for each sea-level rise "
                "scenario. Use this to compare how each strategy bends the damage curve "
                "as the planning horizon extends."
            )

            # Stable action ordering + display labels, matching the other tabs
            trend_action_order = ['No mitigation', 'Raise Utilities', 'WFP B', 'Elevate', 'WFP 1st']
            trend_action_labels = {
                'No mitigation':   'No Mitigation',
                'Raise Utilities': 'Raise Utilities',
                'WFP B':           'WFP Basement',
                'WFP 1st':         'WFP 1st Floor',
                'Elevate':         'Elevate',
            }
            # Fixed per-action colors so the two SLR panels look consistent
            trend_action_colors = {
                'No mitigation':   '#ef4444',   # red — baseline
                'Raise Utilities': '#0ea5e9',   # sky blue
                'WFP B':           '#6366f1',   # indigo
                'WFP 1st':         '#8b5cf6',   # violet
                'Elevate':         '#10b981',   # emerald
            }

            trend_slr_panels = [
                ('50th-percentile', 'Median SLR (P50)'),
                ('90th-percentile', 'High-End SLR (P90)'),
            ]
            # Shared y-axis range so both panels are directly comparable
            _trend_max = df_agg['Total_CumEAD_P95'].max() if 'Total_CumEAD_P95' in df_agg.columns else df_agg['Total_CumEAD_P50'].max()
            trend_ticks, trend_tick_labels = smart_money_ticks(_trend_max, target_n=6)
            trend_y_range = [0, _trend_max * 1.08 if pd.notna(_trend_max) and _trend_max > 0 else 1]

            t_col1, t_col2 = st.columns(2)
            for col, (slr_key, slr_label) in zip((t_col1, t_col2), trend_slr_panels):
                df_slr = df_agg[df_agg['SLR'] == slr_key].copy()
                if df_slr.empty:
                    with col:
                        st.info(f"No data for {slr_label}.")
                    continue

                actions_present = [a for a in trend_action_order if a in df_slr['Action'].unique()]
                fig_trend = go.Figure()
                for action in actions_present:
                    df_a = df_slr[df_slr['Action'] == action].sort_values('TargetYear')
                    if df_a.empty:
                        continue
                    clr = trend_action_colors.get(action, '#475569')
                    fig_trend.add_trace(go.Scatter(
                        x=df_a['TargetYear'],
                        y=df_a['Total_CumEAD_P50'],
                        mode='lines+markers',
                        name=trend_action_labels.get(action, action),
                        line=dict(color=clr, width=3 if action == 'No mitigation' else 2.2,
                                  dash='solid' if action == 'No mitigation' else 'dot'),
                        marker=dict(size=10 if action == 'No mitigation' else 8,
                                    line=dict(color='white', width=1.2)),
                        hovertemplate=(
                            f"<b>{trend_action_labels.get(action, action)}</b><br>"
                            "Year %{x}<br>P50: %{y:$,.0f}<extra></extra>"
                        ),
                    ))

                fig_trend.update_layout(
                    title=dict(text=f"{slr_label} — {occupancy_label}",
                               x=0.02, xanchor='left', font=dict(size=14)),
                    height=450,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=60, r=20, t=60, b=50),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                xanchor='right', x=1, bgcolor='rgba(255,255,255,0.85)'),
                )
                _trend_xs = sorted(df_slr['TargetYear'].unique())
                # Plain integer-year ticks (no 'Potential' / 2025 anchor).
                _trend_x_labels = [str(int(y)) for y in _trend_xs]
                # Tight x-range with a small symmetric padding so the first
                # data marker doesn't sit on the y-axis. Without an explicit
                # range, Plotly auto-extends to ~2020 (or further), which
                # made it look like the data started in some baseline year
                # we never actually computed.
                if _trend_xs:
                    _trend_xpad = max(1.0, 0.04 * (max(_trend_xs) - min(_trend_xs)))
                    _trend_xrange = [min(_trend_xs) - _trend_xpad,
                                     max(_trend_xs) + _trend_xpad]
                else:
                    _trend_xrange = None
                fig_trend.update_xaxes(
                    title="Year", showgrid=True, gridcolor='#e5e7eb',
                    showline=True, linecolor='#cbd5e1',
                    tickmode='array', tickvals=_trend_xs, ticktext=_trend_x_labels,
                    range=_trend_xrange,
                )
                fig_trend.update_yaxes(
                    title="Cumulative Damage", showgrid=True, gridcolor='#e5e7eb',
                    showline=True, linecolor='#cbd5e1', zeroline=False,
                    tickmode='array', tickvals=trend_ticks, ticktext=trend_tick_labels,
                    range=trend_y_range,
                )
                with col:
                    st.plotly_chart(fig_trend, use_container_width=True)

            st.subheader("📋 Full Data Table")
            
            df_display = df_agg.copy()
            for col in ['Total_CumEAD_P05', 'Total_CumEAD_P50', 'Total_CumEAD_P95', 'InFP_CumEAD_P50', 'OutFP_CumEAD_P50']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}")
            
            rename_cols = {'InFP_CumEAD_P50': 'Under_DFE_P50', 'OutFP_CumEAD_P50': 'Above_DFE_P50'}
            df_display = df_display.rename(columns={k: v for k, v in rename_cols.items() if k in df_display.columns})
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.warning("No data available for this location.")

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.divider()
    st.markdown("""
    <div class="footer">
        <div class="footer-org">
            Center for Climate Systems Research<br>
            The Climate School<br>
            Columbia University
        </div>
        <div class="footer-license">
            © 2025 Erfan Amini. All rights reserved.<br>
            DFE = Design Flood Elevation (BFE+2)
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
