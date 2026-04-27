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
    """Convert floodplain terminology to DFE terminology"""
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
# New-format (MC-realization) Excel loader
# ----------------------------------------------------------------------------
# The "new" workbook layout stores every Monte Carlo realization as its own
# column (MC_0001 … MC_nnnn) on three data sheets:
#   * Dmg_agg              — annual damage per (Year, Action, SLR)
#   * CumEAD_Categories    — cumulative damage per (Category, TargetYear,
#                            Action, SLR) for Category ∈ {ALL, RES, NONRES,
#                            RES_InFP, RES_OutFP, NONRES_InFP, NONRES_OutFP}
#   * bldg_CumEAD_MC       — cumulative damage per (BuildingID, TargetYear,
#                            Action, SLR)
# Plus two smaller descriptor sheets (Metadata, Buildings) and two water-level
# sheets (WL_P50, WL_P90). Percentiles (P05, P10, P50, P90, P95) are NOT
# pre-computed — they're derived here by taking np.percentile across the MC
# columns for each row. The per-building and per-category results are then
# packaged into the same DataFrame schema the rest of the app already uses.

# Canonical list of percentiles we derive from MC realizations
# Match the percentile set kept by the ALL loader so both code paths produce
# DataFrames with the same column schema. P25/P75 are needed for true-quartile
# box edges in the Summary and Details boxplots.
_MC_PERCENTILES = [5, 10, 25, 50, 75, 90, 95]

# Map old building-attribute column names → new names on the Buildings sheet
_NEW_BLDG_COL_MAP = {
    'BuildingID':           'id',
    'OccupancyType':        'occupancy_type',
    'NumberOfStories':      'number_of_stories',
    'FoundationType':       'foundation_type',
    'FoundationHeight_ft':  'foundation_height',
    'GroundElevation_ft':   'ground_elevation',
    'StructureValue':       'structure_value',
    'ContentValue':         'content_value',
    # FFE_ft and Floodplain_Status already match the old schema
}


def _percentile_columns_from_mc(df_mc, mc_cols, prefix="CumEAD"):
    """Compute the percentile set in ``_MC_PERCENTILES`` (P05/P10/P25/P50/
    P75/P90/P95) across MC columns for every row. Returns a new DataFrame
    that replaces the MC columns with ``<prefix>_Pxx`` (zero-padded). Both
    quartile (P25/P75) and decile-tail (P05/P10/P90/P95) percentiles are
    included so downstream box-plots can use real stored quartiles instead
    of approximating them by linear-CDF interpolation."""
    arr = df_mc[mc_cols].to_numpy(dtype=float, na_value=np.nan)
    # np.nanpercentile handles the rare case where a row has all-NaN
    pct_arr = np.nanpercentile(arr, _MC_PERCENTILES, axis=1)
    # pct_arr has shape (len(percentiles), n_rows)
    out = df_mc.drop(columns=mc_cols).copy()
    for i, p in enumerate(_MC_PERCENTILES):
        out[f'{prefix}_P{p:02d}'] = pct_arr[i]
    return out


# Prefer the calamine engine for the new-format workbooks — for MC sheets
# with ~8,000 rows x ~1,000 columns it's ~10x faster than the default
# openpyxl reader and turns an app-startup that would otherwise take a minute
# into a few seconds. Falls back to openpyxl if calamine isn't installed.
try:
    import python_calamine  # noqa: F401
    _XLSX_ENGINE = 'calamine'
except Exception:
    _XLSX_ENGINE = None   # let pandas pick its default (openpyxl)


def _read_excel_fast(filepath, sheet_name):
    """pd.read_excel using the fastest engine available on this machine."""
    kwargs = {'sheet_name': sheet_name}
    if _XLSX_ENGINE:
        kwargs['engine'] = _XLSX_ENGINE
    return pd.read_excel(filepath, **kwargs)


# ----------------------------------------------------------------------------
# ALL-format (pre-aggregated percentiles) Excel loader
# ----------------------------------------------------------------------------
# The "_Results_ALL.xlsx" workbook is the lightweight successor to the
# "_Results_new.xlsx" Monte-Carlo workbook. The user has already collapsed the
# 1,000 MC realizations into 99 percentile columns (P01 … P99) directly in
# Excel, so the app no longer needs to perform that aggregation at load time.
# This makes the file ~10x smaller and the loader ~50x faster (no
# np.nanpercentile across 8,000 × 1,000 cells).
#
# Sheets in the ALL format:
#   * Metadata             — key/value pairs
#   * Buildings            — per-building static attributes
#   * WL_P50, WL_P90       — water-level realizations: 76 years × 99 percentiles
#   * Dmg_agg              — annual damage per (Year, Action, SLR) × 99 pct
#   * CumEAD_Categories    — cumulative damage per (Category, TargetYear,
#                            Action, SLR) × 99 pct
#   * bldg_CumEAD          — cumulative damage per (BuildingID, TargetYear,
#                            Action, SLR) × 99 pct
#
# Of the 99 percentiles only P05, P10, P50, P90, P95 are consumed downstream,
# so we drop the rest at load time to keep memory tidy.

# Percentile column names we extract from the ALL workbook. We keep the full
# set the rest of the app needs:
#   P05/P10/P50/P90/P95 — whiskers and shaded bands on box/CDF/timeline plots
#   P25/P75            — exact quartiles for box edges (used by community-level
#                        boxplots in the Summary tab and the per-building
#                        benefit boxplot in the Details tab). Without these,
#                        the app would fall back to linear-CDF interpolation
#                        between P05/P50/P95, which is correct only under a
#                        symmetric distribution and biases the Q1/Q3 edges by
#                        a few percent for the skewed tails we see in flood
#                        damages.
_ALL_PCT_KEEP = ['P05', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95']


@st.cache_data
def load_xlsx_file_all_format(filepath):
    """Load an 'ALL format' result workbook (pre-aggregated percentiles) and
    return the same dict-of-DataFrames the rest of the app already consumes:
    ``{'buildings': df, 'agg_by_occ': {occ: df}, 'bldg_attrs': df}``.

    Compared with :func:`load_xlsx_file_new_format`, this loader does NO
    Monte-Carlo aggregation — it just reads the pre-computed P05/P10/P50/P90/P95
    columns straight out of the workbook. That's the whole point of the ALL
    format: the heavy lifting was done once in Excel and shipped as data.
    """
    # ---------------------- Buildings descriptor table ---------------------
    bld = _read_excel_fast(filepath, 'Buildings')
    bld = bld.rename(columns={k: v for k, v in _NEW_BLDG_COL_MAP.items()
                              if k in bld.columns})
    if 'Floodplain_Status' in bld.columns:
        bld['Floodplain_Status'] = bld['Floodplain_Status'].apply(convert_floodplain_status)
    # Attributes the rest of the app references but that aren't in this file
    for col in ['building_type', 'area', 'year_built', 'address',
                'longitude', 'latitude']:
        if col not in bld.columns:
            bld[col] = np.nan

    # -------------------- Per-building cumulative damage --------------------
    bldg = _read_excel_fast(filepath, 'bldg_CumEAD')
    keep_cols = ['BuildingID', 'TargetYear', 'Action', 'SLR'] + _ALL_PCT_KEEP
    bldg = bldg[[c for c in keep_cols if c in bldg.columns]].copy()
    # Rename Pxx → CumEAD_Pxx and BuildingID → id to match the existing schema
    bldg = bldg.rename(columns={
        'BuildingID': 'id',
        **{p: f'CumEAD_{p}' for p in _ALL_PCT_KEEP},
    })
    attr_cols = [c for c in bld.columns if c != 'NSI_row']
    df_buildings = bldg.merge(bld[attr_cols], on='id', how='left')

    # ---------------------- Aggregated CumEAD (Categories) -----------------
    cat = _read_excel_fast(filepath, 'CumEAD_Categories')
    cat = cat[['Category', 'TargetYear', 'Action', 'SLR'] + _ALL_PCT_KEEP].copy()

    # Map each occupancy filter to:
    #   (top_level_category, [in-FP subcats], [out-FP subcats])
    # The top-level category gives us Total_CumEAD_Pxx directly. The subcat
    # lists give us InFP/OutFP_CumEAD_P50 by summing per-category P50s.
    # NB: Σmedian ≠ median(Σ) in general, but the discrepancy is small (<1%)
    # because the underlying MC realizations share the same hazard. Importantly,
    # this is exactly what the user has already accepted upstream — the whole
    # reason for shipping ALL-format data is that the user wants the app to
    # consume their pre-aggregated values without re-running the math.
    occ_to_cat = {
        'All':             ('ALL',    ['RES_InFP', 'NONRES_InFP'],
                                       ['RES_OutFP', 'NONRES_OutFP']),
        'Residential':     ('RES',    ['RES_InFP'],     ['RES_OutFP']),
        'Non-Residential': ('NONRES', ['NONRES_InFP'],  ['NONRES_OutFP']),
    }

    _is_res = (bld['occupancy_type'].apply(is_residential)
               if 'occupancy_type' in bld.columns
               else pd.Series(False, index=bld.index))
    bldg_counts = {
        'All':             int(len(bld)),
        'Residential':     int(_is_res.sum()),
        'Non-Residential': int((~_is_res).sum()),
    }

    def _split_p50_sum(categories, year, action, slr):
        """Sum of P50 across the named categories for the given filter.
        Returns 0.0 if no rows match (e.g. when an InFP slice is empty)."""
        sub = cat[(cat['TargetYear'] == year) &
                  (cat['Action'] == action) &
                  (cat['SLR'] == slr) &
                  (cat['Category'].isin(categories))]
        if sub.empty:
            return 0.0
        return float(sub['P50'].sum())

    agg_by_occ = {}
    for occ, (top_cat, infp_cats, outfp_cats) in occ_to_cat.items():
        top = cat[cat['Category'] == top_cat].copy()
        if top.empty:
            agg_by_occ[occ] = pd.DataFrame()
            continue
        top = top.rename(columns={p: f'Total_CumEAD_{p}' for p in _ALL_PCT_KEEP})
        top = top.drop(columns=['Category'])
        # ----------------------------------------------------------------
        # DFE split — reconcile to the community total
        # ----------------------------------------------------------------
        # The Under-DFE / Above-DFE P50 values come from summing per-subset
        # P50 medians (e.g. RES_InFP P50 + NONRES_InFP P50). The community
        # total P50 comes directly from the ALL category. Because
        #   median(A + B) ≠ median(A) + median(B)
        # in general, the two derivations don't agree exactly. On the
        # Shinnecock dataset the largest gap I measured was ~4.7%, small
        # but enough that the displayed split would not sum to the
        # displayed total. To keep the user-facing numbers self-consistent,
        # we rescale the InFP/OutFP P50 split so that
        #   InFP_P50 + OutFP_P50  ==  Total_P50
        # while preserving the InFP : OutFP ratio that the underlying
        # category medians imply.
        infp_raw  = top.apply(
            lambda r: _split_p50_sum(infp_cats,  r['TargetYear'], r['Action'], r['SLR']),
            axis=1,
        )
        outfp_raw = top.apply(
            lambda r: _split_p50_sum(outfp_cats, r['TargetYear'], r['Action'], r['SLR']),
            axis=1,
        )
        split_sum = infp_raw + outfp_raw
        total_p50 = top['Total_CumEAD_P50'].astype(float)
        # Where the raw split sum is positive, scale to match the total.
        # Where it's zero (no in-/out-FP buildings), pass through zeros.
        scale = np.where(split_sum > 0, total_p50 / split_sum, 0.0)
        top['InFP_CumEAD_P50']  = (infp_raw  * scale).astype(float)
        top['OutFP_CumEAD_P50'] = (outfp_raw * scale).astype(float)
        top['Num_Buildings'] = bldg_counts[occ]
        agg_by_occ[occ] = top.reset_index(drop=True)

    # -------------------- Water-level percentile sheets --------------------
    # WL_P50 / WL_P90 store annual-maximum stillwater level percentiles by
    # year (Year × P01..P99). The Water Levels tab visualizes these directly.
    # We keep the FULL P01..P99 grid here (rather than the small subset used
    # for damage box-plots) because the WL plots show shaded percentile
    # bands and CDFs that benefit from the dense percentile sampling.
    wl_data = {}
    for slr_key, sheet_name in (('50th-percentile', 'WL_P50'),
                                ('90th-percentile', 'WL_P90')):
        try:
            wl_df = _read_excel_fast(filepath, sheet_name)
            if 'Year' in wl_df.columns:
                wl_data[slr_key] = wl_df
        except Exception:
            # Sheet missing — continue silently; the Water Levels tab will
            # show a graceful "no data" message.
            pass

    return {
        'buildings':  df_buildings,
        'agg_by_occ': agg_by_occ,
        'bldg_attrs': bld,
        'water_levels': wl_data,
    }


@st.cache_data
def load_xlsx_file_new_format(filepath):
    """Load a 'new format' result workbook and return a dict of DataFrames
    ready for the app: per-building (with percentile columns) and per-occupancy
    aggregated (with percentile columns and DFE-split medians)."""
    # Don't hold ExcelFile open — with calamine each sheet is read as its own
    # pass, and it re-opens the file efficiently per call.
    # ---------------------- Buildings descriptor table ---------------------
    bld = _read_excel_fast(filepath, 'Buildings')
    bld = bld.rename(columns={k: v for k, v in _NEW_BLDG_COL_MAP.items()
                              if k in bld.columns})
    if 'Floodplain_Status' in bld.columns:
        bld['Floodplain_Status'] = bld['Floodplain_Status'].apply(convert_floodplain_status)
    # Attributes missing in new format but referenced elsewhere — create
    # NaN columns so downstream code can keep using .get(col, default)
    for col in ['building_type', 'area', 'year_built', 'address',
                'longitude', 'latitude']:
        if col not in bld.columns:
            bld[col] = np.nan

    # -------------------- Per-building cumulative damage --------------------
    bldg_mc = _read_excel_fast(filepath, 'bldg_CumEAD_MC')
    mc_cols_bldg = [c for c in bldg_mc.columns if c.startswith('MC_')]
    bldg_pct = _percentile_columns_from_mc(bldg_mc, mc_cols_bldg, prefix='CumEAD')
    bldg_pct = bldg_pct.rename(columns={'BuildingID': 'id'})

    # Join attributes onto every (id, TargetYear, Action, SLR) row
    attr_cols = [c for c in bld.columns if c != 'NSI_row']
    df_buildings = bldg_pct.merge(bld[attr_cols], on='id', how='left')

    # ---------------------- Aggregated CumEAD (Categories) -----------------
    cat_mc = _read_excel_fast(filepath, 'CumEAD_Categories')
    mc_cols_cat = [c for c in cat_mc.columns if c.startswith('MC_')]

    # Pre-compute percentiles for each Category row (drops MC columns)
    cat_pct = _percentile_columns_from_mc(cat_mc, mc_cols_cat, prefix='CumEAD')

    # DFE-split medians ideally come from summing In/Out-FP MCs across
    # RES+NONRES (for the "All" occupancy view), RES-only, or NONRES-only,
    # then taking the median of the summed MC vector. That's statistically
    # correct whereas summing per-building P50s would not be.
    # We do that in-memory using cat_mc (still with MCs) since we're loading
    # it once per workbook.
    def _split_median(categories, year, action, slr):
        """Median of the (MC-space) sum of `categories` for the given
        (year, action, slr). Returns 0 if no rows match."""
        sub = cat_mc[(cat_mc['TargetYear'] == year) &
                     (cat_mc['Action'] == action) &
                     (cat_mc['SLR'] == slr) &
                     (cat_mc['Category'].isin(categories))]
        if sub.empty:
            return 0.0
        arr = sub[mc_cols_cat].to_numpy(dtype=float, na_value=0.0).sum(axis=0)
        return float(np.nanmedian(arr))

    # Build the per-occupancy aggregated dataframes. For each occupancy
    # filter we pick the right top-level Category to derive the total
    # percentile columns, and assemble the InFP/OutFP medians from the
    # corresponding split categories.
    occ_to_cat = {
        'All':             ('ALL',   ['RES_InFP', 'NONRES_InFP'],
                                      ['RES_OutFP', 'NONRES_OutFP']),
        'Residential':     ('RES',    ['RES_InFP'],     ['RES_OutFP']),
        'Non-Residential': ('NONRES', ['NONRES_InFP'],  ['NONRES_OutFP']),
    }

    # Count the buildings in each occupancy bucket (for 'Num_Buildings')
    _is_res = bld['occupancy_type'].apply(is_residential) if 'occupancy_type' in bld.columns else pd.Series(False, index=bld.index)
    bldg_counts = {
        'All':             int(len(bld)),
        'Residential':     int(_is_res.sum()),
        'Non-Residential': int((~_is_res).sum()),
    }

    agg_by_occ = {}
    for occ, (top_cat, infp_cats, outfp_cats) in occ_to_cat.items():
        top = cat_pct[cat_pct['Category'] == top_cat].copy()
        if top.empty:
            agg_by_occ[occ] = pd.DataFrame()
            continue
        # Rename CumEAD_* → Total_CumEAD_* so the rest of the app keeps working.
        # We rename ALL percentile columns we computed (per _MC_PERCENTILES) so
        # the schema matches the ALL-format loader output exactly — including
        # P25/P75 used by box-plot quartile edges.
        top = top.rename(columns={
            f'CumEAD_P{p:02d}': f'Total_CumEAD_P{p:02d}'
            for p in _MC_PERCENTILES
        }).drop(columns=['Category'])
        # DFE-split medians (MC-space sum of relevant categories, then median).
        # This is the statistically correct "median of sum" — but it is still
        # not guaranteed that median(InFP)+median(OutFP) == median(All), so we
        # rescale below to keep the displayed split summing to the displayed
        # total (resolving Edit #3 from the user feedback).
        infp = top.apply(
            lambda r: _split_median(infp_cats, r['TargetYear'], r['Action'], r['SLR']),
            axis=1,
        )
        outfp = top.apply(
            lambda r: _split_median(outfp_cats, r['TargetYear'], r['Action'], r['SLR']),
            axis=1,
        )
        infp_v  = infp.astype(float).values
        outfp_v = outfp.astype(float).values
        split_sum = infp_v + outfp_v
        total_v = top['Total_CumEAD_P50'].astype(float).values
        scale = np.where(split_sum > 0, total_v / split_sum, 0.0)
        top['InFP_CumEAD_P50']  = infp_v  * scale
        top['OutFP_CumEAD_P50'] = outfp_v * scale
        top['Num_Buildings'] = bldg_counts[occ]
        agg_by_occ[occ] = top.reset_index(drop=True)

    # WL_P50 / WL_P90 — same role as in the ALL loader, see notes there.
    wl_data = {}
    for slr_key, sheet_name in (('50th-percentile', 'WL_P50'),
                                ('90th-percentile', 'WL_P90')):
        try:
            wl_df = _read_excel_fast(filepath, sheet_name)
            if 'Year' in wl_df.columns:
                wl_data[slr_key] = wl_df
        except Exception:
            pass

    return {
        'buildings':  df_buildings,
        'agg_by_occ': agg_by_occ,
        'bldg_attrs': bld,
        'water_levels': wl_data,
    }


@st.cache_data
def load_xlsx_file(filepath):
    """Load Excel file and return a dict of sheets (legacy format)."""
    xls = pd.ExcelFile(filepath)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = pd.read_excel(xls, sheet_name)
    return sheets


def _detect_workbook_format(filepath):
    """Identify the result-workbook variant by inspecting its sheet names.

    Returns one of:
      * ``'all'``    — pre-aggregated percentile workbook (``bldg_CumEAD``)
      * ``'mc'``     — Monte-Carlo realization workbook (``bldg_CumEAD_MC``)
      * ``'legacy'`` — older Aggregated/PerBuilding two-sheet layout
      * ``None``     — unrecognized / unreadable
    """
    try:
        xls = pd.ExcelFile(filepath)
    except Exception:
        return None
    sheet_names = set(xls.sheet_names)
    # The ALL format is checked BEFORE the MC format because both share the
    # 'Buildings' and 'CumEAD_Categories' sheets — the discriminator is
    # 'bldg_CumEAD' (ALL) vs 'bldg_CumEAD_MC' (MC).
    if {'Buildings', 'CumEAD_Categories', 'bldg_CumEAD'}.issubset(sheet_names):
        return 'all'
    if {'Buildings', 'CumEAD_Categories', 'bldg_CumEAD_MC'}.issubset(sheet_names):
        return 'mc'
    if {'Aggregated', 'PerBuilding'}.intersection(sheet_names):
        return 'legacy'
    return None


def _is_new_format(filepath):
    """Backward-compat shim — True for any non-legacy workbook."""
    return _detect_workbook_format(filepath) in ('all', 'mc')


def load_data_from_folder(data_folder="."):
    """Load all data files from the data folder. Supports three workbook
    layouts:

      * **ALL format** (preferred): pre-aggregated percentile columns
        (``bldg_CumEAD``, ``CumEAD_Categories`` with P01–P99). Loaded by
        :func:`load_xlsx_file_all_format` — fastest path.
      * **MC format** (legacy 'new'): raw Monte-Carlo realizations
        (``bldg_CumEAD_MC`` with 1,000 MC_xxxx columns). Loaded by
        :func:`load_xlsx_file_new_format` — falls back to per-load percentile
        computation.
      * **Legacy format**: ``Aggregated`` + ``PerBuilding`` two-sheet layout.

    When multiple workbooks exist for the same location the precedence is
    ``ALL > MC > legacy`` so the lightest, most pre-processed file wins.
    """
    data_store = {}

    if not os.path.exists(data_folder):
        return data_store, []

    available_locations = set()

    # Look for xlsx files first
    xlsx_files = glob.glob(os.path.join(data_folder, "*.xlsx")) + glob.glob(os.path.join(data_folder, "*.XLSX"))

    # Prefer the ALL workbook over the MC workbook over legacy. The filename
    # convention "_Results_ALL.xlsx" / "_Results_new.xlsx" makes this trivial,
    # but we sort by filename only — the actual format is determined by sheet
    # contents in _detect_workbook_format below.
    def _sort_key(path):
        name = os.path.basename(path).lower()
        if '_results_all' in name:
            rank = 0
        elif '_results_new' in name:
            rank = 1
        else:
            rank = 2
        return (rank, name)
    xlsx_files = sorted(xlsx_files, key=_sort_key)

    seen_new_format = set()   # locations already covered by an ALL or MC file

    for filepath in xlsx_files:
        filename = os.path.basename(filepath)
        location = parse_filename(filename)

        # If we already loaded a non-legacy file for this location, skip any
        # remaining file to avoid overwriting with less-precise/heavier data.
        if location in seen_new_format:
            continue

        available_locations.add(location)

        if location not in data_store:
            data_store[location] = {'agg': None, 'buildings': None,
                                    'agg_by_occ': None, 'format': None}

        fmt = _detect_workbook_format(filepath)

        if fmt == 'all':
            new_data = load_xlsx_file_all_format(filepath)
            data_store[location]['buildings'] = new_data['buildings']
            data_store[location]['agg_by_occ'] = new_data['agg_by_occ']
            data_store[location]['agg']        = new_data['agg_by_occ'].get('All')
            data_store[location]['format']     = 'all'
            seen_new_format.add(location)
            continue

        if fmt == 'mc':
            new_data = load_xlsx_file_new_format(filepath)
            data_store[location]['buildings'] = new_data['buildings']
            data_store[location]['agg_by_occ'] = new_data['agg_by_occ']
            data_store[location]['agg']        = new_data['agg_by_occ'].get('All')
            data_store[location]['format']     = 'new'
            seen_new_format.add(location)
            continue

        # --- Legacy summary format fallback ---
        sheets = load_xlsx_file(filepath)

        if 'Aggregated' in sheets:
            data_store[location]['agg'] = sheets['Aggregated']
        if 'PerBuilding' in sheets:
            df = sheets['PerBuilding']
            if 'Floodplain_Status' in df.columns:
                df['Floodplain_Status'] = df['Floodplain_Status'].apply(convert_floodplain_status)
            data_store[location]['buildings'] = df
        data_store[location]['format'] = 'legacy'

    # Fall back to csv files (very old legacy format)
    if not xlsx_files:
        csv_files = glob.glob(os.path.join(data_folder, "*.csv")) + glob.glob(os.path.join(data_folder, "*.CSV"))

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            location = parse_filename(filename)
            available_locations.add(location)

            if location not in data_store:
                data_store[location] = {'agg': None, 'buildings': None,
                                        'agg_by_occ': None, 'format': 'legacy'}

            df = load_csv_file(filepath)

            if 'CSV1' in filename.upper() or 'AGGREGATED' in filename.upper():
                data_store[location]['agg'] = df
            elif 'CSV2' in filename.upper() or 'PERBUILDING' in filename.upper() or 'PER_BUILDING' in filename.upper():
                if 'Floodplain_Status' in df.columns:
                    df['Floodplain_Status'] = df['Floodplain_Status'].apply(convert_floodplain_status)
                data_store[location]['buildings'] = df

    return data_store, sorted(list(available_locations))


def compute_damage_bin_breaks(df_buildings, scenario,
                              p_breaks=(0.20, 0.40, 0.60, 0.80),
                              thr=1000.0):
    """Compute stable damage-bin breakpoints across ALL years for the given
    SLR scenario. The breaks come from the pooled distribution of nonzero
    No-Mitigation P95 damages across every year, so the same building gets
    the same color regardless of which year is selected.
    
    This guarantees that switching the year on the Damage Bins map
    redistributes buildings *across* the same bins, rather than redefining
    the bins themselves — which makes year-to-year comparisons meaningful.
    
    Parameters
    ----------
    df_buildings : DataFrame
        Per-building, per-action, per-(year, SLR) records (already filtered
        by location and occupancy via the sidebar).
    scenario : str
        SLR scenario key, e.g. "50th-percentile" or "90th-percentile".
    p_breaks : tuple[float]
        Quantile points of the pooled nonzero-damage distribution at which
        to place breakpoints. Default 20/40/60/80.
    thr : float
        Damages below this value are treated as zero and excluded.
    
    Returns
    -------
    list[float] | None
        Up to len(p_breaks) sorted breakpoints, snapped to nice rounded
        values; returns None if no nonzero damages exist for this scenario.
    """
    df_nm = df_buildings[
        (df_buildings['Action'] == 'No mitigation') &
        (df_buildings['SLR'] == scenario)
    ]
    if df_nm.empty:
        return None
    
    pooled = df_nm['CumEAD_P95'].values.astype(float)
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
    # If snapping collapsed adjacent breaks, pad upward so we still have
    # the requested number of distinct levels.
    while len(unique_nice) < len(p_breaks) and unique_nice:
        last = unique_nice[-1]
        nxt = nice_round_up(last * 2.5)
        if nxt <= last:
            break
        unique_nice.append(nxt)
    return sorted(unique_nice)[:len(p_breaks)]


def prepare_map_data(df_buildings, target_year, scenario):
    """Prepare building data for map display."""
    df_filtered = df_buildings[
        (df_buildings['TargetYear'] == target_year) &
        (df_buildings['SLR'] == scenario)
    ].copy()
    
    if df_filtered.empty:
        return None
    
    attr_cols = [col for col in df_filtered.columns if col not in 
                 ['Action', 'CumEAD_P05', 'CumEAD_P50', 'CumEAD_P95', 'TargetYear', 'SLR']]
    
    df_base = df_filtered[df_filtered['Action'] == 'No mitigation'][attr_cols].copy()
    
    if df_base.empty:
        first_action = df_filtered['Action'].iloc[0]
        df_base = df_filtered[df_filtered['Action'] == first_action][attr_cols].copy()
    
    for action in df_filtered['Action'].unique():
        df_action = df_filtered[df_filtered['Action'] == action][['id', 'CumEAD_P05', 'CumEAD_P50', 'CumEAD_P95']].copy()
        df_action.columns = ['id', f'{action}_P05', f'{action}_P50', f'{action}_P95']
        df_base = df_base.merge(df_action, on='id', how='left')
    
    if 'Floodplain_Status' in df_base.columns:
        df_base['Floodplain_Status'] = df_base['Floodplain_Status'].apply(convert_floodplain_status)
    
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
        
        if 'Floodplain_Status' in df_action.columns:
            df_under = df_action[df_action['Floodplain_Status'] == 'Under DFE']
            df_above = df_action[df_action['Floodplain_Status'] == 'Above DFE']
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
    "carto-positron":   "© OpenStreetMap contributors • © CARTO",
    "carto-darkmatter": "© OpenStreetMap contributors • © CARTO",
    "stamen-terrain":   "© OpenStreetMap contributors • Stamen Design",
    "stamen-toner":     "© OpenStreetMap contributors • Stamen Design",
    "white-bg":         "",
}


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
                                 mapbox_style="open-street-map"):
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

    # ----- Title and footer in paper coordinates -----
    title_main = f"Building-level Flood Risk — {location}"
    if occupancy and occupancy != "All":
        title_main += f" ({occupancy})"
    title_sub = (f"Year {target_year}  •  {scenario_label}  •  "
                 f"{map_view}")
    today = _date.today().strftime("%B %Y")

    attrib = _BASEMAP_ATTRIB.get(mapbox_style, "© OpenStreetMap contributors")
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
            selected_location = st.selectbox(
                "📍 Location",
                options=available_locations,
                index=0
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

        # When a new-format workbook is loaded we have a pre-computed,
        # MC-correct aggregate per occupancy bucket — pick the one that
        # matches the sidebar selection rather than resumming per-building
        # percentiles (which would be statistically wrong).
        preloaded_agg = None
        if loc_entry is not None and loc_entry.get('agg_by_occ'):
            preloaded_agg = loc_entry['agg_by_occ'].get(selected_occupancy)
        
        st.divider()
        st.header("🎛️ Scenario Filters")
        
        available_years = [2040, 2055, 2100]
        if df_buildings is not None and 'TargetYear' in df_buildings.columns:
            available_years = sorted(df_buildings['TargetYear'].unique())
        elif df_agg_raw is not None and 'TargetYear' in df_agg_raw.columns:
            available_years = sorted(df_agg_raw['TargetYear'].unique())
        
        target_year = st.selectbox(
            "📅 Target Year",
            options=available_years,
            index=0
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
        
        if df_buildings is not None and 'Floodplain_Status' in df_buildings.columns:
            fp_options = df_buildings['Floodplain_Status'].dropna().unique().tolist()
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
            st.caption(f"**Buildings loaded:** {df_buildings['id'].nunique():,}")
    
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
    
    tab1, tab_dist, tab2, tab3, tab4 = st.tabs([
        "📊 Summary",
        "📦 Distributions",
        "🗺️ Map",
        "🏠 Details",
        "📈 Trends",
    ])
    
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
            
            if df_b_year.empty:
                st.warning(f"No per-building data for year {target_year}.")
            else:
                st.subheader(
                    f"Damage Distribution Across Buildings — {location_name} "
                    f"({occupancy_label}) — Year {target_year}"
                )
                
                # ----- Strategy ordering & labels -----
                action_order  = ['No mitigation', 'Raise Utilities', 'WFP B', 'WFP 1st', 'Elevate']
                action_labels_plain = {
                    'No mitigation':   'No Mitigation',
                    'Raise Utilities': 'Raise Utilities',
                    'WFP B':           'WFP Basement',
                    'WFP 1st':         'WFP 1st Floor',
                    'Elevate':         'Elevate',
                }
                actions_present = [a for a in action_order if a in df_b_year['Action'].unique()]
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

                        Returns 5-tuples ``(p10, p25, p50, p75, p90)`` per
                        group: whiskers reach to the 10th/90th percentiles of
                        damage across buildings (per the Distributions-tab
                        range convention), and box edges sit at the 25th/75th
                        percentiles of the same distribution. Computing P25
                        and P75 directly from the building values means the
                        box edges are real quartiles of the cross-building
                        distribution rather than CDF-linear approximations
                        between P10 and P90.
                        """
                        out = {slr_key: [] for slr_key, *_ in SCENARIO_SPECS}
                        for action in actions_present:
                            for slr_key, *_ in SCENARIO_SPECS:
                                vals = pb[(pb['Action'] == action) &
                                          (pb['SLR'] == slr_key)][stat_col].values
                                if len(vals) == 0:
                                    out[slr_key].append(None)
                                    continue
                                p10 = float(np.percentile(vals, 10))
                                p25 = float(np.percentile(vals, 25))
                                p50 = float(np.percentile(vals, 50))
                                p75 = float(np.percentile(vals, 75))
                                p90 = float(np.percentile(vals, 90))
                                # build_box_whisker_panel reads a 5-tuple as
                                # (lower-whisker, Q1, median, Q3, upper-whisker)
                                # — exactly the cross-building 10/25/50/75/90
                                # we just computed. No interpolation needed.
                                out[slr_key].append((p10, p25, p50, p75, p90))
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
                        lower_label="P10", upper_label="P90",
                        lower_pct=0.10, upper_pct=0.90,
                    )
                    fig_pb_right = build_box_whisker_panel(
                        group_labels=[action_labels_plain[a] for a in actions_present],
                        scenario_data=sd_p95,
                        panel_title=(
                            "Upper-tail (P95) per-building damage: distribution across "
                            f"{n_aff:,} damaged buildings"
                        ),
                        y_label="Per-Building Cumulative Damage",
                        lower_label="P10", upper_label="P90",
                        lower_pct=0.10, upper_pct=0.90,
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
                        "and whiskers extend to the 10th and 90th percentiles across buildings."
                    )
                
                st.divider()
                
                # =============================================================
                # PLOTS 3, 4, 5 — Per-building damage classification
                # Ported from VVV_Visualization_for_workshop_MasticBeach.py
                # Uses per-building P95 as upper-tail proxy (workshop uses P90).
                # =============================================================
                st.subheader(f"Building Counts by Adaptation Effectiveness — Year {target_year}")
                
                # Compute per-scenario stats (analog of compute_bldg_stats)
                per_scen_stats = {}
                for slr_key, slr_label, line_clr, _fill in SCENARIO_SPECS:
                    ds = df_b_year[df_b_year['SLR'] == slr_key]
                    if ds.empty:
                        per_scen_stats[slr_key] = None
                        continue
                    
                    d_nomit_s = ds[ds['Action'] == 'No mitigation'].set_index('id')
                    d_wfpb_s  = ds[ds['Action'] == 'WFP B'].set_index('id')
                    d_elev_s  = ds[ds['Action'] == 'Elevate'].set_index('id')
                    
                    n_tot_s = int(d_nomit_s.index.nunique()) if not d_nomit_s.empty else 0
                    if n_tot_s == 0:
                        per_scen_stats[slr_key] = None
                        continue
                    
                    ids_s = d_nomit_s.index
                    no_p50 = d_nomit_s['CumEAD_P50'].reindex(ids_s).fillna(0).values
                    no_p95 = d_nomit_s['CumEAD_P95'].reindex(ids_s).fillna(0).values
                    wb_p95 = (d_wfpb_s['CumEAD_P95'].reindex(ids_s).fillna(np.nan).values
                              if not d_wfpb_s.empty else None)
                    el_p95 = (d_elev_s['CumEAD_P95'].reindex(ids_s).fillna(np.nan).values
                              if not d_elev_s.empty else None)
                    
                    any_damage = no_p95 > thr
                    mask_p50   = no_p50 > thr
                    mask_sev   = no_p95 > thr
                    
                    mask_wfpb = np.zeros(n_tot_s, dtype=bool)
                    if wb_p95 is not None:
                        wb_eff = ~np.isnan(wb_p95) & (wb_p95 <= thr)
                        mask_wfpb = any_damage & wb_eff
                    
                    # --- Elevation-eliminates-P95 count (strict direct rule) ---
                    # A building counts here iff its own Elevate_P95 is at or
                    # below the threshold. We deliberately do NOT propagate
                    # WFP-Basement-success buildings into this bucket. The
                    # previous version did, on the rationale that elevation
                    # cannot be worse than WFP Basement — but the Shinnecock
                    # data refute that rationale: there are buildings where
                    # Elevate_P95 substantially exceeds WFP_B_P95 (and even
                    # WFP_1st_P95), presumably because of how elevation
                    # interacts with foundation type, content placement, and
                    # the depth–damage curve at high water levels. Using only
                    # the direct rule (a) keeps the chart honest about what
                    # the data actually say, and (b) makes the Distributions
                    # counts match the strict least-invasive classifier the
                    # Map tab uses (No Damage → WFP Basement → Elevation →
                    # Residual), so the same building doesn't get assigned
                    # one category here and a different one there.
                    mask_elev_works = np.zeros(n_tot_s, dtype=bool)
                    if el_p95 is not None:
                        elev_arr = np.where(np.isnan(el_p95), no_p95, el_p95)
                        mask_elev_works = any_damage & (elev_arr <= thr)
                    
                    per_scen_stats[slr_key] = {
                        'label':     slr_label,
                        'color':     line_clr,
                        'n_tot':     n_tot_s,
                        'n_p50_dmg': int(mask_p50.sum()),
                        'n_sev_dmg': int(mask_sev.sum()),
                        'n_damaged': int(any_damage.sum()),
                        'n_wfpb':    int(mask_wfpb.sum()),
                        'n_elev':    int(mask_elev_works.sum()),
                    }
                
                valid_stats = {k: v for k, v in per_scen_stats.items() if v is not None}
                if not valid_stats:
                    st.warning("No per-building data available for the selected year.")
                else:
                    n_tot_max = max(s['n_tot'] for s in valid_stats.values())
                    
                    def _make_paired_bar(title, value_fn, count_fn,
                                         x_left='Median damage > $0',
                                         x_right='P95 damage > $0',
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
                        x_right='Upper-tail (P95) damage > $0',
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Plots 4 & 5
                    def _v4(s):
                        nd = s['n_damaged']
                        return [100.0 * s['n_wfpb'] / nd if nd > 0 else 0]
                    def _c4(s): return [s['n_wfpb']]
                    fig4 = _make_paired_bar(
                        f"Damaged buildings where WFP Basement eliminates "
                        f"upper-tail damage by {target_year}",
                        _v4, _c4,
                        x_left='WFP Basement eliminates P95 damage',
                        single_group=True,
                    )
                    
                    def _v5(s):
                        nd = s['n_damaged']
                        return [100.0 * s['n_elev'] / nd if nd > 0 else 0]
                    def _c5(s): return [s['n_elev']]
                    fig5 = _make_paired_bar(
                        f"Damaged buildings where Elevation eliminates "
                        f"upper-tail damage by {target_year}",
                        _v5, _c5,
                        x_left='Elevation eliminates P95 damage',
                        single_group=True,
                    )
                    
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
                        tbl_rows.append({
                            'SLR Scenario':              s['label'],
                            'Buildings':                 f"{s['n_tot']:,}",
                            'Damaged (P95 > $0)':        f"{s['n_sev_dmg']:,}  ({100*s['n_sev_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "—",
                            'Damaged (median > $0)':     f"{s['n_p50_dmg']:,}  ({100*s['n_p50_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "—",
                            'WFP Basement eliminates P95':   f"{s['n_wfpb']:,}  ({100*s['n_wfpb']/nd:.1f}%)" if nd > 0 else "—",
                            'Elevation eliminates P95':      f"{s['n_elev']:,}  ({100*s['n_elev']/nd:.1f}%)" if nd > 0 else "—",
                        })
                    if tbl_rows:
                        st.markdown("**Per-scenario summary**")
                        st.dataframe(pd.DataFrame(tbl_rows),
                                     use_container_width=True, hide_index=True)
                    
                    st.caption(
                        "Per-building counts use the **P95** of the per-building cumulative damage as "
                        "the upper-tail proxy (the workshop visualization uses P90). "
                        "The **damaged-buildings chart** shows the share of buildings with median "
                        "damage greater than zero and the share with P95 damage greater than zero. "
                        "The **WFP Basement chart** shows, among buildings that experience any "
                        "damage, the share for which wet-floodproofing the basement brings P95 "
                        "damage to ≤ $1k. The **Elevation chart** shows the share for which "
                        "elevation alone brings P95 damage to ≤ $1k — the direct, strict rule, "
                        "which matches the Map tab's least-invasive classifier "
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
        
        # The map requires per-building longitude/latitude. The new-format
        # workbook doesn't carry those columns, so fall back to an info
        # message instead of rendering a broken map.
        _has_coords = (
            df_buildings is not None
            and {'longitude', 'latitude'}.issubset(df_buildings.columns)
            and df_buildings['latitude'].notna().any()
            and df_buildings['longitude'].notna().any()
        )

        if df_buildings is not None and not _has_coords:
            st.info(
                "🗺️ Map view is unavailable for this dataset — the result "
                "workbook does not include per-building longitude/latitude "
                "columns. All other tabs remain fully functional."
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
            
            df_map = prepare_map_data(df_buildings, target_year, scenario)
            
            if df_map is None or len(df_map) == 0:
                st.warning("No buildings match the current filters.")
            else:
                if dfe_filter and 'Floodplain_Status' in df_map.columns:
                    df_map = df_map[df_map['Floodplain_Status'].isin(dfe_filter)]
                
                # ----------------------------------------------------------
                # "Hide $0-damage buildings" — pick the right damage metric
                # ----------------------------------------------------------
                # The hide filter must look at the SAME statistic the active
                # map view colors by, otherwise we hide buildings the user
                # would expect to see:
                #   * Damage Heatmap        → P50 (what the heatmap colors)
                #   * Damage Bins           → P95 (the bins are upper-tail)
                #   * Adaptation Effective. → P95 (categories are P95-based)
                # In particular, for the bins/effectiveness views, hiding by
                # P50 silently drops every building with P50 = 0 but
                # P95 > $1k — the very buildings that drive tail-risk
                # planning. Many Shinnecock buildings fall in that bracket.
                if map_view == "Damage Heatmap":
                    zero_filter_col = 'No mitigation_P50' if 'No mitigation_P50' in df_map.columns else None
                else:
                    # P95 view — fall back to P50 only if P95 isn't loaded
                    zero_filter_col = (
                        'No mitigation_P95' if 'No mitigation_P95' in df_map.columns
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
                    hover_texts = []
                    for idx, row in df_map.iterrows():
                        text = f"<b>Building #{row['id']}</b><br>"
                        
                        if 'occupancy_type' in row:
                            text += f"Type: {row['occupancy_type']}<br>"
                        if 'structure_value' in row and pd.notna(row['structure_value']):
                            text += f"Structure Value: {format_currency(row['structure_value'])}<br>"
                        if 'Floodplain_Status' in row:
                            text += f"DFE Status: {row['Floodplain_Status']}<br>"
                        
                        text += "<br><b>━━━ Cumulative Damage ━━━</b><br>"
                        
                        baseline_val = row.get('No mitigation_P50', 0)
                        for col in action_cols_p50:
                            action_name = col.replace('_P50', '')
                            val = row.get(col, 0)
                            
                            display_name = action_name.replace('_', ' ')
                            if action_name == 'WFP B':
                                display_name = 'Wet Floodproof Basement'
                            elif action_name == 'WFP 1st':
                                display_name = 'Wet Floodproof 1st Floor'
                            elif action_name == 'Raise Utilities':
                                display_name = 'Raise Utilities'
                            
                            if action_name == 'No mitigation':
                                text += f"🔴 <b>{display_name}</b>: {format_currency(val)}<br>"
                            else:
                                savings = baseline_val - val if baseline_val > 0 else 0
                                pct = (savings / baseline_val * 100) if baseline_val > 0 else 0
                                if savings > 0:
                                    text += f"🟢 {display_name}: {format_currency(val)} <i>(-{pct:.0f}%)</i><br>"
                                else:
                                    text += f"⚪ {display_name}: {format_currency(val)}<br>"
                        
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
                                marker=dict(size=8, color='#22c55e', opacity=0.85),
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
                                    size=10,
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
                        # Required columns
                        col_nomit = 'No mitigation_P95'
                        col_wfpb  = 'WFP B_P95'
                        col_elev  = 'Elevate_P95'
                        
                        missing = [c for c in (col_nomit, col_wfpb, col_elev)
                                   if c not in df_map.columns]
                        if missing:
                            st.warning(
                                f"This view needs P95 columns for No mitigation, WFP B, and Elevate. "
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
                            
                            # --- MAP classifier (strict direct rule) ---
                            # The map shows, for each building, the LEAST-INVASIVE
                            # adaptation that eliminates P95 damage. The Elevation
                            # category is only assigned to buildings where WFP Basement
                            # is NOT sufficient but Elevation is. This is the same
                            # logic as the original MATLAB generate_action_animation.m.
                            wfpb_direct = wfpb <= thr
                            elev_direct = elev <= thr
                            
                            # Priority classification (controls both marker COLOR and
                            # the LEGEND count on the map):
                            #   1 = No Damage  > 2 = WFP Basement > 3 = Elevation > 4 = Residual
                            cat = np.full(len(df_map), 4, dtype=int)
                            cat[no_mit <= thr] = 1
                            cat[(no_mit > thr) & wfpb_direct] = 2
                            cat[(no_mit > thr) & ~wfpb_direct & elev_direct] = 3
                            
                            df_map['_cat_action'] = cat
                            
                            # Legend counts = strict-priority counts (each building
                            # appears in exactly one bucket).
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
                            
                            # Workshop palette (RGB normalized)
                            cat_specs = [
                                (1, 'No Damage',        '#22c55e'),  # green
                                (2, 'WFP Basement',     '#facc15'),  # yellow
                                (3, 'Elevation',        '#f97316'),  # orange
                                (4, 'Residual Damage',  '#dc2626'),  # red
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
                                        marker=dict(size=8, color=color, opacity=0.92),
                                        name=f"{label} ({legend_count})",
                                        showlegend=True, hoverinfo='skip',
                                    ))
                                    continue
                                _add_nonres_ring(fig_map, df_c, ring_size=13)
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=df_c['latitude'], lon=df_c['longitude'],
                                    mode='markers',
                                    marker=dict(size=8, color=color, opacity=0.92),
                                    hovertemplate='%{customdata[0]}<extra></extra>',
                                    customdata=list(df_c['hover_data']),
                                    name=f"{label} ({legend_count})",
                                ))
                            
                            # Add a legend-only trace for the non-residential ring marker
                            if df_map['_is_nonres'].any():
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=[None], lon=[None],
                                    mode='markers',
                                    marker=dict(size=10, color='black', opacity=0.85),
                                    name='Non-Residential (ringed)',
                                    showlegend=True, hoverinfo='skip',
                                ))
                    
                    # =====================================================
                    # VIEW 3 — Damage Bins (5 categories with dynamic breaks)
                    # Ported from generate_damage_animation_v3.m
                    # =====================================================
                    elif map_view == "Damage Bins":
                        col_nomit = 'No mitigation_P95'
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
                                _add_nonres_ring(fig_map, df_no_dmg, ring_size=13)
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=df_no_dmg['latitude'], lon=df_no_dmg['longitude'],
                                    mode='markers',
                                    marker=dict(size=8, color='#22c55e', opacity=0.85),
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
                                # nonzero No-Mit P95 damages over every year for
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
                                # bin edges for damaged buildings: thr, b1, b2, b3, b4, +inf  (5 bins)
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
                                    _add_nonres_ring(fig_map, df_c, ring_size=13)
                                    fig_map.add_trace(go.Scattermapbox(
                                        lat=df_c['latitude'], lon=df_c['longitude'],
                                        mode='markers',
                                        marker=dict(size=8, color=bin_colors[ci], opacity=0.92),
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
                                    "keeps the same color across 2040 / 2055 / 2100), but recompute when "
                                    "you switch SLR scenario. Buildings with no damage (≤ $1k) are shown in green."
                                )
                            
                            if df_map['_is_nonres'].any():
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=[None], lon=[None],
                                    mode='markers',
                                    marker=dict(size=10, color='black', opacity=0.85),
                                    name='Non-Residential (ringed)',
                                    showlegend=True, hoverinfo='skip',
                                ))
                    
                    # ---- Common layout ----
                    fig_map.update_layout(
                        mapbox=dict(
                            style="open-street-map",
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
                        st.caption(
                            "Each building is colored by the **least-invasive** adaptation that "
                            "eliminates its upper-tail (P95) cumulative damage under the selected "
                            "year and SLR scenario. Categories are checked in order: "
                            "**No Damage** (P95 baseline ≤ $1k) → "
                            "**WFP Basement** (wet floodproofing the basement brings P95 to ≤ $1k) → "
                            "**Elevation** (WFP Basement isn't sufficient but elevating the structure brings P95 to ≤ $1k) → "
                            "**Residual Damage** (even elevation doesn't bring P95 to ≤ $1k). "
                            "Each building appears in exactly one color, and the legend counts "
                            "partition the total — they don't overlap. Non-residential buildings are "
                            "marked with a black ring."
                        )
                    elif map_view == "Damage Bins":
                        st.caption(
                            "Each building is colored by its No-Mitigation P95 cumulative damage. " +
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
                        # Second row: basemap choice
                        export_basemap = st.selectbox(
                            "Basemap (printed export only)",
                            options=[
                                "open-street-map",
                                "carto-positron",
                                "carto-darkmatter",
                                "white-bg (no basemap)",
                            ],
                            index=1,   # carto-positron is the cleanest print default
                            key="map_export_basemap",
                            help="Tile-based basemaps need internet access at export time; "
                                 "if a tile server refuses the request the export will "
                                 "auto-fall-back to a white background. `carto-positron` "
                                 "(light, neutral) and `carto-darkmatter` (dark) are the "
                                 "preferred print styles. Pick `white-bg` for a guaranteed-"
                                 "working export with no roads/labels — useful when the "
                                 "underlying geography is provided by a separate base layer."
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
                    if 'Floodplain_Status' in df_map.columns:
                        display_cols.append('Floodplain_Status')
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
                    rename_map['Floodplain_Status'] = 'DFE Status'
                    top10 = top10.rename(columns=rename_map)
                    
                    st.dataframe(top10, use_container_width=True, hide_index=True)
        else:
            st.warning("No per-building data available for this location.")
    
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
                action_order_cs = ['No mitigation', 'Raise Utilities', 'WFP B', 'WFP 1st', 'Elevate']
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
                    under_dfe_data = []
                    baseline_infp = df_current[df_current['Action'] == 'No mitigation']['InFP_CumEAD_P50'].values
                    baseline_infp = baseline_infp[0] if len(baseline_infp) > 0 else 0
                    
                    for _, row in df_current.iterrows():
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
                    above_dfe_data = []
                    baseline_outfp = df_current[df_current['Action'] == 'No mitigation']['OutFP_CumEAD_P50'].values
                    baseline_outfp = baseline_outfp[0] if len(baseline_outfp) > 0 else 0
                    
                    for _, row in df_current.iterrows():
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
                                 'WFP B', 'WFP 1st', 'Elevate']
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
                
                building_dfe_status = building_info.get('Floodplain_Status', 'Unknown')
                is_above_dfe = building_dfe_status == 'Above DFE'
                
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
                        st.write(building_info.get('year_built', 'N/A'))
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
                    if 'Floodplain_Status' in building_info:
                        st.markdown("**DFE Status**")
                        fp_status = building_info.get('Floodplain_Status', 'N/A')
                        if fp_status == 'Under DFE':
                            st.error(fp_status)
                        else:
                            st.success(fp_status)
                
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
                    fig_building.update_xaxes(
                        showgrid=True, gridcolor='#e5e7eb',
                        tickmode='array', tickvals=sorted(df_traj['TargetYear'].unique()),
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
                    bd_action_order = ['No mitigation', 'Raise Utilities', 'WFP B', 'WFP 1st', 'Elevate']
                    bd_action_labels = {
                        'No mitigation':   'No Mitigation',
                        'Raise Utilities': 'Raise Utilities',
                        'WFP B':           'WFP Basement',
                        'WFP 1st':         'WFP 1st Floor',
                        'Elevate':         'Elevate',
                    }
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
                                "given strategy performs in 2040 vs 2055 vs 2100 — and how its benefit "
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
            trend_action_order = ['No mitigation', 'Raise Utilities', 'WFP B', 'WFP 1st', 'Elevate']
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
                fig_trend.update_xaxes(
                    title="Year", showgrid=True, gridcolor='#e5e7eb',
                    showline=True, linecolor='#cbd5e1',
                    tickmode='array', tickvals=sorted(df_slr['TargetYear'].unique()),
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
