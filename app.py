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

    /* Compact buttons - shorter overall height by trimming vertical padding.
       Applies to every st.button in the app for visual consistency. Width
       and font size are untouched, so button labels still fit on one line.
       `white-space: nowrap` on the inner paragraph forces the icon and the
       label to stay on a single horizontal line - without it, Streamlit
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

    /* Rounded corners on the app logo / brand icon in the sidebar. */
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* Axis text (tick labels and axis titles) rendered in solid black on every
       dynamic Plotly chart, so numbers and labels read clearly against the
       light background (e.g. the Damage distributions page). */
    [data-testid="stPlotlyChart"] .xtick text,
    [data-testid="stPlotlyChart"] .ytick text,
    [data-testid="stPlotlyChart"] .x2tick text,
    [data-testid="stPlotlyChart"] .y2tick text,
    [data-testid="stPlotlyChart"] text.xtitle,
    [data-testid="stPlotlyChart"] text.ytitle,
    [data-testid="stPlotlyChart"] .g-xtitle text,
    [data-testid="stPlotlyChart"] .g-ytitle text {
        fill: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ----------------------------------------------------------------------------
# Per-measure adaptation cost estimates.
#
# Source: the project workshop adaptation-measure table. Only the measure name
# (its first column) and the cost range (its 4th column, "Cost") are used here;
# the expected-benefit and remaining-recovery-cost columns from that table are
# intentionally NOT used. Values are keyed by the app's internal Action names:
#     "Elevate utilities"        -> Raise Utilities
#     "Wet-floodproof basement"  -> WFP B
#     "Wet-floodproof first floor" -> WFP 1st
#     "Elevate house"            -> Elevate
# ----------------------------------------------------------------------------
ADAPTATION_COST_ESTIMATES = {
    'Raise Utilities': '$5,000 - $40,000',
    'WFP B':           '$8,500 - $30,000',
    'WFP 1st':         '$10,000 - $100,000 or more',
    'Elevate':         'Per sq ft: $50 - $120 or more',
}
ADAPTATION_COST_SOURCE = (
    "Cost estimates per measure are taken from the project workshop "
    "adaptation-measure table (measure name and cost columns only)."
)


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
# buildings, the Elevate retrofit provides no further benefit - the
# data generator typically encodes this as a no-op, but we drop the
# row explicitly as a backstop so it doesn't slip through into hovers
# or charts due to numerical drift.
_ABOVE_DFE_STATUS_STRINGS = {'above dfe', 'above_dfe', 'abovedfe'}


def retrofit_applies(action, foundation_type=None, dfe_status=None):
    """Return True if `action` physically applies to a building with the
    given foundation type / DFE status.

    A retrofit that doesn't apply (e.g., wet-floodproofing a basement
    that doesn't exist, or elevating a building that's already above
    DFE) should not appear in the UI - its "damage" value is an
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
    'Above DFE' / 'Under DFE' - those pass through unchanged. Legacy
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
    decimal dropped when it's zero - e.g. $229.23M → $229.2M, $229.04M →
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


# Embedded copy of the NSI Field Survey tool (app2), base64-encoded so app.py
# is fully self-contained (no external HTML asset required). The tool is the
# standalone React/Leaflet survey app, reworked so its controls sit in a
# compact top toolbar and its building-detail form stays as a right panel.
# A sibling `nsi_tool.html` on disk, if present, OVERRIDES this embed - handy
# for editing the tool without regenerating the blob below.
_NSI_TOOL_HTML_B64 = """
PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0iVVRGLTgiPgogIDxtZXRhIG5h
bWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxl
PTEuMCwgdXNlci1zY2FsYWJsZT1ubyI+CiAgPHRpdGxlPk5TSSBGaWVsZCBTdXJ2ZXkgVG9vbDwvdGl0bGU+CiAgPGxpbmsgcmVs
PSJpY29uIiBocmVmPSJkYXRhOmltYWdlL3N2Zyt4bWwsPHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZp
ZXdCb3g9JzAgMCAxMDAgMTAwJz48dGV4dCB5PScuOWVtJyBmb250LXNpemU9JzkwJz7wn4+XPC90ZXh0Pjwvc3ZnPiI+CiAgPHNj
cmlwdCBjcm9zc29yaWdpbiBzcmM9Imh0dHBzOi8vdW5wa2cuY29tL3JlYWN0QDE4L3VtZC9yZWFjdC5wcm9kdWN0aW9uLm1pbi5q
cyI+PC9zY3JpcHQ+CiAgPHNjcmlwdCBjcm9zc29yaWdpbiBzcmM9Imh0dHBzOi8vdW5wa2cuY29tL3JlYWN0LWRvbUAxOC91bWQv
cmVhY3QtZG9tLnByb2R1Y3Rpb24ubWluLmpzIj48L3NjcmlwdD4KICA8c2NyaXB0IGNyb3Nzb3JpZ2luIHNyYz0iaHR0cHM6Ly91
bnBrZy5jb20vQGJhYmVsL3N0YW5kYWxvbmVANy4yNi40L2JhYmVsLm1pbi5qcyI+PC9zY3JpcHQ+CiAgPHNjcmlwdCBjcm9zc29y
aWdpbiBzcmM9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL3hsc3gvMC4xOC41L3hsc3guZnVsbC5taW4u
anMiPjwvc2NyaXB0PgogIDxzdHlsZT4KICAgICogeyBtYXJnaW46IDA7IHBhZGRpbmc6IDA7IGJveC1zaXppbmc6IGJvcmRlci1i
b3g7IH0KICAgIGh0bWwsIGJvZHksICNyb290IHsgaGVpZ2h0OiAxMDAlOyBvdmVyZmxvdzogaGlkZGVuOyB9CiAgICAvKiBFbWJl
ZGRlZCBpbiBTdHJlYW1saXQgdmlhIGNvbXBvbmVudHMuaHRtbDogdGhlIGlmcmFtZSBzZXRzIGEgZml4ZWQKICAgICAgIHBpeGVs
IGhlaWdodCwgYW5kICNyb290IGZpbGxzIGl0LiBUaGUgYXBwJ3Mgb3duIGZsZXggY29sdW1uIHRoZW4KICAgICAgIHNwbGl0cyB0
aGF0IGhlaWdodCBpbnRvIHRoZSB0b3AgY29udHJvbCBiYXIgKyB0aGUgbWFwL3JpZ2h0LXBhbmVsIHJvdy4gKi8KICA8L3N0eWxl
Pgo8L2hlYWQ+Cjxib2R5PgogIDxkaXYgaWQ9InJvb3QiPjwvZGl2PgogIDxzY3JpcHQgdHlwZT0idGV4dC9iYWJlbCIgZGF0YS1w
cmVzZXRzPSJyZWFjdCI+Cgpjb25zdCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QsIHVzZVJlZiwgdXNlQ2FsbGJhY2sgfSA9IFJlYWN0
OwoKCgoKCgpjb25zdCBMT0NBVElPTlMgPSB7CiAgc2hpbm5lY29jazogeyBuYW1lOiAiU2hpbm5lY29jayIsIGNlbnRlcjogWzQw
Ljg3NywgLTcyLjQzMV0sIHpvb206IDE0IH0sCiAgbWFzdGljYmVhY2g6IHsgbmFtZTogIk1hc3RpYyBCZWFjaCIsIGNlbnRlcjog
WzQwLjc2MSwgLTcyLjg0OF0sIHpvb206IDEzIH0sCiAgcGFtdW5rZXk6IHsgbmFtZTogIlBhbXVua2V5IiwgY2VudGVyOiBbMzcu
NTc2LCAtNzcuMDA0XSwgem9vbTogMTQgfSwKICB3ZXN0cG9pbnQ6IHsgbmFtZTogIldlc3QgUG9pbnQiLCBjZW50ZXI6IFszNy41
NTIsIC03Ni44MDFdLCB6b29tOiAxNCB9LAp9OwoKY29uc3QgT0NDX1RZUEVTID0gWwogIHsgY29kZTogIlJFUzEiLCBsYWJlbDog
IlJFUzEg4oCUIFNpbmdsZSBGYW1pbHkgRHdlbGxpbmciIH0sCiAgeyBjb2RlOiAiUkVTMiIsIGxhYmVsOiAiUkVTMiDigJQgTWFu
dWZhY3R1cmVkIEhvdXNpbmciIH0sCiAgeyBjb2RlOiAiUkVTM0EiLCBsYWJlbDogIlJFUzNBIOKAlCBNdWx0aSBGYW1pbHkgKER1
cGxleCkiIH0sCiAgeyBjb2RlOiAiUkVTM0IiLCBsYWJlbDogIlJFUzNCIOKAlCBNdWx0aSBGYW1pbHkgKDPigJM0IFVuaXRzKSIg
fSwKICB7IGNvZGU6ICJSRVMzQyIsIGxhYmVsOiAiUkVTM0Mg4oCUIE11bHRpIEZhbWlseSAoNeKAkzkgVW5pdHMpIiB9LAogIHsg
Y29kZTogIlJFUzNEIiwgbGFiZWw6ICJSRVMzRCDigJQgTXVsdGkgRmFtaWx5ICgxMOKAkzE5IFVuaXRzKSIgfSwKICB7IGNvZGU6
ICJSRVMzRSIsIGxhYmVsOiAiUkVTM0Ug4oCUIE11bHRpIEZhbWlseSAoMjDigJM0OSBVbml0cykiIH0sCiAgeyBjb2RlOiAiUkVT
M0YiLCBsYWJlbDogIlJFUzNGIOKAlCBNdWx0aSBGYW1pbHkgKDUwKyBVbml0cykiIH0sCiAgeyBjb2RlOiAiUkVTNCIsIGxhYmVs
OiAiUkVTNCDigJQgVGVtcG9yYXJ5IExvZGdpbmciIH0sCiAgeyBjb2RlOiAiUkVTNSIsIGxhYmVsOiAiUkVTNSDigJQgSW5zdGl0
dXRpb25hbCBEb3JtaXRvcnkiIH0sCiAgeyBjb2RlOiAiUkVTNiIsIGxhYmVsOiAiUkVTNiDigJQgTnVyc2luZyBIb21lIiB9LAog
IHsgY29kZTogIkNPTTEiLCBsYWJlbDogIkNPTTEg4oCUIFJldGFpbCBUcmFkZSIgfSwKICB7IGNvZGU6ICJDT00yIiwgbGFiZWw6
ICJDT00yIOKAlCBXaG9sZXNhbGUgVHJhZGUiIH0sCiAgeyBjb2RlOiAiQ09NMyIsIGxhYmVsOiAiQ09NMyDigJQgUGVyc29uYWwg
JiBSZXBhaXIgU2VydmljZXMiIH0sCiAgeyBjb2RlOiAiQ09NNCIsIGxhYmVsOiAiQ09NNCDigJQgUHJvZmVzc2lvbmFsL1RlY2hu
aWNhbCIgfSwKICB7IGNvZGU6ICJDT001IiwgbGFiZWw6ICJDT001IOKAlCBCYW5rcy9GaW5hbmNpYWwiIH0sCiAgeyBjb2RlOiAi
Q09NNiIsIGxhYmVsOiAiQ09NNiDigJQgSG9zcGl0YWwiIH0sCiAgeyBjb2RlOiAiQ09NNyIsIGxhYmVsOiAiQ09NNyDigJQgTWVk
aWNhbCBPZmZpY2UvQ2xpbmljIiB9LAogIHsgY29kZTogIkNPTTgiLCBsYWJlbDogIkNPTTgg4oCUIEVudGVydGFpbm1lbnQgJiBS
ZWNyZWF0aW9uIiB9LAogIHsgY29kZTogIkNPTTkiLCBsYWJlbDogIkNPTTkg4oCUIFRoZWF0ZXJzIiB9LAogIHsgY29kZTogIkNP
TTEwIiwgbGFiZWw6ICJDT00xMCDigJQgUGFya2luZyIgfSwKICB7IGNvZGU6ICJJTkQxIiwgbGFiZWw6ICJJTkQxIOKAlCBIZWF2
eSBJbmR1c3RyaWFsIiB9LAogIHsgY29kZTogIklORDIiLCBsYWJlbDogIklORDIg4oCUIExpZ2h0IEluZHVzdHJpYWwiIH0sCiAg
eyBjb2RlOiAiSU5EMyIsIGxhYmVsOiAiSU5EMyDigJQgRm9vZC9EcnVncy9DaGVtaWNhbHMiIH0sCiAgeyBjb2RlOiAiSU5ENCIs
IGxhYmVsOiAiSU5ENCDigJQgTWV0YWxzL01pbmVyYWxzIFByb2Nlc3NpbmciIH0sCiAgeyBjb2RlOiAiSU5ENSIsIGxhYmVsOiAi
SU5ENSDigJQgSGlnaCBUZWNobm9sb2d5IiB9LAogIHsgY29kZTogIklORDYiLCBsYWJlbDogIklORDYg4oCUIENvbnN0cnVjdGlv
biIgfSwKICB7IGNvZGU6ICJBR1IxIiwgbGFiZWw6ICJBR1IxIOKAlCBBZ3JpY3VsdHVyZSIgfSwKICB7IGNvZGU6ICJSRUwxIiwg
bGFiZWw6ICJSRUwxIOKAlCBDaHVyY2gvTm9uLVByb2ZpdCIgfSwKICB7IGNvZGU6ICJHT1YxIiwgbGFiZWw6ICJHT1YxIOKAlCBH
ZW5lcmFsIEdvdmVybm1lbnQiIH0sCiAgeyBjb2RlOiAiR09WMiIsIGxhYmVsOiAiR09WMiDigJQgRW1lcmdlbmN5IFJlc3BvbnNl
IiB9LAogIHsgY29kZTogIkVEVTEiLCBsYWJlbDogIkVEVTEg4oCUIFNjaG9vbHMgKEvigJMxMikiIH0sCiAgeyBjb2RlOiAiRURV
MiIsIGxhYmVsOiAiRURVMiDigJQgQ29sbGVnZXMvVW5pdmVyc2l0aWVzIiB9LApdOwoKY29uc3QgRU1QVFlfRk9STSA9IHsKICBu
dW1TdG9yaWVzOiAiIiwgZmlyc3RGbG9vckhlaWdodDogIiIsIGFkZHJlc3M6ICIiLAogIG9jY1R5cGU6ICIiLCBmb3VuZGF0aW9u
VHlwZTogIiIsIGJ1aWxkaW5nVHlwZTogIiIsIGFyZWE6ICIiLAogIHllYXJCdWlsdDogIiIsIGdyb3VuZEVsZXY6ICIiLCBzdHJ1
Y3R1cmVWYWx1ZTogIiIsIGNvbnRlbnRWYWx1ZTogIiIsCiAgbm90ZXM6ICIiLCBzdXJ2ZXlvcjogIiIsIGZsYWdnZWQ6ICIiLAp9
OwoKLy8gR29vZ2xlIFNoZWV0cyBiYWNrZW5kCi8vIOKaoO+4jyBQQVNURSBZT1VSIERFUExPWUVEIEdPT0dMRSBBUFBTIFNDUklQ
VCBXRUIgQVBQIFVSTCBCRUxPVzoKY29uc3QgQVBJX1VSTCA9ICJodHRwczovL3NjcmlwdC5nb29nbGUuY29tL21hY3Jvcy9zL0FL
ZnljYnhJaWFSakdSVjFfYy1mMXoxTGdHUFVIWV9YRmhkYlVNdjRicEMwMVk1Z285RHRZbTJHMkFhUk9ubG16SnVVOVgxVy9leGVj
IjsKCi8vIOKaoO+4jyBQQVNURSBZT1VSIEdPT0dMRSBTSEVFVCBVUkwgQkVMT1c6CmNvbnN0IFNIRUVUX1VSTCA9ICJodHRwczov
L2RvY3MuZ29vZ2xlLmNvbS9zcHJlYWRzaGVldHMvZC8xM0ZqSVIzVy0yRU5xUGRrSVZWSmprRk5USUlvWl9QYWZ4OTUxNlNCcE5O
US9lZGl0P3VzcD1zaGFyaW5nIjsKCi8vIOKaoO+4jyBQQVNURSBZT1VSIEdJVEhVQiBSRVBPIFVSTCBCRUxPVyAoZS5nLiwgImh0
dHBzOi8vZ2l0aHViLmNvbS91c2VybmFtZS9idWlsZGluZ3Mtc3VydmV5Iik6CmNvbnN0IFJFQURNRV9VUkwgPSAiaHR0cHM6Ly9n
aXRodWIuY29tL2VyZmFuLWFtaW5pL2J1aWxkaW5ncy1zdXJ2ZXkjcmVhZG1lIjsKCi8vIOKUgOKUgOKUgCBSb2J1c3QgZmV0Y2gg
aGVscGVyOiB0aHJvd3Mgb24gbm9uLU9LIHJlc3BvbnNlcyDilIDilIDilIAKYXN5bmMgZnVuY3Rpb24gcG9zdEpzb24ocGF5bG9h
ZCkgewogIGNvbnN0IHIgPSBhd2FpdCBmZXRjaChBUElfVVJMLCB7CiAgICBtZXRob2Q6ICJQT1NUIiwKICAgIGJvZHk6IEpTT04u
c3RyaW5naWZ5KHBheWxvYWQpLAogIH0pOwogIGlmICghci5vaykgewogICAgY29uc3QgdGV4dCA9IGF3YWl0IHIudGV4dCgpLmNh
dGNoKCgpID0+ICIiKTsKICAgIHRocm93IG5ldyBFcnJvcihgSFRUUCAke3Iuc3RhdHVzfTogJHt0ZXh0LnNsaWNlKDAsIDIwMCl9
YCk7CiAgfQogIGNvbnN0IGN0ID0gci5oZWFkZXJzLmdldCgiY29udGVudC10eXBlIikgfHwgIiI7CiAgcmV0dXJuIGN0LmluY2x1
ZGVzKCJhcHBsaWNhdGlvbi9qc29uIikgPyByLmpzb24oKSA6IHIudGV4dCgpOwp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaFN1cnZl
eXMobG9jYXRpb24pIHsKICB0cnkgewogICAgY29uc3QgciA9IGF3YWl0IGZldGNoKEFQSV9VUkwgKyAiP2xvY2F0aW9uPSIgKyBl
bmNvZGVVUklDb21wb25lbnQobG9jYXRpb24gfHwgInNoaW5uZWNvY2siKSk7CiAgICBpZiAoIXIub2spIHsKICAgICAgY29uc3Qg
dGV4dCA9IGF3YWl0IHIudGV4dCgpLmNhdGNoKCgpID0+ICIiKTsKICAgICAgdGhyb3cgbmV3IEVycm9yKGBIVFRQICR7ci5zdGF0
dXN9OiAke3RleHQuc2xpY2UoMCwgMjAwKX1gKTsKICAgIH0KICAgIGNvbnN0IHJhdyA9IGF3YWl0IHIuanNvbigpOwogICAgaWYg
KHJhdy5lcnJvcikgdGhyb3cgbmV3IEVycm9yKHJhdy5lcnJvcik7CiAgICAvLyBNYXAgR29vZ2xlIFNoZWV0IGNvbHVtbiBuYW1l
cyB0byBmb3JtIGZpZWxkIG5hbWVzCiAgICBjb25zdCBtYXBwZWQgPSB7fTsKICAgIGZvciAoY29uc3QgdWlkIGluIHJhdykgewog
ICAgICBjb25zdCBzID0gcmF3W3VpZF07CiAgICAgIG1hcHBlZFt1aWRdID0gewogICAgICAgIG9jY1R5cGU6IHMub2NjdXBhbmN5
X3R5cGUgfHwgcy5vY2NUeXBlIHx8ICIiLAogICAgICAgIGJ1aWxkaW5nVHlwZTogcy5idWlsZGluZ190eXBlIHx8IHMuYnVpbGRp
bmdUeXBlIHx8ICIiLAogICAgICAgIG51bVN0b3JpZXM6IFN0cmluZyhzLm51bWJlcl9vZl9zdG9yaWVzIHx8IHMubnVtU3Rvcmll
cyB8fCAiIiksCiAgICAgICAgYXJlYTogU3RyaW5nKHMuYXJlYSB8fCAiIiksCiAgICAgICAgZm91bmRhdGlvblR5cGU6IHMuZm91
bmRhdGlvbl90eXBlIHx8IHMuZm91bmRhdGlvblR5cGUgfHwgIiIsCiAgICAgICAgZmlyc3RGbG9vckhlaWdodDogU3RyaW5nKHMu
Zm91bmRhdGlvbl9oZWlnaHQgfHwgcy5maXJzdEZsb29ySGVpZ2h0IHx8ICIiKSwKICAgICAgICB5ZWFyQnVpbHQ6IFN0cmluZyhz
LnllYXJfYnVpbHQgfHwgcy55ZWFyQnVpbHQgfHwgIiIpLAogICAgICAgIGdyb3VuZEVsZXY6IFN0cmluZyhzLmdyb3VuZF9lbGV2
YXRpb24gfHwgcy5ncm91bmRFbGV2IHx8ICIiKSwKICAgICAgICBhZGRyZXNzOiBzLmFkZHJlc3MgfHwgIiIsCiAgICAgICAgc3Ry
dWN0dXJlVmFsdWU6IFN0cmluZyhzLnN0cnVjdHVyZV92YWx1ZSB8fCBzLnN0cnVjdHVyZVZhbHVlIHx8ICIiKSwKICAgICAgICBj
b250ZW50VmFsdWU6IFN0cmluZyhzLmNvbnRlbnRfdmFsdWUgfHwgcy5jb250ZW50VmFsdWUgfHwgIiIpLAogICAgICAgIGJhc2Vt
ZW50OiBzLmJhc2VtZW50IHx8ICIiLAogICAgICAgIG5vdGVzOiBzLm5vdGVzIHx8ICIiLAogICAgICAgIHN1cnZleW9yOiBzLnN1
cnZleW9yIHx8ICIiLAogICAgICAgIHNhdmVkQXQ6IHMuc2F2ZWRBdCB8fCAiIiwKICAgICAgICBmbGFnZ2VkOiAoKCkgPT4geyBj
b25zdCB2ID0gU3RyaW5nKHMuZmxhZ2dlZCB8fCAiIikudHJpbSgpOyBpZiAoL15kZW1vbGlzaGVkJC9pLnRlc3QodikpIHJldHVy
biAiRGVtb2xpc2hlZCI7IGlmICgvXih5ZXN8dHJ1ZXwxKSQvaS50ZXN0KHYpKSByZXR1cm4gIlllcyI7IHJldHVybiAiIjsgfSko
KSwKICAgICAgICBsb25naXR1ZGU6IHMubG9uZ2l0dWRlIHx8ICIiLAogICAgICAgIGxhdGl0dWRlOiBzLmxhdGl0dWRlIHx8ICIi
LAogICAgICAgIHN1cnZleV90eXBlOiBzLnN1cnZleV90eXBlIHx8ICIiLAogICAgICAgIElEOiBzLklEIHx8ICIiLAogICAgICB9
OwogICAgfQogICAgcmV0dXJuIHsgZGF0YTogbWFwcGVkLCBlcnJvcjogbnVsbCB9OwogIH0gY2F0Y2goZSkgewogICAgY29uc29s
ZS5lcnJvcigiRmV0Y2ggc3VydmV5cyBmYWlsZWQ6IiwgZSk7CiAgICByZXR1cm4geyBkYXRhOiB7fSwgZXJyb3I6IGUubWVzc2Fn
ZSB9OwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gc2F2ZVN1cnZleUVudHJ5KHVpZCwgc3VydmV5VHlwZSwgbnNpSWQsIGxuZywgbGF0
LCBmb3JtRGF0YSwgbG9jYXRpb24pIHsKICBjb25zdCByZXNwID0gYXdhaXQgcG9zdEpzb24oewogICAgYWN0aW9uOiAic2F2ZSIs
CiAgICBsb2NhdGlvbjogbG9jYXRpb24gfHwgInNoaW5uZWNvY2siLAogICAgZGF0YTogewogICAgICB1aWQsCiAgICAgIHN1cnZl
eV90eXBlOiBzdXJ2ZXlUeXBlLAogICAgICBJRDogbnNpSWQgfHwgIiIsCiAgICAgIG9jY3VwYW5jeV90eXBlOiBmb3JtRGF0YS5v
Y2NUeXBlIHx8ICIiLAogICAgICBidWlsZGluZ190eXBlOiBmb3JtRGF0YS5idWlsZGluZ1R5cGUgfHwgIiIsCiAgICAgIG51bWJl
cl9vZl9zdG9yaWVzOiBmb3JtRGF0YS5udW1TdG9yaWVzIHx8ICIiLAogICAgICBhcmVhOiBmb3JtRGF0YS5hcmVhIHx8ICIiLAog
ICAgICBmb3VuZGF0aW9uX3R5cGU6IGZvcm1EYXRhLmZvdW5kYXRpb25UeXBlIHx8ICIiLAogICAgICBmb3VuZGF0aW9uX2hlaWdo
dDogZm9ybURhdGEuZmlyc3RGbG9vckhlaWdodCB8fCAiIiwKICAgICAgeWVhcl9idWlsdDogZm9ybURhdGEueWVhckJ1aWx0IHx8
ICIiLAogICAgICBncm91bmRfZWxldmF0aW9uOiBmb3JtRGF0YS5ncm91bmRFbGV2IHx8ICIiLAogICAgICBhZGRyZXNzOiBmb3Jt
RGF0YS5hZGRyZXNzIHx8ICIiLAogICAgICBsb25naXR1ZGU6IGxuZywKICAgICAgbGF0aXR1ZGU6IGxhdCwKICAgICAgc3RydWN0
dXJlX3ZhbHVlOiBmb3JtRGF0YS5zdHJ1Y3R1cmVWYWx1ZSB8fCAiIiwKICAgICAgY29udGVudF92YWx1ZTogZm9ybURhdGEuY29u
dGVudFZhbHVlIHx8ICIiLAogICAgICBiYXNlbWVudDogKGZvcm1EYXRhLmZvdW5kYXRpb25UeXBlIHx8ICIiKS50b1VwcGVyQ2Fz
ZSgpID09PSAiQiIgPyAiWWVzIiA6IChmb3JtRGF0YS5iYXNlbWVudCB8fCAiTm8iKSwKICAgICAgbm90ZXM6IGZvcm1EYXRhLm5v
dGVzIHx8ICIiLAogICAgICBzdXJ2ZXlvcjogZm9ybURhdGEuc3VydmV5b3IgfHwgIiIsCiAgICAgIHNhdmVkQXQ6IGZvcm1EYXRh
LnNhdmVkQXQgfHwgIiIsCiAgICAgIGZsYWdnZWQ6IGZvcm1EYXRhLmZsYWdnZWQgfHwgIiIsCiAgICB9CiAgfSk7CiAgaWYgKHJl
c3AuZXJyb3IpIHRocm93IG5ldyBFcnJvcihyZXNwLmVycm9yKTsKICByZXR1cm4gcmVzcDsKfQoKYXN5bmMgZnVuY3Rpb24gdXBk
YXRlRmxhZ0VudHJ5KHVpZCwgZmxhZ2dlZCwgbG9jYXRpb24pIHsKICB0cnkgewogICAgY29uc3QgcmVzcCA9IGF3YWl0IHBvc3RK
c29uKHsKICAgICAgYWN0aW9uOiAidXBkYXRlRmxhZyIsCiAgICAgIGxvY2F0aW9uOiBsb2NhdGlvbiB8fCAibWFzdGljYmVhY2gi
LAogICAgICB1aWQsCiAgICAgIGZsYWdnZWQ6IGZsYWdnZWQgfHwgIiIsCiAgICB9KTsKICAgIGlmIChyZXNwLmVycm9yICYmIHJl
c3AuZXJyb3IuaW5jbHVkZXMoIlVua25vd24gYWN0aW9uIikpIHsKICAgICAgY29uc29sZS53YXJuKCJ1cGRhdGVGbGFnIG5vdCBz
dXBwb3J0ZWQgYnkgYmFja2VuZCwgZmxhZyB3aWxsIHBlcnNpc3Qgb24gbmV4dCBTYXZlIik7CiAgICAgIHJldHVybiB7IG9rOiB0
cnVlLCBmYWxsYmFjazogdHJ1ZSB9OwogICAgfQogICAgaWYgKHJlc3AuZXJyb3IgJiYgcmVzcC5lcnJvci5pbmNsdWRlcygiVUlE
IG5vdCBmb3VuZCIpKSB7CiAgICAgIC8vIFJvdyBkb2Vzbid0IGV4aXN0IHlldCDigJQgbGV0IGNhbGxlciBoYW5kbGUgYnkgY3Jl
YXRpbmcgdGhlIHJvdwogICAgICByZXR1cm4geyBvazogZmFsc2UsIGVycm9yOiByZXNwLmVycm9yLCBuZWVkc1JvdzogdHJ1ZSB9
OwogICAgfQogICAgaWYgKHJlc3AuZXJyb3IpIHRocm93IG5ldyBFcnJvcihyZXNwLmVycm9yKTsKICAgIHJldHVybiByZXNwOwog
IH0gY2F0Y2ggKGVycikgewogICAgdGhyb3cgZXJyOwogIH0KfQoKYXN5bmMgZnVuY3Rpb24gZGVsZXRlU3VydmV5RW50cnkodWlk
LCBsb2NhdGlvbikgewogIGNvbnN0IHJlc3AgPSBhd2FpdCBwb3N0SnNvbih7IGFjdGlvbjogImRlbGV0ZSIsIHVpZCwgbG9jYXRp
b246IGxvY2F0aW9uIHx8ICJzaGlubmVjb2NrIiB9KTsKICBpZiAocmVzcC5lcnJvcikgdGhyb3cgbmV3IEVycm9yKHJlc3AuZXJy
b3IpOwogIHJldHVybiByZXNwOwp9Cgphc3luYyBmdW5jdGlvbiBmZXRjaERldkVkaXRzKGxvY2F0aW9uKSB7CiAgdHJ5IHsKICAg
IGNvbnN0IHJlc3AgPSBhd2FpdCBwb3N0SnNvbih7IGFjdGlvbjogImdldERldiIsIGxvY2F0aW9uOiBsb2NhdGlvbiB8fCAic2hp
bm5lY29jayIgfSk7CiAgICBpZiAocmVzcC5lcnJvcikgdGhyb3cgbmV3IEVycm9yKHJlc3AuZXJyb3IpOwogICAgcmV0dXJuIHJl
c3A7CiAgfSBjYXRjaChlKSB7IGNvbnNvbGUuZXJyb3IoIkZldGNoIGRldiBlZGl0cyBmYWlsZWQ6IiwgZSk7IHJldHVybiB7IHJl
bW92ZWQ6IFtdLCBtb3ZlZDoge30sIGFkZGVkOiBbXSB9OyB9Cn0KCmFzeW5jIGZ1bmN0aW9uIHNhdmVEZXZFZGl0c1JlbW90ZShk
LCBsb2NhdGlvbikgewogIGNvbnN0IHJlc3AgPSBhd2FpdCBwb3N0SnNvbih7IGFjdGlvbjogInNhdmVEZXYiLCBkYXRhOiBkLCBs
b2NhdGlvbjogbG9jYXRpb24gfHwgInNoaW5uZWNvY2siIH0pOwogIGlmIChyZXNwLmVycm9yKSB0aHJvdyBuZXcgRXJyb3IocmVz
cC5lcnJvcik7CiAgcmV0dXJuIHJlc3A7Cn0KCi8vIOKUgOKUgOKUgCBNaWNyb3NvZnQgQnVpbGRpbmcgRm9vdHByaW50cyAoQXJj
R0lTKSDilIDilIDilIAKY29uc3QgTVNCRlBfVVJMID0gImh0dHBzOi8vc2VydmljZXMuYXJjZ2lzLmNvbS9QM2VQTE1ZczJSVkNo
a0p4L2FyY2dpcy9yZXN0L3NlcnZpY2VzL01TQkZQMi9GZWF0dXJlU2VydmVyLzAvcXVlcnkiOwoKYXN5bmMgZnVuY3Rpb24gZmV0
Y2hCdWlsZGluZ0Zvb3RwcmludEFyZWEobG5nLCBsYXQpIHsKICBjb25zdCBwYXJhbXMgPSBuZXcgVVJMU2VhcmNoUGFyYW1zKHsK
ICAgIGdlb21ldHJ5OiBgJHtsbmd9LCR7bGF0fWAsCiAgICBnZW9tZXRyeVR5cGU6ICJlc3JpR2VvbWV0cnlQb2ludCIsCiAgICBz
cGF0aWFsUmVsOiAiZXNyaVNwYXRpYWxSZWxJbnRlcnNlY3RzIiwKICAgIHJldHVybkdlb21ldHJ5OiAidHJ1ZSIsCiAgICBvdXRG
aWVsZHM6ICIqIiwKICAgIGluU1I6ICI0MzI2IiwKICAgIG91dFNSOiAiNDMyNiIsCiAgICBmOiAianNvbiIsCiAgfSk7CiAgY29u
c3QgciA9IGF3YWl0IGZldGNoKGAke01TQkZQX1VSTH0/JHtwYXJhbXN9YCk7CiAgaWYgKCFyLm9rKSB0aHJvdyBuZXcgRXJyb3Io
YEFyY0dJUyBIVFRQICR7ci5zdGF0dXN9YCk7CiAgY29uc3QgZGF0YSA9IGF3YWl0IHIuanNvbigpOwogIGlmICghZGF0YS5mZWF0
dXJlcyB8fCBkYXRhLmZlYXR1cmVzLmxlbmd0aCA9PT0gMCkgcmV0dXJuIG51bGw7CgogIC8vIENvbXB1dGUgYXJlYSBmb3IgZWFj
aCBtYXRjaGluZyBwb2x5Z29uLCBwaWNrIHRoZSBzbWFsbGVzdCAobW9zdCBzcGVjaWZpYyBidWlsZGluZykKICBmdW5jdGlvbiBj
YWxjUmluZ3NBcmVhKHJpbmdzKSB7CiAgICBsZXQgdG90YWxTcU0gPSAwOwogICAgZm9yIChsZXQgcmkgPSAwOyByaSA8IHJpbmdz
Lmxlbmd0aDsgcmkrKykgewogICAgICBjb25zdCByaW5nID0gcmluZ3NbcmldOwogICAgICBjb25zdCBtaWRMYXQgPSByaW5nLnJl
ZHVjZSgocywgcCkgPT4gcyArIHBbMV0sIDApIC8gcmluZy5sZW5ndGg7CiAgICAgIGNvbnN0IGRlZ0xuZzJtID0gTWF0aC5jb3Mo
bWlkTGF0ICogTWF0aC5QSSAvIDE4MCkgKiAxMTEzMjA7CiAgICAgIGNvbnN0IGRlZ0xhdDJtID0gMTEwNTQwOwogICAgICBsZXQg
YXJlYSA9IDA7CiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcmluZy5sZW5ndGg7IGkrKykgewogICAgICAgIGNvbnN0IGogPSAo
aSArIDEpICUgcmluZy5sZW5ndGg7CiAgICAgICAgY29uc3QgeGkgPSByaW5nW2ldWzBdICogZGVnTG5nMm0sIHlpID0gcmluZ1tp
XVsxXSAqIGRlZ0xhdDJtOwogICAgICAgIGNvbnN0IHhqID0gcmluZ1tqXVswXSAqIGRlZ0xuZzJtLCB5aiA9IHJpbmdbal1bMV0g
KiBkZWdMYXQybTsKICAgICAgICBhcmVhICs9IHhpICogeWogLSB4aiAqIHlpOwogICAgICB9CiAgICAgIGNvbnN0IHNpZ25lZEFy
ZWEgPSBhcmVhIC8gMjsKICAgICAgdG90YWxTcU0gKz0gKHJpID09PSAwKSA/IE1hdGguYWJzKHNpZ25lZEFyZWEpIDogLU1hdGgu
YWJzKHNpZ25lZEFyZWEpOwogICAgfQogICAgcmV0dXJuIE1hdGgubWF4KDAsIHRvdGFsU3FNKTsKICB9CgogIGxldCBiZXN0U3FN
ID0gSW5maW5pdHk7CiAgZm9yIChjb25zdCBmZWF0IG9mIGRhdGEuZmVhdHVyZXMpIHsKICAgIGNvbnN0IHJpbmdzID0gZmVhdC5n
ZW9tZXRyeSAmJiBmZWF0Lmdlb21ldHJ5LnJpbmdzOwogICAgaWYgKCFyaW5ncyB8fCByaW5ncy5sZW5ndGggPT09IDApIGNvbnRp
bnVlOwogICAgY29uc3Qgc3FtID0gY2FsY1JpbmdzQXJlYShyaW5ncyk7CiAgICBpZiAoc3FtID4gMCAmJiBzcW0gPCBiZXN0U3FN
KSBiZXN0U3FNID0gc3FtOwogIH0KICBpZiAoIWlzRmluaXRlKGJlc3RTcU0pIHx8IGJlc3RTcU0gPD0gMCkgcmV0dXJuIG51bGw7
CiAgcmV0dXJuIE1hdGgucm91bmQoYmVzdFNxTSAqIDEwLjc2MzkpOwp9CgovLyDilIDilIDilIAgVVNHUyAzREVQIEVsZXZhdGlv
biBQb2ludCBRdWVyeSAoTkFWRDg4LCAxbSBsaWRhciB3aGVyZSBhdmFpbGFibGUpIOKUgOKUgOKUgApjb25zdCBVU0dTX0VQUVNf
VVJMID0gImh0dHBzOi8vZXBxcy5uYXRpb25hbG1hcC5nb3YvdjEvanNvbiI7Cgphc3luYyBmdW5jdGlvbiBmZXRjaFVTR1NFbGV2
YXRpb24obG5nLCBsYXQpIHsKICBjb25zdCBwYXJhbXMgPSBuZXcgVVJMU2VhcmNoUGFyYW1zKHsKICAgIHg6IFN0cmluZyhsbmcp
LAogICAgeTogU3RyaW5nKGxhdCksCiAgICB3a2lkOiAiNDMyNiIsCiAgICB1bml0czogIkZlZXQiLAogICAgaW5jbHVkZURhdGU6
ICJmYWxzZSIsCiAgfSk7CiAgY29uc3QgciA9IGF3YWl0IGZldGNoKGAke1VTR1NfRVBRU19VUkx9PyR7cGFyYW1zfWApOwogIGlm
ICghci5vaykgdGhyb3cgbmV3IEVycm9yKGBVU0dTIEhUVFAgJHtyLnN0YXR1c31gKTsKICBjb25zdCBkYXRhID0gYXdhaXQgci5q
c29uKCk7CiAgLy8gRVBRUyB2MSByZXR1cm5zIHsgdmFsdWU6IDxudW1iZXI+IH0gb3IgeyB2YWx1ZTogLTEwMDAwMDAgfSBmb3Ig
b2NlYW4vdm9pZAogIGNvbnN0IGVsZXYgPSBwYXJzZUZsb2F0KGRhdGEudmFsdWUpOwogIGlmICghaXNGaW5pdGUoZWxldikgfHwg
ZWxldiA8IC0xMDAwKSByZXR1cm4gbnVsbDsKICByZXR1cm4gTWF0aC5yb3VuZChlbGV2ICogMTAwKSAvIDEwMDsgLy8gMiBkZWNp
bWFsIHBsYWNlcywgaW4gZmVldCBOQVZEODgKfQoKLy8g4pSA4pSA4pSAIENvc3QgRXN0aW1hdG9yIChPTFMgcmVncmVzc2lvbiBv
biBsb2NhbCBidWlsZGluZyBzdG9jaykg4pSA4pSA4pSACi8vIFVzZXMgYWxsIGJ1aWxkaW5ncyBpbiB0aGUgY3VycmVudCBsb2Nh
dGlvbiB3aXRoIGtub3duIGdyb3NzIGFyZWEgYW5kCi8vIHN0cnVjdHVyZSB2YWx1ZSB0byBmaXQ6IHN0cnVjdHVyZVZhbHVlID0g
c2xvcGUgw5cgZ3Jvc3NBcmVhICsgaW50ZXJjZXB0Ci8vIFRoZW4gcHJlZGljdHMgZm9yIHRoZSB0YXJnZXQgYnVpbGRpbmcuIENv
bnRlbnQgPSBzdHJ1Y3R1cmUgLyAyLgovLyBNaW5pbXVtIDUgcmVmZXJlbmNlIHBvaW50cyByZXF1aXJlZDsgZmFsbHMgYmFjayB0
byBtZWRpYW4gJC9zcWZ0IG90aGVyd2lzZS4KCmZ1bmN0aW9uIGVzdGltYXRlQ29zdFJlZ3Jlc3Npb24oZm9vdHByaW50QXJlYSwg
c3RvcmllcywgYWxsU3VydmV5cywgYWxsQnVpbGRpbmdzLCBvY2NUeXBlKSB7CiAgY29uc3QgZnAgPSBwYXJzZUZsb2F0KGZvb3Rw
cmludEFyZWEpIHx8IDA7CiAgY29uc3QgcyA9IE1hdGgubWF4KDEsIE1hdGgucm91bmQocGFyc2VGbG9hdChzdG9yaWVzKSB8fCAx
KSk7CiAgY29uc3QgZ3Jvc3NUYXJnZXQgPSBmcCAqIHM7CiAgaWYgKGdyb3NzVGFyZ2V0IDw9IDApIHJldHVybiBudWxsOwoKICAv
LyBDb2xsZWN0IHJlZmVyZW5jZSBkYXRhOiBidWlsZGluZ3Mgd2l0aCBrbm93biBhcmVhLCBzdG9yaWVzLCBhbmQgc3RydWN0dXJl
IHZhbHVlCiAgY29uc3QgY29sbGVjdFJlZnMgPSAoZmlsdGVyKSA9PiB7CiAgICBjb25zdCByZWZzID0gW107CiAgICBhbGxCdWls
ZGluZ3MuZm9yRWFjaChiID0+IHsKICAgICAgY29uc3Qgc3YgPSBhbGxTdXJ2ZXlzW2IudWlkXTsKICAgICAgY29uc3QgcGYgPSBi
LnByZWZpbGwgfHwge307CiAgICAgIGNvbnN0IGQgPSBzdiB8fCBwZjsKICAgICAgaWYgKGZpbHRlciAmJiAhKGQub2NjVHlwZSB8
fCBwZi5vY2NUeXBlIHx8ICIiKS50b1VwcGVyQ2FzZSgpLnN0YXJ0c1dpdGgoZmlsdGVyKSkgcmV0dXJuOwogICAgICBjb25zdCBh
cmVhID0gcGFyc2VGbG9hdChkLmFyZWEgfHwgcGYuYXJlYSk7CiAgICAgIGNvbnN0IHN0ID0gcGFyc2VGbG9hdChkLm51bVN0b3Jp
ZXMgfHwgcGYubnVtU3Rvcmllcyk7CiAgICAgIGNvbnN0IHZhbCA9IHBhcnNlRmxvYXQoZC5zdHJ1Y3R1cmVWYWx1ZSB8fCBwZi5z
dHJ1Y3R1cmVWYWx1ZSk7CiAgICAgIGlmIChpc0Zpbml0ZShhcmVhKSAmJiBhcmVhID4gMCAmJiBpc0Zpbml0ZShzdCkgJiYgc3Qg
Pj0gMSAmJiBpc0Zpbml0ZSh2YWwpICYmIHZhbCA+IDApIHsKICAgICAgICByZWZzLnB1c2goeyBncm9zczogYXJlYSAqIE1hdGgu
cm91bmQoc3QpLCB2YWwgfSk7CiAgICAgIH0KICAgIH0pOwogICAgcmV0dXJuIHJlZnM7CiAgfTsKCiAgLy8gVHJ5IHNhbWUgb2Nj
dXBhbmN5IHByZWZpeCBmaXJzdCAoZS5nLiwgIlJFUyIsICJDT00iKSwgZmFsbCBiYWNrIHRvIGFsbAogIGNvbnN0IG9jY1ByZWZp
eCA9IChvY2NUeXBlIHx8ICIiKS50b1VwcGVyQ2FzZSgpLnJlcGxhY2UoL1swLTldLiovLCIiKTsKICBsZXQgcmVmcyA9IG9jY1By
ZWZpeCA/IGNvbGxlY3RSZWZzKG9jY1ByZWZpeCkgOiBbXTsKICBjb25zdCB1c2VkRmlsdGVyID0gcmVmcy5sZW5ndGggPj0gMzsK
ICBpZiAocmVmcy5sZW5ndGggPCAzKSByZWZzID0gY29sbGVjdFJlZnMobnVsbCk7CiAgaWYgKHJlZnMubGVuZ3RoIDwgMikgcmV0
dXJuIG51bGw7CgogIC8vIFByb3BlciBzdGF0aXN0aWNhbCBtZWRpYW4KICBmdW5jdGlvbiBtZWRpYW4oYXJyKSB7CiAgICBjb25z
dCBzb3J0ZWQgPSBbLi4uYXJyXS5zb3J0KChhLGIpID0+IGEgLSBiKTsKICAgIGNvbnN0IG1pZCA9IE1hdGguZmxvb3Ioc29ydGVk
Lmxlbmd0aCAvIDIpOwogICAgcmV0dXJuIHNvcnRlZC5sZW5ndGggJSAyID09PSAwID8gKHNvcnRlZFttaWQtMV0gKyBzb3J0ZWRb
bWlkXSkgLyAyIDogc29ydGVkW21pZF07CiAgfQoKICAvLyBPTFM6IHkgPSBzbG9wZSAqIHggKyBpbnRlcmNlcHQKICBjb25zdCBu
ID0gcmVmcy5sZW5ndGg7CiAgbGV0IHN4ID0gMCwgc3kgPSAwLCBzeHkgPSAwLCBzeHggPSAwOwogIGZvciAoY29uc3QgciBvZiBy
ZWZzKSB7IHN4ICs9IHIuZ3Jvc3M7IHN5ICs9IHIudmFsOyBzeHkgKz0gci5ncm9zcyAqIHIudmFsOyBzeHggKz0gci5ncm9zcyAq
IHIuZ3Jvc3M7IH0KICBjb25zdCBkZW5vbSA9IG4gKiBzeHggLSBzeCAqIHN4OwoKICBsZXQgc3YsIG1ldGhvZDsKICBpZiAoTWF0
aC5hYnMoZGVub20pIDwgMWUtMTAgfHwgbiA8IDUpIHsKICAgIGNvbnN0IG1lZFBzZiA9IG1lZGlhbihyZWZzLm1hcChyID0+IHIu
dmFsIC8gci5ncm9zcykpOwogICAgc3YgPSBNYXRoLnJvdW5kKGdyb3NzVGFyZ2V0ICogbWVkUHNmKTsKICAgIG1ldGhvZCA9ICJt
ZWRpYW4iOwogIH0gZWxzZSB7CiAgICBjb25zdCBzbG9wZSA9IChuICogc3h5IC0gc3ggKiBzeSkgLyBkZW5vbTsKICAgIGNvbnN0
IGludGVyY2VwdCA9IChzeSAtIHNsb3BlICogc3gpIC8gbjsKICAgIHN2ID0gTWF0aC5yb3VuZChzbG9wZSAqIGdyb3NzVGFyZ2V0
ICsgaW50ZXJjZXB0KTsKICAgIG1ldGhvZCA9ICJPTFMiOwogICAgaWYgKHN2IDwgZ3Jvc3NUYXJnZXQgKiAxMCkgewogICAgICBj
b25zdCBtZWRQc2YgPSBtZWRpYW4ocmVmcy5tYXAociA9PiByLnZhbCAvIHIuZ3Jvc3MpKTsKICAgICAgc3YgPSBNYXRoLnJvdW5k
KGdyb3NzVGFyZ2V0ICogbWVkUHNmKTsKICAgICAgbWV0aG9kID0gIm1lZGlhbiI7CiAgICB9CiAgfQoKICByZXR1cm4geyBzdHJ1
Y3R1cmU6IHN2LCBjb250ZW50OiBNYXRoLnJvdW5kKHN2IC8gMiksIHJlZkNvdW50OiBuLCBtZXRob2QsIGZpbHRlcmVkOiB1c2Vk
RmlsdGVyLCBvY2NQcmVmaXggfTsKfQoKZnVuY3Rpb24gYXBwbHlEZXZFZGl0cyhiYXNlLCBlZGl0cykgewogIGxldCBsaXN0ID0g
YmFzZS5maWx0ZXIoYiA9PiAhZWRpdHMucmVtb3ZlZC5pbmNsdWRlcyhiLnVpZCkpOwogIGxpc3QgPSBsaXN0Lm1hcChiID0+IHsK
ICAgIGlmIChlZGl0cy5tb3ZlZFtiLnVpZF0pIHJldHVybiB7IC4uLmIsIGxhdDogZWRpdHMubW92ZWRbYi51aWRdLmxhdCwgbG5n
OiBlZGl0cy5tb3ZlZFtiLnVpZF0ubG5nIH07CiAgICByZXR1cm4gYjsKICB9KTsKICAvLyBBZGQgZGV2LWFkZGVkIHBvaW50cywg
c2tpcHBpbmcgYW55IHRoYXQgYWxyZWFkeSBleGlzdCBpbiBiYXNlIChmcm9tIHNoZWV0IHJvdyBhdXRvLWNyZWF0aW9uKQogIGNv
bnN0IGJhc2VVaWRzID0gbmV3IFNldChsaXN0Lm1hcChiID0+IGIudWlkKSk7CiAgZWRpdHMuYWRkZWQuZm9yRWFjaChhID0+IHsK
ICAgIGlmIChiYXNlVWlkcy5oYXMoYS51aWQpKSByZXR1cm47IC8vIGFscmVhZHkgaW4gc2hlZXQsIHNraXAgZHVwbGljYXRlCiAg
ICAvLyBBcHBseSBtb3ZlZCBjb29yZGluYXRlcyB0byBhZGRlZCBwb2ludHMgdG9vCiAgICBpZiAoZWRpdHMubW92ZWRbYS51aWRd
KSB7CiAgICAgIGxpc3QucHVzaCh7IC4uLmEsIGxhdDogZWRpdHMubW92ZWRbYS51aWRdLmxhdCwgbG5nOiBlZGl0cy5tb3ZlZFth
LnVpZF0ubG5nIH0pOwogICAgfSBlbHNlIHsKICAgICAgbGlzdC5wdXNoKGEpOwogICAgfQogIH0pOwogIHJldHVybiBsaXN0Owp9
CgovLyBTaGFyZWQgaGVscGVyOiBidWlsZCBidWlsZGluZ3MgYXJyYXkgZnJvbSBzaGVldCBkYXRhLCBza2lwcGluZyBpbnZhbGlk
IGNvb3JkcwpmdW5jdGlvbiBzaGVldERhdGFUb0J1aWxkaW5ncyhzaGVldERhdGEpIHsKICByZXR1cm4gT2JqZWN0LmVudHJpZXMo
c2hlZXREYXRhKQogICAgLm1hcCgoW3VpZCwgc10pID0+IHsKICAgICAgY29uc3QgbG5nID0gcGFyc2VGbG9hdChzLmxvbmdpdHVk
ZSk7CiAgICAgIGNvbnN0IGxhdCA9IHBhcnNlRmxvYXQocy5sYXRpdHVkZSk7CiAgICAgIGlmICghaXNGaW5pdGUobG5nKSB8fCAh
aXNGaW5pdGUobGF0KSB8fCAobG5nID09PSAwICYmIGxhdCA9PT0gMCkpIHJldHVybiBudWxsOwogICAgICByZXR1cm4gewogICAg
ICAgIHVpZCwKICAgICAgICB0eXBlOiBzLnN1cnZleV90eXBlIHx8ICh1aWQuc3RhcnRzV2l0aCgibnNpLSIpID8gInZlcmlmeSIg
OiAic3VydmV5IiksCiAgICAgICAgbnNpSWQ6IHMuSUQgfHwgIiIsCiAgICAgICAgbG5nLCBsYXQsCiAgICAgICAgcHJlZmlsbDog
ewogICAgICAgICAgb2NjVHlwZTogcy5vY2NUeXBlIHx8ICIiLCBidWlsZGluZ1R5cGU6IHMuYnVpbGRpbmdUeXBlIHx8ICIiLAog
ICAgICAgICAgbnVtU3Rvcmllczogcy5udW1TdG9yaWVzIHx8ICIiLCBhcmVhOiBzLmFyZWEgfHwgIiIsCiAgICAgICAgICBmb3Vu
ZGF0aW9uVHlwZTogcy5mb3VuZGF0aW9uVHlwZSB8fCAiIiwgZmlyc3RGbG9vckhlaWdodDogcy5maXJzdEZsb29ySGVpZ2h0IHx8
ICIiLAogICAgICAgICAgeWVhckJ1aWx0OiBzLnllYXJCdWlsdCB8fCAiIiwgZ3JvdW5kRWxldjogcy5ncm91bmRFbGV2IHx8ICIi
LAogICAgICAgICAgYWRkcmVzczogcy5hZGRyZXNzIHx8ICIiLAogICAgICAgICAgc3RydWN0dXJlVmFsdWU6IHMuc3RydWN0dXJl
VmFsdWUgfHwgIiIsIGNvbnRlbnRWYWx1ZTogcy5jb250ZW50VmFsdWUgfHwgIiIsCiAgICAgICAgICBiYXNlbWVudDogcy5iYXNl
bWVudCB8fCAiIiwgbm90ZXM6IHMubm90ZXMgfHwgIiIsIHN1cnZleW9yOiBzLnN1cnZleW9yIHx8ICIiLAogICAgICAgICAgZmxh
Z2dlZDogcy5mbGFnZ2VkIHx8ICIiLAogICAgICAgIH0sCiAgICAgIH07CiAgICB9KQogICAgLmZpbHRlcihCb29sZWFuKTsKfQoK
Ly8gU2hhcmVkIGhlbHBlcjogbG9hZCBhbmQgbWVyZ2UgZGV2IGVkaXRzIGZyb20gbG9jYWwgKyByZW1vdGUKYXN5bmMgZnVuY3Rp
b24gbG9hZERldkVkaXRzKGxvY2F0aW9uKSB7CiAgbGV0IGJlc3QgPSB7IHJlbW92ZWQ6IFtdLCBtb3ZlZDoge30sIGFkZGVkOiBb
XSwgX3RzOiAwIH07CiAgLy8gQ2hlY2sgbG9jYWxTdG9yYWdlCiAgdHJ5IHsKICAgIGNvbnN0IHJhdyA9IGxvY2FsU3RvcmFnZS5n
ZXRJdGVtKCJuc2ktZGV2LWVkaXRzLSIgKyBsb2NhdGlvbik7CiAgICBpZiAocmF3KSB7CiAgICAgIGNvbnN0IGxvY2FsID0gSlNP
Ti5wYXJzZShyYXcpOwogICAgICBpZiAoIWxvY2FsLnJlbW92ZWQpIGxvY2FsLnJlbW92ZWQgPSBbXTsKICAgICAgaWYgKCFsb2Nh
bC5tb3ZlZCkgbG9jYWwubW92ZWQgPSB7fTsKICAgICAgaWYgKCFsb2NhbC5hZGRlZCkgbG9jYWwuYWRkZWQgPSBbXTsKICAgICAg
aWYgKCFsb2NhbC5fdHMpIGxvY2FsLl90cyA9IDE7CiAgICAgIGJlc3QgPSBsb2NhbDsKICAgIH0KICB9IGNhdGNoIHt9CiAgLy8g
Q2hlY2sgcmVtb3RlCiAgdHJ5IHsKICAgIGNvbnN0IHJlbW90ZSA9IGF3YWl0IGZldGNoRGV2RWRpdHMobG9jYXRpb24pOwogICAg
aWYgKHJlbW90ZSkgewogICAgICBpZiAoIXJlbW90ZS5yZW1vdmVkKSByZW1vdGUucmVtb3ZlZCA9IFtdOwogICAgICBpZiAoIXJl
bW90ZS5tb3ZlZCkgcmVtb3RlLm1vdmVkID0ge307CiAgICAgIGlmICghcmVtb3RlLmFkZGVkKSByZW1vdGUuYWRkZWQgPSBbXTsK
ICAgICAgaWYgKCFyZW1vdGUuX3RzKSByZW1vdGUuX3RzID0gMDsKICAgICAgaWYgKHJlbW90ZS5fdHMgPiBiZXN0Ll90cykgYmVz
dCA9IHJlbW90ZTsKICAgIH0KICB9IGNhdGNoIHt9CiAgcmV0dXJuIGJlc3Q7Cn0KCmZ1bmN0aW9uIEFwcCgpIHsKICBjb25zdCBt
YXBSZWYgPSB1c2VSZWYobnVsbCk7CiAgY29uc3QgbWFwSW5zdCA9IHVzZVJlZihudWxsKTsKICBjb25zdCBtYXJrZXJzUmVmID0g
dXNlUmVmKHt9KTsKICBjb25zdCBmbGFnTWFya2Vyc1JlZiA9IHVzZVJlZih7fSk7CgogIGNvbnN0IGlzTW9iaWxlID0gdHlwZW9m
IHdpbmRvdyAhPT0gJ3VuZGVmaW5lZCcgJiYgd2luZG93LmlubmVyV2lkdGggPCA3Njg7CiAgY29uc3QgbXJrUiA9IGlzTW9iaWxl
ID8gMTAgOiA3OwogIGNvbnN0IG1ya1J2ID0gaXNNb2JpbGUgPyA5IDogNjsKICBjb25zdCBtcmtSc2VsID0gaXNNb2JpbGUgPyAx
NCA6IDEwOwoKICAvLyBJbml0aWFsIGxvY2F0aW9uIGNvbWVzIGZyb20gdGhlIGVtYmVkZGluZyBhcHAgKEFEQVBUIGdsb2JhbCBy
YWlsKS4gSXQgaXMKICAvLyBpbmplY3RlZCBhcyB3aW5kb3cuX19BREFQVF9MT0NBVElPTiAoYW4gYXBwMiBzbHVnKSBiZWZvcmUg
dGhpcyBzY3JpcHQKICAvLyBydW5zLiBGYWxsIGJhY2sgdG8gJ21hc3RpY2JlYWNoJyBpZiBhYnNlbnQgb3IgdW5rbm93bi4KICBj
b25zdCBfaW5qTG9jID0gKHR5cGVvZiB3aW5kb3cgIT09ICJ1bmRlZmluZWQiICYmIHdpbmRvdy5fX0FEQVBUX0xPQ0FUSU9OCiAg
ICAgICAgICAgICAgICAgICAmJiBMT0NBVElPTlNbd2luZG93Ll9fQURBUFRfTE9DQVRJT05dKQogICAgICAgICAgICAgICAgICA/
IHdpbmRvdy5fX0FEQVBUX0xPQ0FUSU9OIDogIm1hc3RpY2JlYWNoIjsKICBjb25zdCBbY3VyTG9jLCBzZXRDdXJMb2NdID0gdXNl
U3RhdGUoX2luakxvYyk7CiAgY29uc3QgY3VyTG9jUmVmID0gdXNlUmVmKGN1ckxvYyk7CgogIGNvbnN0IHN3aXRjaExvY2F0aW9u
ID0gdXNlQ2FsbGJhY2soYXN5bmMgKG5ld0xvYykgPT4gewogICAgaWYgKG5ld0xvYyA9PT0gY3VyTG9jUmVmLmN1cnJlbnQpIHJl
dHVybjsKICAgIGN1ckxvY1JlZi5jdXJyZW50ID0gbmV3TG9jOwogICAgc2V0Q3VyTG9jKG5ld0xvYyk7CiAgICBzZXRTZWxlY3Rl
ZChudWxsKTsKICAgIHNldEZvcm0oRU1QVFlfRk9STSk7CiAgICBzZXRMb2FkaW5nKHRydWUpOwogICAgc2V0RGV2QWN0aW9uKG51
bGwpOwoKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlc3VsdCA9IGF3YWl0IGZldGNoU3VydmV5cyhuZXdMb2MpOwogICAgICBjb25z
dCBzaGVldERhdGEgPSAoIXJlc3VsdC5lcnJvciAmJiByZXN1bHQuZGF0YSkgPyByZXN1bHQuZGF0YSA6IHt9OwogICAgICBjb25z
dCBzaGVldEJ1aWxkaW5ncyA9IHNoZWV0RGF0YVRvQnVpbGRpbmdzKHNoZWV0RGF0YSk7CiAgICAgIGJhc2VCdWlsZGluZ3NSZWYu
Y3VycmVudCA9IHNoZWV0QnVpbGRpbmdzOwogICAgICBzZXRTdXJ2ZXlzKHNoZWV0RGF0YSk7CgogICAgICAvLyBBbHdheXMgcmVz
dG9yZSBkZXYgZWRpdHMgKG1heSBoYXZlIGFkZGVkIHBvaW50cyBldmVuIGlmIHNoZWV0IGlzIGVtcHR5KQogICAgICBjb25zdCBi
ZXN0ID0gYXdhaXQgbG9hZERldkVkaXRzKG5ld0xvYyk7CiAgICAgIGRldkVkaXRzUmVmLmN1cnJlbnQgPSBiZXN0OwogICAgICBz
ZXREZXZFZGl0cyhiZXN0KTsKICAgICAgc2V0QnVpbGRpbmdzKGFwcGx5RGV2RWRpdHMoc2hlZXRCdWlsZGluZ3MsIGJlc3QpKTsK
ICAgICAgc2F2ZUxvY2FsRGV2KGJlc3QpOwogICAgfSBjYXRjaCB7CiAgICAgIGJhc2VCdWlsZGluZ3NSZWYuY3VycmVudCA9IFtd
OwogICAgICBzZXRTdXJ2ZXlzKHt9KTsKICAgICAgLy8gU3RpbGwgdHJ5IHRvIGxvYWQgZGV2IGVkaXRzIChtYXkgaGF2ZSBhZGRl
ZCBwb2ludHMpCiAgICAgIHRyeSB7CiAgICAgICAgY29uc3QgYmVzdCA9IGF3YWl0IGxvYWREZXZFZGl0cyhuZXdMb2MpOwogICAg
ICAgIGRldkVkaXRzUmVmLmN1cnJlbnQgPSBiZXN0OwogICAgICAgIHNldERldkVkaXRzKGJlc3QpOwogICAgICAgIHNldEJ1aWxk
aW5ncyhhcHBseURldkVkaXRzKFtdLCBiZXN0KSk7CiAgICAgICAgc2F2ZUxvY2FsRGV2KGJlc3QpOwogICAgICB9IGNhdGNoIHsg
c2V0QnVpbGRpbmdzKFtdKTsgfQogICAgfQoKICAgIC8vIE1vdmUgbWFwCiAgICBjb25zdCBsb2MgPSBMT0NBVElPTlNbbmV3TG9j
XTsKICAgIGlmIChtYXBJbnN0LmN1cnJlbnQpIG1hcEluc3QuY3VycmVudC5zZXRWaWV3KGxvYy5jZW50ZXIsIGxvYy56b29tLCB7
IGFuaW1hdGU6IHRydWUgfSk7CgogICAgc2V0TG9hZGluZyhmYWxzZSk7CiAgfSwgW10pOwoKICBjb25zdCBbc2VsZWN0ZWQsIHNl
dFNlbGVjdGVkXSA9IHVzZVN0YXRlKG51bGwpOwogIGNvbnN0IFtmb3JtLCBzZXRGb3JtXSA9IHVzZVN0YXRlKEVNUFRZX0ZPUk0p
OwogIGNvbnN0IFtzdXJ2ZXlzLCBzZXRTdXJ2ZXlzXSA9IHVzZVN0YXRlKHt9KTsKICBjb25zdCBbbG9hZGluZywgc2V0TG9hZGlu
Z10gPSB1c2VTdGF0ZSh0cnVlKTsKICBjb25zdCBbc2F2aW5nLCBzZXRTYXZpbmddID0gdXNlU3RhdGUoZmFsc2UpOwogIGNvbnN0
IFt0b2FzdCwgc2V0VG9hc3RdID0gdXNlU3RhdGUobnVsbCk7CiAgY29uc3QgW2ZpbHRlciwgc2V0RmlsdGVyXSA9IHVzZVN0YXRl
KCJhbGwiKTsKICBjb25zdCBbYWR2RmlsdGVycywgc2V0QWR2RmlsdGVyc10gPSB1c2VTdGF0ZSh7IGZsYWdnZWQ6ICJhbGwiLCBv
Y2NDbGFzczogImFsbCIsIGZvdW5kYXRpb246ICJhbGwiLCBibGRnVHlwZTogImFsbCIgfSk7CiAgY29uc3QgW2RldkFjdGlvbiwg
c2V0RGV2QWN0aW9uXSA9IHVzZVN0YXRlKG51bGwpOyAvLyAibW92ZSIgfCAiYWRkIiB8IG51bGwKICBjb25zdCBbZGV2RWRpdHMs
IHNldERldkVkaXRzXSA9IHVzZVN0YXRlKHsgcmVtb3ZlZDogW10sIG1vdmVkOiB7fSwgYWRkZWQ6IFtdIH0pOwogIGNvbnN0IFtk
ZXZEaXJ0eSwgc2V0RGV2RGlydHldID0gdXNlU3RhdGUoZmFsc2UpOwogIGNvbnN0IFtkZXZTeW5jaW5nLCBzZXREZXZTeW5jaW5n
XSA9IHVzZVN0YXRlKGZhbHNlKTsKICBjb25zdCBbYnVpbGRpbmdzLCBzZXRCdWlsZGluZ3NdID0gdXNlU3RhdGUoW10pOwogIGNv
bnN0IGJhc2VCdWlsZGluZ3NSZWYgPSB1c2VSZWYoW10pOwogIGNvbnN0IGRldkVkaXRzUmVmID0gdXNlUmVmKGRldkVkaXRzKTsK
ICBjb25zdCBkZWJvdW5jZVRpbWVyUmVmID0gdXNlUmVmKG51bGwpOwoKICAvLyBIZWxwZXI6IHVwZGF0ZSByZWYgKyBzdGF0ZSAr
IGJ1aWxkaW5ncyArIGxvY2FsU3RvcmFnZSBzeW5jaHJvbm91c2x5LCB0aGVuIGF1dG8tc2F2ZSB0byBzZXJ2ZXIKICBjb25zdCBj
b21taXREZXZFZGl0cyA9IHVzZUNhbGxiYWNrKChuZXh0KSA9PiB7CiAgICAvLyBBZGQgdGltZXN0YW1wIGZvciBjb25mbGljdCBy
ZXNvbHV0aW9uCiAgICBuZXh0Ll90cyA9IERhdGUubm93KCk7CiAgICBkZXZFZGl0c1JlZi5jdXJyZW50ID0gbmV4dDsKICAgIHNl
dERldkVkaXRzKG5leHQpOwogICAgc2V0QnVpbGRpbmdzKGFwcGx5RGV2RWRpdHMoYmFzZUJ1aWxkaW5nc1JlZi5jdXJyZW50LCBu
ZXh0KSk7CiAgICBzYXZlTG9jYWxEZXYobmV4dCk7CiAgICBzZXREZXZEaXJ0eSh0cnVlKTsKICAgIC8vIEF1dG8tc2F2ZSB0byBz
ZXJ2ZXIgd2l0aCAzcyBkZWJvdW5jZQogICAgaWYgKGRlYm91bmNlVGltZXJSZWYuY3VycmVudCkgY2xlYXJUaW1lb3V0KGRlYm91
bmNlVGltZXJSZWYuY3VycmVudCk7CiAgICBkZWJvdW5jZVRpbWVyUmVmLmN1cnJlbnQgPSBzZXRUaW1lb3V0KGFzeW5jICgpID0+
IHsKICAgICAgc2F2aW5nUmVmLmN1cnJlbnQgPSB0cnVlOwogICAgICB0cnkgewogICAgICAgIGNvbnN0IGN1ciA9IGRldkVkaXRz
UmVmLmN1cnJlbnQ7CiAgICAgICAgY29uc3QgY3VyQnVpbGRpbmdzID0gYXBwbHlEZXZFZGl0cyhiYXNlQnVpbGRpbmdzUmVmLmN1
cnJlbnQsIGN1cik7CiAgICAgICAgYXdhaXQgc2F2ZURldkVkaXRzUmVtb3RlKGN1ciwgY3VyTG9jUmVmLmN1cnJlbnQpOwogICAg
ICAgIC8vIERlbGV0ZSByZW1vdmVkIHN1cnZleSByb3dzCiAgICAgICAgZm9yIChjb25zdCB1aWQgb2YgKGN1ci5yZW1vdmVkfHxb
XSkpIHsKICAgICAgICAgIGF3YWl0IGRlbGV0ZVN1cnZleUVudHJ5KHVpZCwgY3VyTG9jUmVmLmN1cnJlbnQpOwogICAgICAgIH0K
ICAgICAgICAvLyBVcGRhdGUgbW92ZWQgcG9pbnRzJyBjb29yZGluYXRlcwogICAgICAgIGZvciAoY29uc3QgdWlkIGluIChjdXIu
bW92ZWR8fHt9KSkgewogICAgICAgICAgY29uc3QgYiA9IGN1ckJ1aWxkaW5ncy5maW5kKHggPT4geC51aWQgPT09IHVpZCk7CiAg
ICAgICAgICBpZiAoIWIpIGNvbnRpbnVlOwogICAgICAgICAgY29uc3QgcyA9IHN1cnZleXNSZWYuY3VycmVudFt1aWRdIHx8IChi
LnByZWZpbGwgPyB7Li4uYi5wcmVmaWxsfSA6IG51bGwpOwogICAgICAgICAgaWYgKHMpIHsKICAgICAgICAgICAgYXdhaXQgc2F2
ZVN1cnZleUVudHJ5KHVpZCwgYi50eXBlLCBiLm5zaUlkLCBiLmxuZywgYi5sYXQsIHsuLi5zLCBzYXZlZEF0OiBzLnNhdmVkQXQg
fHwgIiJ9LCBjdXJMb2NSZWYuY3VycmVudCk7CiAgICAgICAgICB9CiAgICAgICAgfQogICAgICAgIHNldERldkRpcnR5KGZhbHNl
KTsKICAgICAgfSBjYXRjaCAoZXJyKSB7CiAgICAgICAgY29uc29sZS5lcnJvcigiRGV2IGF1dG8tc2F2ZSBmYWlsZWQ6IiwgZXJy
KTsKICAgICAgICBpZiAodHlwZW9mIHdpbmRvdy5fX3Nob3dUb2FzdCA9PT0gImZ1bmN0aW9uIikgd2luZG93Ll9fc2hvd1RvYXN0
KCJEZXYgc3luYyBmYWlsZWQ6ICIgKyBlcnIubWVzc2FnZSwgImVycm9yIik7CiAgICAgIH0gZmluYWxseSB7CiAgICAgICAgc2F2
aW5nUmVmLmN1cnJlbnQgPSBmYWxzZTsKICAgICAgfQogICAgfSwgMzAwMCk7CiAgfSwgW10pOwoKICAvLyBXYXJuIGJlZm9yZSBs
ZWF2aW5nIHdpdGggdW5zYXZlZCBkZXYgZWRpdHMKICB1c2VFZmZlY3QoKCkgPT4gewogICAgY29uc3QgaGFuZGxlciA9IChlKSA9
PiB7CiAgICAgIGlmIChkZXZEaXJ0eSkgeyBlLnByZXZlbnREZWZhdWx0KCk7IGUucmV0dXJuVmFsdWUgPSAiIjsgfQogICAgfTsK
ICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKCJiZWZvcmV1bmxvYWQiLCBoYW5kbGVyKTsKICAgIHJldHVybiAoKSA9PiB3aW5k
b3cucmVtb3ZlRXZlbnRMaXN0ZW5lcigiYmVmb3JldW5sb2FkIiwgaGFuZGxlcik7CiAgfSwgW2RldkRpcnR5XSk7CgogIC8vIExp
c3RlbiBmb3IgbG9jYWxTdG9yYWdlIGNoYW5nZXMgZnJvbSBvdGhlciB0YWJzCiAgdXNlRWZmZWN0KCgpID0+IHsKICAgIGNvbnN0
IGhhbmRsZXIgPSAoZSkgPT4gewogICAgICBpZiAoZS5rZXkgPT09ICgibnNpLWRldi1lZGl0cy0iICsgY3VyTG9jUmVmLmN1cnJl
bnQpICYmIGUubmV3VmFsdWUpIHsKICAgICAgICB0cnkgewogICAgICAgICAgY29uc3Qgb3RoZXIgPSBKU09OLnBhcnNlKGUubmV3
VmFsdWUpOwogICAgICAgICAgY29uc3QgbWluZSA9IGRldkVkaXRzUmVmLmN1cnJlbnQ7CiAgICAgICAgICAvLyBPbmx5IGFjY2Vw
dCBpZiBuZXdlciB0aW1lc3RhbXAKICAgICAgICAgIGlmICgob3RoZXIuX3RzIHx8IDApID4gKG1pbmUuX3RzIHx8IDApKSB7CiAg
ICAgICAgICAgIGRldkVkaXRzUmVmLmN1cnJlbnQgPSBvdGhlcjsKICAgICAgICAgICAgc2V0RGV2RWRpdHMob3RoZXIpOwogICAg
ICAgICAgICBzZXRCdWlsZGluZ3MoYXBwbHlEZXZFZGl0cyhiYXNlQnVpbGRpbmdzUmVmLmN1cnJlbnQsIG90aGVyKSk7CiAgICAg
ICAgICB9CiAgICAgICAgfSBjYXRjaCB7fQogICAgICB9CiAgICB9OwogICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoInN0b3Jh
Z2UiLCBoYW5kbGVyKTsKICAgIHJldHVybiAoKSA9PiB3aW5kb3cucmVtb3ZlRXZlbnRMaXN0ZW5lcigic3RvcmFnZSIsIGhhbmRs
ZXIpOwogIH0sIFtdKTsKICBjb25zdCBzYXZpbmdSZWYgPSB1c2VSZWYoZmFsc2UpOwogIGNvbnN0IHN1cnZleXNSZWYgPSB1c2VS
ZWYoc3VydmV5cyk7CiAgY29uc3QgZGV2QWN0aW9uUmVmID0gdXNlUmVmKG51bGwpOwogIGNvbnN0IHNlbGVjdGVkUmVmID0gdXNl
UmVmKG51bGwpOwogIGNvbnN0IGZvcm1SZWYgPSB1c2VSZWYoRU1QVFlfRk9STSk7CiAgY29uc3QgcGVuZGluZ1NhdmVSZWYgPSB1
c2VSZWYobnVsbCk7IC8vIHRyYWNrcyBpbi1mbGlnaHQgcm93IGNyZWF0aW9uIGZvciBuZXcgcG9pbnRzCiAgdXNlRWZmZWN0KCgp
ID0+IHsgc3VydmV5c1JlZi5jdXJyZW50ID0gc3VydmV5czsgfSwgW3N1cnZleXNdKTsKCiAgLy8gRGV2IGVkaXRzOiBsb2NhbFN0
b3JhZ2UgaGVscGVyCiAgZnVuY3Rpb24gc2F2ZUxvY2FsRGV2KGQpIHsKICAgIHRyeSB7IGxvY2FsU3RvcmFnZS5zZXRJdGVtKCJu
c2ktZGV2LWVkaXRzLSIgKyBjdXJMb2NSZWYuY3VycmVudCwgSlNPTi5zdHJpbmdpZnkoZCkpOyB9IGNhdGNoKGUpIHt9CiAgfQoK
ICBjb25zdCByZWZyZXNoU3VydmV5cyA9IHVzZUNhbGxiYWNrKGFzeW5jICgpID0+IHsKICAgIGlmIChzYXZpbmdSZWYuY3VycmVu
dCkgcmV0dXJuOwogICAgY29uc3QgcmVzdWx0ID0gYXdhaXQgZmV0Y2hTdXJ2ZXlzKGN1ckxvY1JlZi5jdXJyZW50KTsKICAgIGlm
IChzYXZpbmdSZWYuY3VycmVudCkgcmV0dXJuOwogICAgaWYgKHJlc3VsdC5lcnJvcikgeyBjb25zb2xlLndhcm4oIlN1cnZleSBy
ZWZyZXNoIGZhaWxlZDoiLCByZXN1bHQuZXJyb3IpOyByZXR1cm47IH0KICAgIGNvbnN0IHNoZWV0RGF0YSA9IHJlc3VsdC5kYXRh
IHx8IHt9OwogICAgLy8gUHJlc2VydmUgbG9jYWwgZmxhZ2dlZCB2YWx1ZXMg4oCUIHRoZSBzaGVldCBtYXkgbm90IGhhdmUgdGhl
IGNvbHVtbiB5ZXQsCiAgICAvLyBvciB0aGVyZSBtYXkgYmUgYSByYWNlIGJldHdlZW4gb3VyIHNhdmUgYW5kIHRoaXMgcG9sbCBm
ZXRjaC4KICAgIGNvbnN0IHByZXYgPSBzdXJ2ZXlzUmVmLmN1cnJlbnQ7CiAgICBmb3IgKGNvbnN0IHVpZCBpbiBzaGVldERhdGEp
IHsKICAgICAgaWYgKCFzaGVldERhdGFbdWlkXS5mbGFnZ2VkICYmIHByZXZbdWlkXSAmJiBwcmV2W3VpZF0uZmxhZ2dlZCkgewog
ICAgICAgIHNoZWV0RGF0YVt1aWRdLmZsYWdnZWQgPSBwcmV2W3VpZF0uZmxhZ2dlZDsKICAgICAgfQogICAgfQogICAgc2V0U3Vy
dmV5cyhzaGVldERhdGEpOwogICAgLy8gQWx3YXlzIHJlYnVpbGQgYnVpbGRpbmdzIGZyb20gc2hlZXQgKGV2ZW4gaWYgZW1wdHkg
4oCUIGNsZWFycyBzdGFsZSBkYXRhKQogICAgY29uc3QgZnJlc2hCdWlsZGluZ3MgPSBzaGVldERhdGFUb0J1aWxkaW5ncyhzaGVl
dERhdGEpOwogICAgYmFzZUJ1aWxkaW5nc1JlZi5jdXJyZW50ID0gZnJlc2hCdWlsZGluZ3M7CiAgICAvLyBBbHdheXMgcmVmcmVz
aCByZW1vdGUgZGV2IGVkaXRzIChhZG9wdCBpZiBuZXdlcikKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJlbW90ZSA9IGF3YWl0IGZl
dGNoRGV2RWRpdHMoY3VyTG9jUmVmLmN1cnJlbnQpOwogICAgICBpZiAocmVtb3RlICYmIChyZW1vdGUuX3RzIHx8IDApID4gKGRl
dkVkaXRzUmVmLmN1cnJlbnQuX3RzIHx8IDApKSB7CiAgICAgICAgaWYgKCFyZW1vdGUucmVtb3ZlZCkgcmVtb3RlLnJlbW92ZWQg
PSBbXTsKICAgICAgICBpZiAoIXJlbW90ZS5tb3ZlZCkgcmVtb3RlLm1vdmVkID0ge307CiAgICAgICAgaWYgKCFyZW1vdGUuYWRk
ZWQpIHJlbW90ZS5hZGRlZCA9IFtdOwogICAgICAgIGRldkVkaXRzUmVmLmN1cnJlbnQgPSByZW1vdGU7CiAgICAgICAgc2V0RGV2
RWRpdHMocmVtb3RlKTsKICAgICAgICBzYXZlTG9jYWxEZXYocmVtb3RlKTsKICAgICAgfQogICAgfSBjYXRjaCB7fQogICAgc2V0
QnVpbGRpbmdzKGFwcGx5RGV2RWRpdHMoZnJlc2hCdWlsZGluZ3MsIGRldkVkaXRzUmVmLmN1cnJlbnQpKTsKICB9LCBbXSk7Cgog
IHVzZUVmZmVjdCgoKSA9PiB7CiAgICAvLyBBdXRvLXB1bGwgZnJvbSBHb29nbGUgU2hlZXQgb24gc3RhcnR1cAogICAgY29uc3Qg
aW5pdEZyb21TaGVldCA9IGFzeW5jICgpID0+IHsKICAgICAgdHJ5IHsKICAgICAgICBjb25zdCByZXN1bHQgPSBhd2FpdCBmZXRj
aFN1cnZleXMoY3VyTG9jUmVmLmN1cnJlbnQpOwogICAgICAgIGlmIChyZXN1bHQuZXJyb3IpIHsKICAgICAgICAgIGNvbnNvbGUu
ZXJyb3IoIkluaXRpYWwgbG9hZCBmYWlsZWQ6IiwgcmVzdWx0LmVycm9yKTsKICAgICAgICAgIHNldFRpbWVvdXQoKCkgPT4gewog
ICAgICAgICAgICBpZiAodHlwZW9mIHdpbmRvdy5fX3Nob3dUb2FzdCA9PT0gImZ1bmN0aW9uIikgd2luZG93Ll9fc2hvd1RvYXN0
KCLimqAgRmFpbGVkIHRvIGxvYWQgZnJvbSBHb29nbGUgU2hlZXQ6ICIgKyByZXN1bHQuZXJyb3IsICJlcnJvciIpOwogICAgICAg
ICAgfSwgNTAwKTsKICAgICAgICAgIHNldExvYWRpbmcoZmFsc2UpOwogICAgICAgICAgcmV0dXJuOwogICAgICAgIH0KCiAgICAg
ICAgY29uc3Qgc2hlZXREYXRhID0gcmVzdWx0LmRhdGEgfHwge307CiAgICAgICAgY29uc3Qgc2hlZXRCdWlsZGluZ3MgPSBzaGVl
dERhdGFUb0J1aWxkaW5ncyhzaGVldERhdGEpOwogICAgICAgIGJhc2VCdWlsZGluZ3NSZWYuY3VycmVudCA9IHNoZWV0QnVpbGRp
bmdzOwogICAgICAgIHNldFN1cnZleXMoc2hlZXREYXRhKTsKCiAgICAgICAgLy8gQWx3YXlzIHJlc3RvcmUgZGV2IGVkaXRzICht
YXkgY29udGFpbiBhZGRlZCBwb2ludHMgZXZlbiBpZiBzaGVldCBpcyBlbXB0eSkKICAgICAgICBjb25zdCBiZXN0ID0gYXdhaXQg
bG9hZERldkVkaXRzKGN1ckxvY1JlZi5jdXJyZW50KTsKICAgICAgICBkZXZFZGl0c1JlZi5jdXJyZW50ID0gYmVzdDsKICAgICAg
ICBzZXREZXZFZGl0cyhiZXN0KTsKICAgICAgICBzZXRCdWlsZGluZ3MoYXBwbHlEZXZFZGl0cyhzaGVldEJ1aWxkaW5ncywgYmVz
dCkpOwogICAgICAgIHNhdmVMb2NhbERldihiZXN0KTsKICAgICAgfSBjYXRjaCAoZXJyKSB7CiAgICAgICAgY29uc29sZS5lcnJv
cigiSW5pdCBmYWlsZWQ6IiwgZXJyKTsKICAgICAgfQogICAgICBzZXRMb2FkaW5nKGZhbHNlKTsKICAgIH07CiAgICBpbml0RnJv
bVNoZWV0KCk7CiAgICAvLyBQb2xsIHN1cnZleXMgZXZlcnkgMzBzCiAgICBjb25zdCBpbnRlcnZhbCA9IHNldEludGVydmFsKHJl
ZnJlc2hTdXJ2ZXlzLCAzMDAwMCk7CiAgICByZXR1cm4gKCkgPT4gY2xlYXJJbnRlcnZhbChpbnRlcnZhbCk7CiAgfSwgW10pOwoK
ICBjb25zdCBzaG93VG9hc3QgPSB1c2VDYWxsYmFjaygobXNnLCB0eXBlPSJzdWNjZXNzIikgPT4gewogICAgc2V0VG9hc3Qoe21z
Zyx0eXBlfSk7IHNldFRpbWVvdXQoKCkgPT4gc2V0VG9hc3QobnVsbCksIDMwMDApOwogIH0sIFtdKTsKICB1c2VFZmZlY3QoKCkg
PT4geyB3aW5kb3cuX19zaG93VG9hc3QgPSBzaG93VG9hc3Q7IHJldHVybiAoKSA9PiB7IGRlbGV0ZSB3aW5kb3cuX19zaG93VG9h
c3Q7IH07IH0sIFtzaG93VG9hc3RdKTsKCiAgdXNlRWZmZWN0KCgpID0+IHsKICAgIGlmIChtYXBJbnN0LmN1cnJlbnQgfHwgIW1h
cFJlZi5jdXJyZW50KSByZXR1cm47CiAgICBjb25zdCBsaW5rID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgibGluayIpOwogICAg
bGluay5yZWwgPSAic3R5bGVzaGVldCI7CiAgICBsaW5rLmhyZWYgPSAiaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4
L2xpYnMvbGVhZmxldC8xLjkuNC9sZWFmbGV0Lm1pbi5jc3MiOwogICAgZG9jdW1lbnQuaGVhZC5hcHBlbmRDaGlsZChsaW5rKTsK
ICAgIGNvbnN0IHNjcmlwdCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoInNjcmlwdCIpOwogICAgc2NyaXB0LnNyYyA9ICJodHRw
czovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0LzEuOS40L2xlYWZsZXQubWluLmpzIjsKICAgIHNjcmlw
dC5vbmxvYWQgPSAoKSA9PiB7CiAgICAgIGNvbnN0IEwgPSB3aW5kb3cuTDsKICAgICAgY29uc3QgX2xvYzAgPSBMT0NBVElPTlNb
Y3VyTG9jUmVmLmN1cnJlbnRdIHx8IExPQ0FUSU9OUy5tYXN0aWNiZWFjaDsKICAgICAgY29uc3QgbWFwID0gTC5tYXAobWFwUmVm
LmN1cnJlbnQsIHsgem9vbUNvbnRyb2w6IGZhbHNlIH0pLnNldFZpZXcoX2xvYzAuY2VudGVyLCBfbG9jMC56b29tKTsKICAgICAg
TC5jb250cm9sLnpvb20oeyBwb3NpdGlvbjogInRvcHJpZ2h0IiB9KS5hZGRUbyhtYXApOwogICAgICBjb25zdCBzdHJlZXRMYXll
ciA9IEwudGlsZUxheWVyKCJodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZyIsIHsKICAg
ICAgICBhdHRyaWJ1dGlvbjogIiZjb3B5OyBPU00iLCBtYXhab29tOiAxOSwKICAgICAgfSk7CiAgICAgIGNvbnN0IGFlcmlhbExh
eWVyID0gTC50aWxlTGF5ZXIoImh0dHBzOi8vc2VydmVyLmFyY2dpc29ubGluZS5jb20vQXJjR0lTL3Jlc3Qvc2VydmljZXMvV29y
bGRfSW1hZ2VyeS9NYXBTZXJ2ZXIvdGlsZS97en0ve3l9L3t4fSIsIHsKICAgICAgICBhdHRyaWJ1dGlvbjogIiZjb3B5OyBFc3Jp
LCBNYXhhciwgRWFydGhzdGFyIEdlb2dyYXBoaWNzIiwgbWF4Wm9vbTogMTksCiAgICAgIH0pOwogICAgICBzdHJlZXRMYXllci5h
ZGRUbyhtYXApOwogICAgICBMLmNvbnRyb2wubGF5ZXJzKHsgIlN0cmVldCI6IHN0cmVldExheWVyLCAiQWVyaWFsIjogYWVyaWFs
TGF5ZXIgfSwgbnVsbCwgeyBwb3NpdGlvbjogInRvcHJpZ2h0IiwgY29sbGFwc2VkOiBmYWxzZSB9KS5hZGRUbyhtYXApOwogICAg
ICBtYXBJbnN0LmN1cnJlbnQgPSBtYXA7CgogICAgICBtYXAub24oImNsaWNrIiwgKGUpID0+IHsKICAgICAgICBpZiAod2luZG93
Ll9fZHJhd01vZGUgJiYgd2luZG93Ll9fZHJhd0NsaWNrKSB7CiAgICAgICAgICB3aW5kb3cuX19kcmF3Q2xpY2soZS5sYXRsbmcu
bGF0LCBlLmxhdGxuZy5sbmcpOwogICAgICAgIH0gZWxzZSBpZiAoZGV2QWN0aW9uUmVmLmN1cnJlbnQgPT09ICJtb3ZlIiAmJiBz
ZWxlY3RlZFJlZi5jdXJyZW50KSB7CiAgICAgICAgICB3aW5kb3cuX19kZXZNb3ZlKGUubGF0bG5nLmxhdCwgZS5sYXRsbmcubG5n
KTsKICAgICAgICB9IGVsc2UgaWYgKGRldkFjdGlvblJlZi5jdXJyZW50ID09PSAiYWRkIiB8fCBkZXZBY3Rpb25SZWYuY3VycmVu
dCA9PT0gImRlbW9saXNoZWQiKSB7CiAgICAgICAgICB3aW5kb3cuX19kZXZBZGQoZS5sYXRsbmcubGF0LCBlLmxhdGxuZy5sbmcp
OwogICAgICAgIH0KICAgICAgfSk7CiAgICB9OwogICAgZG9jdW1lbnQuaGVhZC5hcHBlbmRDaGlsZChzY3JpcHQpOwogIH0sIFtd
KTsKCiAgLy8gU3luYyBtYXJrZXJzIHdpdGggYnVpbGRpbmdzIHN0YXRlCiAgdXNlRWZmZWN0KCgpID0+IHsKICAgIGlmICghbWFw
SW5zdC5jdXJyZW50IHx8ICF3aW5kb3cuTCkgcmV0dXJuOwogICAgY29uc3QgTCA9IHdpbmRvdy5MOwogICAgLy8gUmVtb3ZlIG9s
ZCBtYXJrZXJzCiAgICBPYmplY3QudmFsdWVzKG1hcmtlcnNSZWYuY3VycmVudCkuZm9yRWFjaChtID0+IG0ucmVtb3ZlKCkpOwog
ICAgbWFya2Vyc1JlZi5jdXJyZW50ID0ge307CiAgICBPYmplY3QudmFsdWVzKGZsYWdNYXJrZXJzUmVmLmN1cnJlbnQpLmZvckVh
Y2gobSA9PiBtLnJlbW92ZSgpKTsKICAgIGZsYWdNYXJrZXJzUmVmLmN1cnJlbnQgPSB7fTsKICAgIC8vIENyZWF0ZSBjaXJjbGUg
bWFya2VycyBmb3IgYWxsIGJ1aWxkaW5ncwogICAgYnVpbGRpbmdzLmZvckVhY2goYiA9PiB7CiAgICAgIGNvbnN0IGlzViA9IGIu
dHlwZSA9PT0gInZlcmlmeSI7CiAgICAgIGNvbnN0IG0gPSBMLmNpcmNsZU1hcmtlcihbYi5sYXQsIGIubG5nXSwgewogICAgICAg
IHJhZGl1czogaXNWID8gbXJrUnYgOiBtcmtSLAogICAgICAgIGZpbGxDb2xvcjogaXNWID8gIiMzYjgyZjYiIDogIiNlZjQ0NDQi
LAogICAgICAgIGNvbG9yOiAiI2ZmZiIsIHdlaWdodDogMS41LCBmaWxsT3BhY2l0eTogMC44NSwKICAgICAgfSkuYWRkVG8obWFw
SW5zdC5jdXJyZW50KTsKICAgICAgbS5vbigiY2xpY2siLCAoKSA9PiB3aW5kb3cuX19zZWwoYi51aWQpKTsKICAgICAgbWFya2Vy
c1JlZi5jdXJyZW50W2IudWlkXSA9IG07CiAgICB9KTsKICB9LCBbYnVpbGRpbmdzXSk7CgogIGNvbnN0IHNlbFJpbmdSZWYgPSB1
c2VSZWYobnVsbCk7CgogIC8vIEhlbHBlcjogY2hlY2sgaWYgYSBidWlsZGluZyBwYXNzZXMgYWxsIGFjdGl2ZSBmaWx0ZXJzCiAg
Y29uc3QgcGFzc2VzRmlsdGVycyA9IHVzZUNhbGxiYWNrKChiKSA9PiB7CiAgICBjb25zdCBzdiA9IHN1cnZleXNbYi51aWRdOwog
ICAgY29uc3QgcGYgPSBiLnByZWZpbGwgfHwge307CiAgICBjb25zdCBkID0gc3YgfHwgcGY7CiAgICBjb25zdCBkb25lID0gISEo
c3YgJiYgc3Yuc2F2ZWRBdCk7CiAgICBjb25zdCBpc1YgPSBiLnR5cGUgPT09ICJ2ZXJpZnkiOwogICAgY29uc3QgaXNGbGFnZ2Vk
ID0gZC5mbGFnZ2VkID09PSAiWWVzIjsKICAgIGNvbnN0IGlzRGVtb2xpc2hlZCA9IGQuZmxhZ2dlZCA9PT0gIkRlbW9saXNoZWQi
OwogICAgLy8gU3RhdHVzIGZpbHRlcgogICAgaWYgKGZpbHRlciA9PT0gInN1cnZleSIgJiYgaXNWKSByZXR1cm4gZmFsc2U7CiAg
ICBpZiAoZmlsdGVyID09PSAiZG9uZSIgJiYgIWRvbmUpIHJldHVybiBmYWxzZTsKICAgIGlmIChmaWx0ZXIgPT09ICJwZW5kaW5n
IiAmJiBkb25lKSByZXR1cm4gZmFsc2U7CiAgICAvLyBBZHZhbmNlZCBmaWx0ZXJzCiAgICBpZiAoYWR2RmlsdGVycy5mbGFnZ2Vk
ID09PSAieWVzIiAmJiAhaXNGbGFnZ2VkKSByZXR1cm4gZmFsc2U7CiAgICBpZiAoYWR2RmlsdGVycy5mbGFnZ2VkID09PSAibm8i
ICYmIChpc0ZsYWdnZWQgfHwgaXNEZW1vbGlzaGVkKSkgcmV0dXJuIGZhbHNlOwogICAgaWYgKGFkdkZpbHRlcnMuZmxhZ2dlZCA9
PT0gImRlbW9saXNoZWQiICYmICFpc0RlbW9saXNoZWQpIHJldHVybiBmYWxzZTsKICAgIGlmIChhZHZGaWx0ZXJzLm9jY0NsYXNz
ICE9PSAiYWxsIikgewogICAgICBjb25zdCBvY2MgPSAoZC5vY2NUeXBlIHx8ICIiKS50b1VwcGVyQ2FzZSgpOwogICAgICBpZiAo
YWR2RmlsdGVycy5vY2NDbGFzcyA9PT0gIlJFUyIgJiYgIW9jYy5zdGFydHNXaXRoKCJSRVMiKSkgcmV0dXJuIGZhbHNlOwogICAg
ICBpZiAoYWR2RmlsdGVycy5vY2NDbGFzcyA9PT0gIkNPTSIgJiYgIW9jYy5zdGFydHNXaXRoKCJDT00iKSkgcmV0dXJuIGZhbHNl
OwogICAgICBpZiAoYWR2RmlsdGVycy5vY2NDbGFzcyA9PT0gIklORCIgJiYgIW9jYy5zdGFydHNXaXRoKCJJTkQiKSkgcmV0dXJu
IGZhbHNlOwogICAgICBpZiAoYWR2RmlsdGVycy5vY2NDbGFzcyA9PT0gIk9USEVSIiAmJiAob2NjLnN0YXJ0c1dpdGgoIlJFUyIp
IHx8IG9jYy5zdGFydHNXaXRoKCJDT00iKSB8fCBvY2Muc3RhcnRzV2l0aCgiSU5EIikpKSByZXR1cm4gZmFsc2U7CiAgICB9CiAg
ICBpZiAoYWR2RmlsdGVycy5mb3VuZGF0aW9uICE9PSAiYWxsIiAmJiAoZC5mb3VuZGF0aW9uVHlwZSB8fCAiIikudG9VcHBlckNh
c2UoKSAhPT0gYWR2RmlsdGVycy5mb3VuZGF0aW9uKSByZXR1cm4gZmFsc2U7CiAgICBpZiAoYWR2RmlsdGVycy5ibGRnVHlwZSAh
PT0gImFsbCIgJiYgKGQuYnVpbGRpbmdUeXBlIHx8ICIiKS50b1VwcGVyQ2FzZSgpICE9PSBhZHZGaWx0ZXJzLmJsZGdUeXBlKSBy
ZXR1cm4gZmFsc2U7CiAgICByZXR1cm4gdHJ1ZTsKICB9LCBbc3VydmV5cywgZmlsdGVyLCBhZHZGaWx0ZXJzXSk7CgogIC8vIFVw
ZGF0ZSBtYXJrZXJzIHN0eWxlCiAgdXNlRWZmZWN0KCgpID0+IHsKICAgIC8vIFJlbW92ZSBvbGQgc2VsZWN0aW9uIHJpbmcKICAg
IGlmIChzZWxSaW5nUmVmLmN1cnJlbnQpIHsgc2VsUmluZ1JlZi5jdXJyZW50LnJlbW92ZSgpOyBzZWxSaW5nUmVmLmN1cnJlbnQg
PSBudWxsOyB9CiAgICAvLyBSZW1vdmUgb2xkIGZsYWcgbWFya2VycwogICAgT2JqZWN0LnZhbHVlcyhmbGFnTWFya2Vyc1JlZi5j
dXJyZW50KS5mb3JFYWNoKG0gPT4gbS5yZW1vdmUoKSk7CiAgICBmbGFnTWFya2Vyc1JlZi5jdXJyZW50ID0ge307CgogICAgY29u
c3QgTCA9IHdpbmRvdy5MOwoKICAgIGJ1aWxkaW5ncy5mb3JFYWNoKGIgPT4gewogICAgICBjb25zdCBjbSA9IG1hcmtlcnNSZWYu
Y3VycmVudFtiLnVpZF07CiAgICAgIGlmICghY20pIHJldHVybjsKICAgICAgY29uc3Qgc3YgPSBzdXJ2ZXlzW2IudWlkXTsKICAg
ICAgY29uc3QgcGYgPSBiLnByZWZpbGwgfHwge307CiAgICAgIGNvbnN0IGQgPSBzdiB8fCBwZjsKICAgICAgY29uc3QgZG9uZSA9
ICEhKHN2ICYmIHN2LnNhdmVkQXQpOwogICAgICBjb25zdCBpc0ZsYWdnZWQgPSBkLmZsYWdnZWQgPT09ICJZZXMiOwogICAgICBj
b25zdCBpc0RlbW9saXNoZWQgPSBkLmZsYWdnZWQgPT09ICJEZW1vbGlzaGVkIjsKICAgICAgY29uc3QgaXNWID0gYi50eXBlID09
PSAidmVyaWZ5IjsKICAgICAgY29uc3Qgc2hvdyA9IHBhc3Nlc0ZpbHRlcnMoYik7CgogICAgICBpZiAoaXNGbGFnZ2VkICYmIEwg
JiYgbWFwSW5zdC5jdXJyZW50KSB7CiAgICAgICAgLy8gRmxhZ2dlZDogaGlkZSBjaXJjbGUgbWFya2VyLCBzaG93IG9yYW5nZSBm
bGFnIERpdkljb24KICAgICAgICBjbS5zZXRTdHlsZSh7IGZpbGxPcGFjaXR5OiAwLCBvcGFjaXR5OiAwLCByYWRpdXM6IDAgfSk7
CiAgICAgICAgY29uc3QgZmxhZ1N2ZyA9IGA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgd2lkdGg9IjI0
IiBoZWlnaHQ9IjI4IiB2aWV3Qm94PSIwIDAgMjQgMjgiPjxsaW5lIHgxPSI0IiB5MT0iMiIgeDI9IjQiIHkyPSIyNyIgc3Ryb2tl
PSIjYjQ1MzA5IiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTUgMyBMMjEgNyBM
NSAxMyBaIiBmaWxsPSIjZjk3MzE2IiBzdHJva2U9IiNiNDUzMDkiIHN0cm9rZS13aWR0aD0iMSIvPjwvc3ZnPmA7CiAgICAgICAg
Y29uc3QgZmxhZ0ljb24gPSBMLmRpdkljb24oewogICAgICAgICAgaHRtbDogYDxkaXYgc3R5bGU9Im9wYWNpdHk6JHtzaG93PzE6
MC4wOH07dHJhbnNpdGlvbjpvcGFjaXR5IC4yczsiPiR7ZmxhZ1N2Z308L2Rpdj5gLAogICAgICAgICAgY2xhc3NOYW1lOiAnJywK
ICAgICAgICAgIGljb25TaXplOiBbMjQsIDI4XSwKICAgICAgICAgIGljb25BbmNob3I6IFs0LCAyN10sCiAgICAgICAgfSk7CiAg
ICAgICAgY29uc3QgZm0gPSBMLm1hcmtlcihbYi5sYXQsIGIubG5nXSwgeyBpY29uOiBmbGFnSWNvbiwgaW50ZXJhY3RpdmU6IHRy
dWUsIHpJbmRleE9mZnNldDogNDAwIH0pLmFkZFRvKG1hcEluc3QuY3VycmVudCk7CiAgICAgICAgZm0ub24oImNsaWNrIiwgKCkg
PT4gd2luZG93Ll9fc2VsKGIudWlkKSk7CiAgICAgICAgZmxhZ01hcmtlcnNSZWYuY3VycmVudFtiLnVpZF0gPSBmbTsKICAgICAg
fSBlbHNlIGlmIChpc0RlbW9saXNoZWQgJiYgTCAmJiBtYXBJbnN0LmN1cnJlbnQpIHsKICAgICAgICAvLyBEZW1vbGlzaGVkOiBo
aWRlIGNpcmNsZSBtYXJrZXIsIHNob3cgYmxhY2sg4pyVIGljb24KICAgICAgICBjbS5zZXRTdHlsZSh7IGZpbGxPcGFjaXR5OiAw
LCBvcGFjaXR5OiAwLCByYWRpdXM6IDAgfSk7CiAgICAgICAgY29uc3QgeFN2ZyA9IGA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3Lncz
Lm9yZy8yMDAwL3N2ZyIgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4IiB2aWV3Qm94PSIwIDAgMTggMTgiPjxsaW5lIHgxPSIzIiB5MT0i
MyIgeDI9IjE1IiB5Mj0iMTUiIHN0cm9rZT0iIzFlMjkzYiIgc3Ryb2tlLXdpZHRoPSIzLjUiIHN0cm9rZS1saW5lY2FwPSJyb3Vu
ZCIvPjxsaW5lIHgxPSIxNSIgeTE9IjMiIHgyPSIzIiB5Mj0iMTUiIHN0cm9rZT0iIzFlMjkzYiIgc3Ryb2tlLXdpZHRoPSIzLjUi
IHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxsaW5lIHgxPSIzIiB5MT0iMyIgeDI9IjE1IiB5Mj0iMTUiIHN0cm9rZT0iIzk0YTNi
OCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48bGluZSB4MT0iMTUiIHkxPSIzIiB4Mj0iMyIgeTI9
IjE1IiBzdHJva2U9IiM5NGEzYjgiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+YDsKICAg
ICAgICBjb25zdCB4SWNvbiA9IEwuZGl2SWNvbih7CiAgICAgICAgICBodG1sOiBgPGRpdiBzdHlsZT0ib3BhY2l0eToke3Nob3c/
MTowLjA4fTt0cmFuc2l0aW9uOm9wYWNpdHkgLjJzOyI+JHt4U3ZnfTwvZGl2PmAsCiAgICAgICAgICBjbGFzc05hbWU6ICcnLAog
ICAgICAgICAgaWNvblNpemU6IFsxOCwgMThdLAogICAgICAgICAgaWNvbkFuY2hvcjogWzksIDldLAogICAgICAgIH0pOwogICAg
ICAgIGNvbnN0IGZtID0gTC5tYXJrZXIoW2IubGF0LCBiLmxuZ10sIHsgaWNvbjogeEljb24sIGludGVyYWN0aXZlOiB0cnVlLCB6
SW5kZXhPZmZzZXQ6IDMwMCB9KS5hZGRUbyhtYXBJbnN0LmN1cnJlbnQpOwogICAgICAgIGZtLm9uKCJjbGljayIsICgpID0+IHdp
bmRvdy5fX3NlbChiLnVpZCkpOwogICAgICAgIGZsYWdNYXJrZXJzUmVmLmN1cnJlbnRbYi51aWRdID0gZm07CiAgICAgIH0gZWxz
ZSB7CiAgICAgICAgLy8gTm9uLWZsYWdnZWQ6IG5vcm1hbCBjaXJjbGUgbWFya2VyCiAgICAgICAgY20uc2V0U3R5bGUoewogICAg
ICAgICAgZmlsbENvbG9yOiBkb25lID8gIiMyMmM1NWUiIDogaXNWID8gIiMzYjgyZjYiIDogIiNlZjQ0NDQiLAogICAgICAgICAg
Y29sb3I6ICIjZmZmIiwKICAgICAgICAgIHdlaWdodDogMS41LAogICAgICAgICAgb3BhY2l0eTogc2hvdyA/IDEgOiAwLjA4LAog
ICAgICAgICAgZmlsbE9wYWNpdHk6IHNob3cgPyAwLjg1IDogMC4wNSwKICAgICAgICAgIHJhZGl1czogc2hvdyA/IChpc1YgPyBt
cmtSdiA6IG1ya1IpIDogMywKICAgICAgICB9KTsKICAgICAgfQogICAgfSk7CgogICAgLy8gRHJhdyBzZWxlY3Rpb24gcmluZyBh
cm91bmQgc2VsZWN0ZWQgcG9pbnQKICAgIGlmIChzZWxlY3RlZCAmJiBtYXBJbnN0LmN1cnJlbnQgJiYgd2luZG93LkwpIHsKICAg
ICAgY29uc3QgYiA9IGJ1aWxkaW5ncy5maW5kKHggPT4geC51aWQgPT09IHNlbGVjdGVkKTsKICAgICAgaWYgKGIpIHsKICAgICAg
ICBzZWxSaW5nUmVmLmN1cnJlbnQgPSB3aW5kb3cuTC5jaXJjbGVNYXJrZXIoW2IubGF0LCBiLmxuZ10sIHsKICAgICAgICAgIHJh
ZGl1czogaXNNb2JpbGUgPyAyMCA6IDE2LAogICAgICAgICAgZmlsbENvbG9yOiAidHJhbnNwYXJlbnQiLAogICAgICAgICAgZmls
bE9wYWNpdHk6IDAsCiAgICAgICAgICBjb2xvcjogIiNmYmJmMjQiLAogICAgICAgICAgd2VpZ2h0OiAzLAogICAgICAgICAgb3Bh
Y2l0eTogMC45LAogICAgICAgIH0pLmFkZFRvKG1hcEluc3QuY3VycmVudCk7CiAgICAgIH0KICAgIH0KICB9LCBbc3VydmV5cywg
ZmlsdGVyLCBhZHZGaWx0ZXJzLCBwYXNzZXNGaWx0ZXJzLCBidWlsZGluZ3MsIHNlbGVjdGVkXSk7CgogIHVzZUVmZmVjdCgoKSA9
PiB7CiAgICB3aW5kb3cuX19zZWwgPSAodWlkKSA9PiB7CiAgICAgIGlmIChkZXZBY3Rpb25SZWYuY3VycmVudCkgcmV0dXJuOwog
ICAgICBpZiAod2luZG93Ll9fZHJhd01vZGUpIHJldHVybjsKICAgICAgY29uc3QgYiA9IGJ1aWxkaW5ncy5maW5kKHggPT4geC51
aWQgPT09IHVpZCk7CiAgICAgIGlmICghYikgcmV0dXJuOwogICAgICAvLyBTa2lwIHdhcm5pbmcgaWYgY2xpY2tpbmcgdGhlIGFs
cmVhZHktc2VsZWN0ZWQgYnVpbGRpbmcKICAgICAgY29uc3QgY3VyU2VsID0gc2VsZWN0ZWRSZWYuY3VycmVudDsKICAgICAgaWYg
KGN1clNlbCAmJiBjdXJTZWwgIT09IHVpZCAmJiBmb3JtUmVmLmN1cnJlbnQpIHsKICAgICAgICBjb25zdCBzdiA9IHN1cnZleXNS
ZWYuY3VycmVudFtjdXJTZWxdOwogICAgICAgIGNvbnN0IGIwID0gYnVpbGRpbmdzLmZpbmQoeCA9PiB4LnVpZCA9PT0gY3VyU2Vs
KTsKICAgICAgICBjb25zdCBiYXNlbGluZSA9IHN2IHx8IChiMCAmJiBiMC5wcmVmaWxsKSB8fCBFTVBUWV9GT1JNOwogICAgICAg
IGNvbnN0IGZvcm1GaWVsZHMgPSBbIm51bVN0b3JpZXMiLCJmaXJzdEZsb29ySGVpZ2h0IiwiYWRkcmVzcyIsIm9jY1R5cGUiLCJm
b3VuZGF0aW9uVHlwZSIsImJ1aWxkaW5nVHlwZSIsImFyZWEiLCJ5ZWFyQnVpbHQiLCJncm91bmRFbGV2Iiwic3RydWN0dXJlVmFs
dWUiLCJjb250ZW50VmFsdWUiLCJub3RlcyIsInN1cnZleW9yIl07CiAgICAgICAgY29uc3QgaGFzRWRpdHMgPSBmb3JtRmllbGRz
LnNvbWUoayA9PiAoZm9ybVJlZi5jdXJyZW50W2tdIHx8ICIiKSAhPT0gKGJhc2VsaW5lW2tdIHx8ICIiKSk7CiAgICAgICAgaWYg
KGhhc0VkaXRzKSB7CiAgICAgICAgICBpZiAoIWNvbmZpcm0oIuKaoO+4jyBZb3UgaGF2ZSB1bnNhdmVkIGNoYW5nZXMgb24gdGhl
IGN1cnJlbnQgYnVpbGRpbmcuXG5cbklmIHlvdSBzd2l0Y2ggdG8gYW5vdGhlciBidWlsZGluZywgdGhlc2UgY2hhbmdlcyB3aWxs
IGJlIGxvc3QuXG5cbkNvbnRpbnVlPyIpKSByZXR1cm47CiAgICAgICAgfQogICAgICB9CiAgICAgIHNldFNlbGVjdGVkKHVpZCk7
CiAgICAgIGNvbnN0IGV4aXN0aW5nID0gc3VydmV5c1t1aWRdOwogICAgICBpZiAoZXhpc3RpbmcpIHNldEZvcm0oey4uLkVNUFRZ
X0ZPUk0sIC4uLmV4aXN0aW5nfSk7CiAgICAgIGVsc2UgaWYgKGIucHJlZmlsbCkgc2V0Rm9ybSh7Li4uRU1QVFlfRk9STSwgLi4u
Yi5wcmVmaWxsfSk7CiAgICAgIGVsc2Ugc2V0Rm9ybShFTVBUWV9GT1JNKTsKICAgICAgaWYgKG1hcEluc3QuY3VycmVudCkgbWFw
SW5zdC5jdXJyZW50LnBhblRvKFtiLmxhdCwgYi5sbmddLCB7YW5pbWF0ZTogdHJ1ZX0pOwogICAgfTsKICAgIHJldHVybiAoKSA9
PiB7IGRlbGV0ZSB3aW5kb3cuX19zZWw7IH07CiAgfSwgW3N1cnZleXMsIGJ1aWxkaW5nc10pOwoKICAvLyBEZXYgbW9kZSBoYW5k
bGVycyBleHBvc2VkIHRvIG1hcCBjbGljawogIHVzZUVmZmVjdCgoKSA9PiB7CiAgICBkZXZBY3Rpb25SZWYuY3VycmVudCA9IGRl
dkFjdGlvbjsKICAgIHNlbGVjdGVkUmVmLmN1cnJlbnQgPSBzZWxlY3RlZDsKICB9LCBbZGV2QWN0aW9uLCBzZWxlY3RlZF0pOwoK
ICAvLyBLZWVwIGZvcm0gcmVmIGN1cnJlbnQgZm9yIHVuc2F2ZWQtY2hhbmdlcyBjaGVja3MKICB1c2VFZmZlY3QoKCkgPT4geyBm
b3JtUmVmLmN1cnJlbnQgPSBmb3JtOyB9LCBbZm9ybV0pOwoKICB1c2VFZmZlY3QoKCkgPT4gewogICAgd2luZG93Ll9fZGV2TW92
ZSA9IChsYXQsIGxuZykgPT4gewogICAgICBjb25zdCBzZWwgPSBzZWxlY3RlZFJlZi5jdXJyZW50OwogICAgICBpZiAoIXNlbCkg
cmV0dXJuOwogICAgICBjb25zdCBjdXIgPSBkZXZFZGl0c1JlZi5jdXJyZW50OwogICAgICBjb25zdCBuZXh0ID0geyAuLi5jdXIs
IG1vdmVkOiB7IC4uLmN1ci5tb3ZlZCwgW3NlbF06IHsgbGF0LCBsbmcgfSB9IH07CiAgICAgIC8vIEltbWVkaWF0ZWx5IHVwZGF0
ZSB0aGUgbWFya2VyIHBvc2l0aW9uIHdpdGhvdXQgd2FpdGluZyBmb3IgZnVsbCByZWJ1aWxkCiAgICAgIGNvbnN0IGV4aXN0aW5n
TWFya2VyID0gbWFya2Vyc1JlZi5jdXJyZW50W3NlbF07CiAgICAgIGlmIChleGlzdGluZ01hcmtlcikgZXhpc3RpbmdNYXJrZXIu
c2V0TGF0TG5nKFtsYXQsIGxuZ10pOwogICAgICBjb25zdCBleGlzdGluZ0ZsYWcgPSBmbGFnTWFya2Vyc1JlZi5jdXJyZW50W3Nl
bF07CiAgICAgIGlmIChleGlzdGluZ0ZsYWcpIGV4aXN0aW5nRmxhZy5zZXRMYXRMbmcoW2xhdCwgbG5nXSk7CiAgICAgIGlmIChz
ZWxSaW5nUmVmLmN1cnJlbnQpIHNlbFJpbmdSZWYuY3VycmVudC5zZXRMYXRMbmcoW2xhdCwgbG5nXSk7CiAgICAgIGNvbW1pdERl
dkVkaXRzKG5leHQpOwogICAgICBzZXREZXZBY3Rpb24obnVsbCk7CiAgICAgIHNob3dUb2FzdCgiUG9pbnQgbW92ZWQg4oCUIGF1
dG8tc2F2aW5nLi4uIik7CiAgICB9OwogICAgd2luZG93Ll9fZGV2QWRkID0gKGxhdCwgbG5nKSA9PiB7CiAgICAgIGNvbnN0IGlz
RGVtb2xpc2hlZCA9IGRldkFjdGlvblJlZi5jdXJyZW50ID09PSAiZGVtb2xpc2hlZCI7CiAgICAgIC8vIENoZWNrIGZvciB1bnNh
dmVkIGZvcm0gY2hhbmdlcyBvbiBjdXJyZW50IGJ1aWxkaW5nIChza2lwIGZvciBkZW1vbGlzaGVkIHNpbmNlIGl0IGRvZXNuJ3Qg
Y2hhbmdlIGZvcm0pCiAgICAgIGlmICghaXNEZW1vbGlzaGVkKSB7CiAgICAgICAgY29uc3QgY3VyU2VsID0gc2VsZWN0ZWRSZWYu
Y3VycmVudDsKICAgICAgICBpZiAoY3VyU2VsKSB7CiAgICAgICAgICBjb25zdCBzdiA9IHN1cnZleXNSZWYuY3VycmVudFtjdXJT
ZWxdOwogICAgICAgICAgY29uc3QgYjAgPSBidWlsZGluZ3MuZmluZCh4ID0+IHgudWlkID09PSBjdXJTZWwpOwogICAgICAgICAg
Y29uc3QgYmFzZWxpbmUgPSBzdiB8fCAoYjAgJiYgYjAucHJlZmlsbCkgfHwgRU1QVFlfRk9STTsKICAgICAgICAgIGNvbnN0IGZv
cm1GaWVsZHMgPSBbIm51bVN0b3JpZXMiLCJmaXJzdEZsb29ySGVpZ2h0IiwiYWRkcmVzcyIsIm9jY1R5cGUiLCJmb3VuZGF0aW9u
VHlwZSIsImJ1aWxkaW5nVHlwZSIsImFyZWEiLCJ5ZWFyQnVpbHQiLCJncm91bmRFbGV2Iiwic3RydWN0dXJlVmFsdWUiLCJjb250
ZW50VmFsdWUiLCJub3RlcyIsInN1cnZleW9yIl07CiAgICAgICAgICBjb25zdCBoYXNFZGl0cyA9IGZvcm1GaWVsZHMuc29tZShr
ID0+IChmb3JtUmVmLmN1cnJlbnRba10gfHwgIiIpICE9PSAoYmFzZWxpbmVba10gfHwgIiIpKTsKICAgICAgICAgIGlmIChoYXNF
ZGl0cykgewogICAgICAgICAgICBpZiAoIWNvbmZpcm0oIuKaoO+4jyBZb3UgaGF2ZSB1bnNhdmVkIGNoYW5nZXMgb24gdGhlIGN1
cnJlbnQgYnVpbGRpbmcuXG5cbklmIHlvdSBhZGQgYSBuZXcgcG9pbnQsIHRoZXNlIGNoYW5nZXMgd2lsbCBiZSBsb3N0LlxuXG5D
b250aW51ZT8iKSkgcmV0dXJuOwogICAgICAgICAgfQogICAgICAgIH0KICAgICAgfQoKICAgICAgY29uc3QgY3VyID0gZGV2RWRp
dHNSZWYuY3VycmVudDsKICAgICAgbGV0IG1heE5ld0lkID0gMTAwMDA7CiAgICAgIGJ1aWxkaW5ncy5mb3JFYWNoKGIgPT4gewog
ICAgICAgIGNvbnN0IGlkID0gcGFyc2VJbnQoYi5uc2lJZCB8fCAoYi51aWQuc3RhcnRzV2l0aCgibmV3LSIpID8gYi51aWQucmVw
bGFjZSgibmV3LSIsIiIpIDogIjAiKSk7CiAgICAgICAgaWYgKGlkID49IDEwMDAwICYmIGlkID4gbWF4TmV3SWQpIG1heE5ld0lk
ID0gaWQ7CiAgICAgIH0pOwogICAgICAoY3VyLmFkZGVkfHxbXSkuZm9yRWFjaChhID0+IHsKICAgICAgICBjb25zdCBpZCA9IHBh
cnNlSW50KGEubnNpSWQgfHwgIjAiKTsKICAgICAgICBpZiAoaWQgPj0gMTAwMDAgJiYgaWQgPiBtYXhOZXdJZCkgbWF4TmV3SWQg
PSBpZDsKICAgICAgfSk7CiAgICAgIGNvbnN0IG5ld0lkID0gbWF4TmV3SWQgKyAxOwogICAgICBjb25zdCB1aWQgPSAibmV3LSIg
KyBuZXdJZDsKICAgICAgY29uc3QgbmV3UHQgPSB7IHVpZCwgdHlwZTogInN1cnZleSIsIG5zaUlkOiBTdHJpbmcobmV3SWQpLCBs
bmcsIGxhdCwgcHJlZmlsbDogbnVsbCB9OwogICAgICBjb25zdCBuZXh0ID0geyAuLi5jdXIsIGFkZGVkOiBbLi4uKGN1ci5hZGRl
ZHx8W10pLCBuZXdQdF0gfTsKICAgICAgY29tbWl0RGV2RWRpdHMobmV4dCk7CiAgICAgIHNldERldkFjdGlvbihudWxsKTsKCiAg
ICAgIGlmIChpc0RlbW9saXNoZWQpIHsKICAgICAgICBjb25zdCBkZW1EYXRhID0geyAuLi5FTVBUWV9GT1JNLCBmbGFnZ2VkOiAi
RGVtb2xpc2hlZCIgfTsKICAgICAgICBzZXRTdXJ2ZXlzKHByZXYgPT4gKHsuLi5wcmV2LCBbdWlkXTogZGVtRGF0YX0pKTsKICAg
ICAgICBwZW5kaW5nU2F2ZVJlZi5jdXJyZW50ID0gc2F2ZVN1cnZleUVudHJ5KHVpZCwgInN1cnZleSIsIFN0cmluZyhuZXdJZCks
IGxuZywgbGF0LCB7IC4uLmRlbURhdGEsIGJhc2VtZW50OiAiTm8iIH0sIGN1ckxvY1JlZi5jdXJyZW50KS5jYXRjaChlcnIgPT4g
ewogICAgICAgICAgY29uc29sZS53YXJuKCJGYWlsZWQgdG8gY3JlYXRlIGRlbW9saXNoZWQgcG9pbnQgcm93OiIsIGVycik7CiAg
ICAgICAgfSkuZmluYWxseSgoKSA9PiB7IHBlbmRpbmdTYXZlUmVmLmN1cnJlbnQgPSBudWxsOyB9KTsKICAgICAgICBzZXRTZWxl
Y3RlZCh1aWQpOwogICAgICAgIHNldEZvcm0oZGVtRGF0YSk7CiAgICAgICAgaWYgKG1hcEluc3QuY3VycmVudCkgbWFwSW5zdC5j
dXJyZW50LnBhblRvKFtsYXQsIGxuZ10sIHthbmltYXRlOiB0cnVlfSk7CiAgICAgICAgc2hvd1RvYXN0KCLinJUgRGVtb2xpc2hl
ZCBzaXRlICMiICsgbmV3SWQgKyAiIG1hcmtlZCIpOwogICAgICB9IGVsc2UgewogICAgICAgIHBlbmRpbmdTYXZlUmVmLmN1cnJl
bnQgPSBzYXZlU3VydmV5RW50cnkodWlkLCAic3VydmV5IiwgU3RyaW5nKG5ld0lkKSwgbG5nLCBsYXQsIHsgLi4uRU1QVFlfRk9S
TSB9LCBjdXJMb2NSZWYuY3VycmVudCkuY2F0Y2goZXJyID0+IHsKICAgICAgICAgIGNvbnNvbGUud2FybigiRmFpbGVkIHRvIGNy
ZWF0ZSBzaGVldCByb3cgZm9yIG5ldyBwb2ludDoiLCBlcnIpOwogICAgICAgIH0pLmZpbmFsbHkoKCkgPT4geyBwZW5kaW5nU2F2
ZVJlZi5jdXJyZW50ID0gbnVsbDsgfSk7CiAgICAgICAgc2V0U2VsZWN0ZWQodWlkKTsKICAgICAgICBzZXRGb3JtKEVNUFRZX0ZP
Uk0pOwogICAgICAgIGlmIChtYXBJbnN0LmN1cnJlbnQpIG1hcEluc3QuY3VycmVudC5wYW5UbyhbbGF0LCBsbmddLCB7YW5pbWF0
ZTogdHJ1ZX0pOwogICAgICAgIHNob3dUb2FzdCgiUG9pbnQgIyIgKyBuZXdJZCArICIgYWRkZWQg4oCUIGF1dG8tc2F2aW5nLi4u
Iik7CiAgICAgIH0KICAgIH07CiAgICByZXR1cm4gKCkgPT4geyBkZWxldGUgd2luZG93Ll9fZGV2TW92ZTsgZGVsZXRlIHdpbmRv
dy5fX2RldkFkZDsgfTsKICB9LCBbc2hvd1RvYXN0LCBjb21taXREZXZFZGl0c10pOwoKICBjb25zdCBkZXZSZW1vdmUgPSBhc3lu
YyAoKSA9PiB7CiAgICBpZiAoIXNlbGVjdGVkKSByZXR1cm47CiAgICBjb25zdCBiID0gYnVpbGRpbmdzLmZpbmQoeCA9PiB4LnVp
ZCA9PT0gc2VsZWN0ZWQpOwogICAgY29uc3QgYkxhYmVsID0gYiA/IChiLm5zaUlkIHx8IGIudWlkKSA6IHNlbGVjdGVkOwogICAg
aWYgKCFjb25maXJtKCLimqDvuI8gUmVtb3ZlIGJ1aWxkaW5nICMiICsgYkxhYmVsICsgIj9cblxuVGhpcyB3aWxsIHBlcm1hbmVu
dGx5IGRlbGV0ZSB0aGUgcG9pbnQgYW5kIEFMTCBvZiBpdHMgZGF0YSAoc3VydmV5IGZpZWxkcywgbm90ZXMsIGZsYWcpIGZyb20g
dGhlIG1hcCBhbmQgdGhlIEdvb2dsZSBTaGVldC5cblxuVGhpcyBhY3Rpb24gY2Fubm90IGJlIHVuZG9uZS5cblxuQ29udGludWU/
IikpIHJldHVybjsKICAgIGNvbnN0IHVpZFRvUmVtb3ZlID0gc2VsZWN0ZWQ7CiAgICBjb25zdCBjdXIgPSBkZXZFZGl0c1JlZi5j
dXJyZW50OwogICAgY29uc3QgbmV3TW92ZWQgPSB7IC4uLmN1ci5tb3ZlZCB9OwogICAgZGVsZXRlIG5ld01vdmVkW3VpZFRvUmVt
b3ZlXTsKICAgIGNvbnN0IG5leHQgPSB7CiAgICAgIHJlbW92ZWQ6IFsuLi4oY3VyLnJlbW92ZWR8fFtdKSwgdWlkVG9SZW1vdmVd
LAogICAgICBtb3ZlZDogbmV3TW92ZWQsCiAgICAgIGFkZGVkOiAoY3VyLmFkZGVkfHxbXSkuZmlsdGVyKGEgPT4gYS51aWQgIT09
IHVpZFRvUmVtb3ZlKSwKICAgIH07CiAgICBjb21taXREZXZFZGl0cyhuZXh0KTsKICAgIHNldFN1cnZleXMocHJldiA9PiB7IGNv
bnN0IHUgPSB7Li4ucHJldn07IGRlbGV0ZSB1W3VpZFRvUmVtb3ZlXTsgcmV0dXJuIHU7IH0pOwogICAgc2V0U2VsZWN0ZWQobnVs
bCk7IHNldEZvcm0oRU1QVFlfRk9STSk7CiAgICAvLyBEZWxldGUgZnJvbSBzaGVldCBpbW1lZGlhdGVseSDigJQgZG9uJ3Qgd2Fp
dCBmb3IgZGVib3VuY2UKICAgIHRyeSB7CiAgICAgIGF3YWl0IGRlbGV0ZVN1cnZleUVudHJ5KHVpZFRvUmVtb3ZlLCBjdXJMb2NS
ZWYuY3VycmVudCk7CiAgICAgIHNob3dUb2FzdCgiUG9pbnQgIyIgKyBiTGFiZWwgKyAiIHJlbW92ZWQgJiBkZWxldGVkIGZyb20g
U2hlZXQiLCAiaW5mbyIpOwogICAgfSBjYXRjaCAoZXJyKSB7CiAgICAgIHNob3dUb2FzdCgiUG9pbnQgcmVtb3ZlZCBsb2NhbGx5
IGJ1dCBzaGVldCBkZWxldGUgZmFpbGVkOiAiICsgZXJyLm1lc3NhZ2UsICJlcnJvciIpOwogICAgfQogIH07CgogIGNvbnN0IGRl
dkR1cGxpY2F0ZSA9ICgpID0+IHsKICAgIGlmICghc2VsZWN0ZWQpIHJldHVybjsKICAgIGNvbnN0IGIgPSBidWlsZGluZ3MuZmlu
ZCh4ID0+IHgudWlkID09PSBzZWxlY3RlZCk7CiAgICBpZiAoIWIpIHJldHVybjsKICAgIGNvbnN0IGN1ciA9IGRldkVkaXRzUmVm
LmN1cnJlbnQ7CiAgICAvLyBGaW5kIG5leHQgYXZhaWxhYmxlIElECiAgICBsZXQgbWF4TmV3SWQgPSAxMDAwMDsKICAgIGJ1aWxk
aW5ncy5mb3JFYWNoKGJpID0+IHsKICAgICAgY29uc3QgaWQgPSBwYXJzZUludChiaS5uc2lJZCB8fCAoYmkudWlkLnN0YXJ0c1dp
dGgoIm5ldy0iKSA/IGJpLnVpZC5yZXBsYWNlKCJuZXctIiwiIikgOiAiMCIpKTsKICAgICAgaWYgKGlkID49IDEwMDAwICYmIGlk
ID4gbWF4TmV3SWQpIG1heE5ld0lkID0gaWQ7CiAgICB9KTsKICAgIChjdXIuYWRkZWR8fFtdKS5mb3JFYWNoKGEgPT4gewogICAg
ICBjb25zdCBpZCA9IHBhcnNlSW50KGEubnNpSWQgfHwgIjAiKTsKICAgICAgaWYgKGlkID49IDEwMDAwICYmIGlkID4gbWF4TmV3
SWQpIG1heE5ld0lkID0gaWQ7CiAgICB9KTsKICAgIGNvbnN0IG5ld0lkID0gbWF4TmV3SWQgKyAxOwogICAgY29uc3QgdWlkID0g
Im5ldy0iICsgbmV3SWQ7CiAgICAvLyBPZmZzZXQgc2xpZ2h0bHkgdG8gdGhlIHJpZ2h0ICh+MTVtIGF0IG1pZC1sYXRpdHVkZXMp
CiAgICBjb25zdCBvZmZzZXRMbmcgPSBiLmxuZyArIDAuMDAwMTU7CiAgICBjb25zdCBuZXdQdCA9IHsgdWlkLCB0eXBlOiAic3Vy
dmV5IiwgbnNpSWQ6IFN0cmluZyhuZXdJZCksIGxuZzogb2Zmc2V0TG5nLCBsYXQ6IGIubGF0LCBwcmVmaWxsOiBudWxsIH07CiAg
ICBjb25zdCBuZXh0ID0geyAuLi5jdXIsIGFkZGVkOiBbLi4uKGN1ci5hZGRlZHx8W10pLCBuZXdQdF0gfTsKICAgIGNvbW1pdERl
dkVkaXRzKG5leHQpOwogICAgLy8gQ29weSBhbGwgY3VycmVudCBkYXRhIGZyb20gdGhlIHNvdXJjZSBidWlsZGluZwogICAgY29u
c3Qgc3YgPSBzdXJ2ZXlzUmVmLmN1cnJlbnRbc2VsZWN0ZWRdOwogICAgY29uc3QgcGYgPSBiLnByZWZpbGwgfHwge307CiAgICBj
b25zdCBzb3VyY2VEYXRhID0geyAuLi5FTVBUWV9GT1JNLCAuLi4oc3YgfHwgcGYpIH07CiAgICAvLyBDbGVhciBzYXZlZEF0IHNv
IHRoZSBkdXBsaWNhdGUgc3RhcnRzIGFzIHVuc2F2ZWQKICAgIGNvbnN0IGR1cERhdGEgPSB7IC4uLnNvdXJjZURhdGEsIHNhdmVk
QXQ6ICIiIH07CiAgICBzZXRTdXJ2ZXlzKHByZXYgPT4gKHsuLi5wcmV2LCBbdWlkXTogZHVwRGF0YX0pKTsKICAgIC8vIENyZWF0
ZSBzaGVldCByb3cgd2l0aCB0aGUgZHVwbGljYXRlZCBkYXRhCiAgICBjb25zdCBiYXNlbWVudCA9IChkdXBEYXRhLmZvdW5kYXRp
b25UeXBlIHx8ICIiKS50b1VwcGVyQ2FzZSgpID09PSAiQiIgPyAiWWVzIiA6ICJObyI7CiAgICBwZW5kaW5nU2F2ZVJlZi5jdXJy
ZW50ID0gc2F2ZVN1cnZleUVudHJ5KHVpZCwgInN1cnZleSIsIFN0cmluZyhuZXdJZCksIG9mZnNldExuZywgYi5sYXQsIHsgLi4u
ZHVwRGF0YSwgYmFzZW1lbnQgfSwgY3VyTG9jUmVmLmN1cnJlbnQpLmNhdGNoKGVyciA9PiB7CiAgICAgIGNvbnNvbGUud2Fybigi
RmFpbGVkIHRvIGNyZWF0ZSBzaGVldCByb3cgZm9yIGR1cGxpY2F0ZWQgcG9pbnQ6IiwgZXJyKTsKICAgIH0pLmZpbmFsbHkoKCkg
PT4geyBwZW5kaW5nU2F2ZVJlZi5jdXJyZW50ID0gbnVsbDsgfSk7CiAgICAvLyBTZWxlY3QgdGhlIG5ldyBkdXBsaWNhdGUKICAg
IHNldFNlbGVjdGVkKHVpZCk7CiAgICBzZXRGb3JtKGR1cERhdGEpOwogICAgaWYgKG1hcEluc3QuY3VycmVudCkgbWFwSW5zdC5j
dXJyZW50LnBhblRvKFtiLmxhdCwgb2Zmc2V0TG5nXSwge2FuaW1hdGU6IHRydWV9KTsKICAgIHNob3dUb2FzdCgiUG9pbnQgIyIg
KyBuZXdJZCArICIgZHVwbGljYXRlZCBmcm9tICIgKyAoYi5uc2lJZCB8fCBiLnVpZCkpOwogIH07CgogIGNvbnN0IGRldlJlc2V0
QWxsID0gYXN5bmMgKCkgPT4gewogICAgaWYgKGRlYm91bmNlVGltZXJSZWYuY3VycmVudCkgY2xlYXJUaW1lb3V0KGRlYm91bmNl
VGltZXJSZWYuY3VycmVudCk7CiAgICBzYXZpbmdSZWYuY3VycmVudCA9IHRydWU7CiAgICB0cnkgewogICAgICBjb25zdCBmcmVz
aCA9IHsgcmVtb3ZlZDogW10sIG1vdmVkOiB7fSwgYWRkZWQ6IFtdLCBfdHM6IERhdGUubm93KCkgfTsKICAgICAgZGV2RWRpdHNS
ZWYuY3VycmVudCA9IGZyZXNoOwogICAgICBzZXREZXZFZGl0cyhmcmVzaCk7CiAgICAgIHNldEJ1aWxkaW5ncyhhcHBseURldkVk
aXRzKGJhc2VCdWlsZGluZ3NSZWYuY3VycmVudCwgZnJlc2gpKTsKICAgICAgc2F2ZUxvY2FsRGV2KGZyZXNoKTsKICAgICAgc2V0
U2VsZWN0ZWQobnVsbCk7IHNldEZvcm0oRU1QVFlfRk9STSk7CiAgICAgIGF3YWl0IHNhdmVEZXZFZGl0c1JlbW90ZShmcmVzaCwg
Y3VyTG9jUmVmLmN1cnJlbnQpOwogICAgICBzZXREZXZEaXJ0eShmYWxzZSk7CiAgICAgIHNob3dUb2FzdCgiQWxsIGRldiBlZGl0
cyByZXNldCAmIHN5bmNlZCIsICJpbmZvIik7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgc2hvd1RvYXN0KCJSZXNldCBmYWls
ZWQ6ICIgKyBlcnIubWVzc2FnZSwgImVycm9yIik7CiAgICB9IGZpbmFsbHkgewogICAgICBzYXZpbmdSZWYuY3VycmVudCA9IGZh
bHNlOwogICAgfQogIH07CgogIGNvbnN0IGRldlN5bmNUb1NlcnZlciA9IGFzeW5jICgpID0+IHsKICAgIGlmIChkZWJvdW5jZVRp
bWVyUmVmLmN1cnJlbnQpIGNsZWFyVGltZW91dChkZWJvdW5jZVRpbWVyUmVmLmN1cnJlbnQpOwogICAgc2F2aW5nUmVmLmN1cnJl
bnQgPSB0cnVlOwogICAgc2V0RGV2U3luY2luZyh0cnVlKTsKICAgIHRyeSB7CiAgICAgIGNvbnN0IGN1ciA9IGRldkVkaXRzUmVm
LmN1cnJlbnQ7CiAgICAgIGNvbnN0IGN1ckJ1aWxkaW5ncyA9IGFwcGx5RGV2RWRpdHMoYmFzZUJ1aWxkaW5nc1JlZi5jdXJyZW50
LCBjdXIpOwogICAgICBhd2FpdCBzYXZlRGV2RWRpdHNSZW1vdGUoY3VyLCBjdXJMb2NSZWYuY3VycmVudCk7CiAgICAgIGZvciAo
Y29uc3QgdWlkIG9mIChjdXIucmVtb3ZlZHx8W10pKSB7CiAgICAgICAgYXdhaXQgZGVsZXRlU3VydmV5RW50cnkodWlkLCBjdXJM
b2NSZWYuY3VycmVudCk7CiAgICAgIH0KICAgICAgZm9yIChjb25zdCB1aWQgaW4gKGN1ci5tb3ZlZHx8e30pKSB7CiAgICAgICAg
Y29uc3QgYiA9IGN1ckJ1aWxkaW5ncy5maW5kKHggPT4geC51aWQgPT09IHVpZCk7CiAgICAgICAgaWYgKCFiKSBjb250aW51ZTsK
ICAgICAgICBjb25zdCBzID0gc3VydmV5c1JlZi5jdXJyZW50W3VpZF0gfHwgKGIucHJlZmlsbCA/IHsuLi5iLnByZWZpbGx9IDog
bnVsbCk7CiAgICAgICAgaWYgKHMpIHsKICAgICAgICAgIGF3YWl0IHNhdmVTdXJ2ZXlFbnRyeSh1aWQsIGIudHlwZSwgYi5uc2lJ
ZCwgYi5sbmcsIGIubGF0LCB7Li4ucywgc2F2ZWRBdDogcy5zYXZlZEF0IHx8ICIifSwgY3VyTG9jUmVmLmN1cnJlbnQpOwogICAg
ICAgIH0KICAgICAgfQogICAgICBzZXREZXZEaXJ0eShmYWxzZSk7CiAgICAgIHNob3dUb2FzdCgiRGV2IGVkaXRzIHN5bmNlZCB0
byBHb29nbGUgU2hlZXQhIik7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgc2hvd1RvYXN0KCJEZXYgc3luYyBmYWlsZWQ6ICIg
KyBlcnIubWVzc2FnZSwgImVycm9yIik7CiAgICB9IGZpbmFsbHkgewogICAgICBzZXREZXZTeW5jaW5nKGZhbHNlKTsKICAgICAg
c2F2aW5nUmVmLmN1cnJlbnQgPSBmYWxzZTsKICAgIH0KICB9OwoKICBjb25zdCBbZmxhZ2dpbmcsIHNldEZsYWdnaW5nXSA9IHVz
ZVN0YXRlKGZhbHNlKTsKICBjb25zdCB0b2dnbGVGbGFnID0gYXN5bmMgKCkgPT4gewogICAgaWYgKCFzZWxlY3RlZCB8fCBmbGFn
Z2luZyB8fCBzYXZpbmcpIHJldHVybjsKICAgIGNvbnN0IGIgPSBidWlsZGluZ3MuZmluZCh4ID0+IHgudWlkID09PSBzZWxlY3Rl
ZCk7CiAgICBpZiAoIWIpIHJldHVybjsKICAgIGNvbnN0IG5ld0ZsYWdnZWQgPSBmb3JtLmZsYWdnZWQgPT09ICJZZXMiID8gIiIg
OiAiWWVzIjsKICAgIHNhdmluZ1JlZi5jdXJyZW50ID0gdHJ1ZTsKICAgIGNvbnN0IGN1cnJlbnRGb3JtID0geyAuLi5mb3JtLCBm
bGFnZ2VkOiBuZXdGbGFnZ2VkIH07CiAgICAvLyBVcGRhdGUgVUkgaW1tZWRpYXRlbHkKICAgIHNldEZvcm0oZiA9PiAoey4uLmYs
IGZsYWdnZWQ6IG5ld0ZsYWdnZWR9KSk7CiAgICBzZXRTdXJ2ZXlzKHByZXYgPT4gewogICAgICBjb25zdCBleGlzdGluZyA9IHBy
ZXZbc2VsZWN0ZWRdIHx8IHt9OwogICAgICByZXR1cm4gey4uLnByZXYsIFtzZWxlY3RlZF06IHsuLi5leGlzdGluZywgZmxhZ2dl
ZDogbmV3RmxhZ2dlZH19OwogICAgfSk7CiAgICBzZXRGbGFnZ2luZyh0cnVlKTsKICAgIHRyeSB7CiAgICAgIC8vIFdhaXQgZm9y
IGFueSBpbi1mbGlnaHQgcm93IGNyZWF0aW9uIHRvIGNvbXBsZXRlIGZpcnN0CiAgICAgIGlmIChwZW5kaW5nU2F2ZVJlZi5jdXJy
ZW50KSB7CiAgICAgICAgYXdhaXQgcGVuZGluZ1NhdmVSZWYuY3VycmVudDsKICAgICAgICBwZW5kaW5nU2F2ZVJlZi5jdXJyZW50
ID0gbnVsbDsKICAgICAgfQogICAgICAvLyBUcnkgdGhlIHBhdGNoLW9ubHkgYXBwcm9hY2ggZmlyc3QKICAgICAgbGV0IHBhdGNo
U3VjY2VlZGVkID0gZmFsc2U7CiAgICAgIHRyeSB7CiAgICAgICAgY29uc3QgcmVzdWx0ID0gYXdhaXQgdXBkYXRlRmxhZ0VudHJ5
KHNlbGVjdGVkLCBuZXdGbGFnZ2VkLCBjdXJMb2NSZWYuY3VycmVudCk7CiAgICAgICAgcGF0Y2hTdWNjZWVkZWQgPSByZXN1bHQg
JiYgcmVzdWx0Lm9rICYmICFyZXN1bHQuZmFsbGJhY2sgJiYgIXJlc3VsdC5uZWVkc1JvdzsKICAgICAgfSBjYXRjaCAoZSkgewog
ICAgICAgIC8vIHBhdGNoIGZhaWxlZCDigJQgZmFsbCB0aHJvdWdoIHRvIGZ1bGwgc2F2ZQogICAgICB9CiAgICAgIC8vIElmIHBh
dGNoIGRpZG4ndCBzdWNjZWVkIGZvciBhbnkgcmVhc29uLCBkbyBhIGZ1bGwgcm93IHNhdmUgdG8gZ3VhcmFudGVlIHBlcnNpc3Rl
bmNlCiAgICAgIGlmICghcGF0Y2hTdWNjZWVkZWQpIHsKICAgICAgICBjb25zdCBiYXNlbWVudCA9IChjdXJyZW50Rm9ybS5mb3Vu
ZGF0aW9uVHlwZSB8fCAiIikudG9VcHBlckNhc2UoKSA9PT0gIkIiID8gIlllcyIgOiAiTm8iOwogICAgICAgIGNvbnN0IGVudHJ5
ID0geyAuLi5jdXJyZW50Rm9ybSwgYmFzZW1lbnQgfTsKICAgICAgICBhd2FpdCBzYXZlU3VydmV5RW50cnkoc2VsZWN0ZWQsIGIu
dHlwZSwgYi5uc2lJZCwgYi5sbmcsIGIubGF0LCBlbnRyeSwgY3VyTG9jUmVmLmN1cnJlbnQpOwogICAgICAgIHNldFN1cnZleXMo
cHJldiA9PiAoey4uLnByZXYsIFtzZWxlY3RlZF06IGVudHJ5fSkpOwogICAgICB9CiAgICAgIHNob3dUb2FzdChuZXdGbGFnZ2Vk
ID09PSAiWWVzIiA/ICLwn5qpIEZsYWdnZWQgZm9yIHNpdGUgdmlzaXQiIDogIkZsYWcgcmVtb3ZlZCIpOwogICAgfSBjYXRjaCAo
ZXJyKSB7CiAgICAgIHNob3dUb2FzdCgiRmxhZyBzeW5jIGZhaWxlZDogIiArIGVyci5tZXNzYWdlLCAiZXJyb3IiKTsKICAgICAg
Ly8gUmV2ZXJ0IFVJIG9uIHRvdGFsIGZhaWx1cmUKICAgICAgY29uc3Qgb2xkRmxhZ2dlZCA9IG5ld0ZsYWdnZWQgPT09ICJZZXMi
ID8gIiIgOiAiWWVzIjsKICAgICAgc2V0Rm9ybShmID0+ICh7Li4uZiwgZmxhZ2dlZDogb2xkRmxhZ2dlZH0pKTsKICAgICAgc2V0
U3VydmV5cyhwcmV2ID0+IHsKICAgICAgICBjb25zdCBleGlzdGluZyA9IHByZXZbc2VsZWN0ZWRdIHx8IHt9OwogICAgICAgIHJl
dHVybiB7Li4ucHJldiwgW3NlbGVjdGVkXTogey4uLmV4aXN0aW5nLCBmbGFnZ2VkOiBvbGRGbGFnZ2VkfX07CiAgICAgIH0pOwog
ICAgfSBmaW5hbGx5IHsKICAgICAgc2V0RmxhZ2dpbmcoZmFsc2UpOwogICAgICBzYXZpbmdSZWYuY3VycmVudCA9IGZhbHNlOwog
ICAgfQogIH07CgogIGNvbnN0IFtzYXZpbmdOb3Rlcywgc2V0U2F2aW5nTm90ZXNdID0gdXNlU3RhdGUoZmFsc2UpOwogIGNvbnN0
IHNhdmVOb3RlcyA9IGFzeW5jICgpID0+IHsKICAgIGlmICghc2VsZWN0ZWQgfHwgc2F2aW5nTm90ZXMpIHJldHVybjsKICAgIGNv
bnN0IGIgPSBidWlsZGluZ3MuZmluZCh4ID0+IHgudWlkID09PSBzZWxlY3RlZCk7CiAgICBpZiAoIWIpIHJldHVybjsKICAgIHNh
dmluZ1JlZi5jdXJyZW50ID0gdHJ1ZTsKICAgIHNldFNhdmluZ05vdGVzKHRydWUpOwogICAgdHJ5IHsKICAgICAgaWYgKHBlbmRp
bmdTYXZlUmVmLmN1cnJlbnQpIHsKICAgICAgICBhd2FpdCBwZW5kaW5nU2F2ZVJlZi5jdXJyZW50OwogICAgICAgIHBlbmRpbmdT
YXZlUmVmLmN1cnJlbnQgPSBudWxsOwogICAgICB9CiAgICAgIC8vIE1lcmdlIGN1cnJlbnQgbm90ZXMgaW50byBleGlzdGluZyBz
dXJ2ZXkgZGF0YSB3aXRob3V0IHRvdWNoaW5nIHNhdmVkQXQKICAgICAgY29uc3Qgc3YgPSBzdXJ2ZXlzUmVmLmN1cnJlbnRbc2Vs
ZWN0ZWRdOwogICAgICBjb25zdCBwZiA9IGIucHJlZmlsbCB8fCB7fTsKICAgICAgY29uc3QgY3VycmVudCA9IHsgLi4uRU1QVFlf
Rk9STSwgLi4uKHN2IHx8IHBmKSB9OwogICAgICBjb25zdCBlbnRyeSA9IHsgLi4uY3VycmVudCwgbm90ZXM6IGZvcm0ubm90ZXMg
fTsKICAgICAgY29uc3QgYmFzZW1lbnQgPSAoZW50cnkuZm91bmRhdGlvblR5cGUgfHwgIiIpLnRvVXBwZXJDYXNlKCkgPT09ICJC
IiA/ICJZZXMiIDogIk5vIjsKICAgICAgYXdhaXQgc2F2ZVN1cnZleUVudHJ5KHNlbGVjdGVkLCBiLnR5cGUsIGIubnNpSWQsIGIu
bG5nLCBiLmxhdCwgeyAuLi5lbnRyeSwgYmFzZW1lbnQgfSwgY3VyTG9jUmVmLmN1cnJlbnQpOwogICAgICBzZXRTdXJ2ZXlzKHBy
ZXYgPT4gKHsuLi5wcmV2LCBbc2VsZWN0ZWRdOiBlbnRyeX0pKTsKICAgICAgc2hvd1RvYXN0KCLwn5OdIE5vdGVzIHNhdmVkIik7
CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgc2hvd1RvYXN0KCJOb3RlcyBzYXZlIGZhaWxlZDogIiArIGVyci5tZXNzYWdlLCAi
ZXJyb3IiKTsKICAgIH0gZmluYWxseSB7CiAgICAgIHNldFNhdmluZ05vdGVzKGZhbHNlKTsKICAgICAgc2F2aW5nUmVmLmN1cnJl
bnQgPSBmYWxzZTsKICAgIH0KICB9OwoKICBjb25zdCBoYW5kbGVTYXZlID0gYXN5bmMgKCkgPT4gewogICAgaWYgKGZsYWdnaW5n
KSByZXR1cm47CiAgICBjb25zdCByZXF1aXJlZCA9IFsKICAgICAgeyBrZXk6ICJvY2NUeXBlIiwgICAgICAgICAgbGFiZWw6ICJP
Y2N1cGFuY3kgVHlwZSIgfSwKICAgICAgeyBrZXk6ICJudW1TdG9yaWVzIiwgICAgICAgbGFiZWw6ICJTdG9yaWVzIiB9LAogICAg
ICB7IGtleTogImJ1aWxkaW5nVHlwZSIsICAgICBsYWJlbDogIkJ1aWxkaW5nIFR5cGUiIH0sCiAgICAgIHsga2V5OiAiZm91bmRh
dGlvblR5cGUiLCAgIGxhYmVsOiAiRm91bmRhdGlvbiBUeXBlIiB9LAogICAgICB7IGtleTogImZpcnN0Rmxvb3JIZWlnaHQiLCBs
YWJlbDogIjFzdCBGbG9vciBIZWlnaHQiIH0sCiAgICAgIHsga2V5OiAiYXJlYSIsICAgICAgICAgICAgIGxhYmVsOiAiRm9vdHBy
aW50IChzcWZ0KSIgfSwKICAgICAgeyBrZXk6ICJncm91bmRFbGV2IiwgICAgICAgbGFiZWw6ICJHcm91bmQgRWxldmF0aW9uIiB9
LAogICAgICB7IGtleTogInN0cnVjdHVyZVZhbHVlIiwgICBsYWJlbDogIlN0cnVjdHVyZSBWYWx1ZSIgfSwKICAgICAgeyBrZXk6
ICJjb250ZW50VmFsdWUiLCAgICAgbGFiZWw6ICJDb250ZW50IFZhbHVlIiB9LAogICAgXTsKICAgIGNvbnN0IG1pc3NpbmcgPSBy
ZXF1aXJlZC5maWx0ZXIoZiA9PiAhZm9ybVtmLmtleV0gJiYgZm9ybVtmLmtleV0gIT09IDApLm1hcChmID0+IGYubGFiZWwpOwog
ICAgaWYgKG1pc3NpbmcubGVuZ3RoKSB7CiAgICAgIHNob3dUb2FzdCgiUmVxdWlyZWQ6ICIgKyBtaXNzaW5nLmpvaW4oIiwgIiks
ICJlcnJvciIpOyByZXR1cm47CiAgICB9CiAgICBzYXZpbmdSZWYuY3VycmVudCA9IHRydWU7CiAgICBzZXRTYXZpbmcodHJ1ZSk7
CiAgICBjb25zdCBiID0gYnVpbGRpbmdzLmZpbmQoeCA9PiB4LnVpZCA9PT0gc2VsZWN0ZWQpOwogICAgY29uc3QgYmFzZW1lbnQg
PSBmb3JtLmZvdW5kYXRpb25UeXBlID09PSAiQiIgPyAiWWVzIiA6ICJObyI7CiAgICBjb25zdCBlbnRyeSA9IHsuLi5mb3JtLCBi
YXNlbWVudCwgc2F2ZWRBdDogbmV3IERhdGUoKS50b0lTT1N0cmluZygpLCBmbGFnZ2VkOiBmb3JtLmZsYWdnZWQgfHwgIiJ9Owog
ICAgdHJ5IHsKICAgICAgYXdhaXQgc2F2ZVN1cnZleUVudHJ5KHNlbGVjdGVkLCBiLnR5cGUsIGIubnNpSWQsIGIubG5nLCBiLmxh
dCwgZW50cnksIGN1ckxvY1JlZi5jdXJyZW50KTsKICAgICAgc2V0U3VydmV5cyhwcmV2ID0+ICh7Li4ucHJldiwgW3NlbGVjdGVk
XTogZW50cnl9KSk7CiAgICAgIHNob3dUb2FzdCgiU2F2ZWQgJiBzeW5jZWQhIik7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAg
c2hvd1RvYXN0KCJTYXZlIGZhaWxlZDogIiArIGVyci5tZXNzYWdlLCAiZXJyb3IiKTsKICAgICAgY29uc29sZS5lcnJvcigiU2F2
ZSBlcnJvcjoiLCBlcnIpOwogICAgfSBmaW5hbGx5IHsKICAgICAgc2V0U2F2aW5nKGZhbHNlKTsKICAgICAgc2F2aW5nUmVmLmN1
cnJlbnQgPSBmYWxzZTsKICAgIH0KICB9OwoKICBjb25zdCBoYW5kbGVVbmRvU2F2ZSA9IGFzeW5jICgpID0+IHsKICAgIGlmIChz
ZWxlY3RlZCA9PSBudWxsKSByZXR1cm47CiAgICBpZiAoIWNvbmZpcm0oIuKaoO+4jyBXYXJuaW5nOiBUaGlzIHdpbGwgY2xlYXIg
YWxsIHN1cnZleSBkYXRhIGZvciB0aGlzIGJ1aWxkaW5nIChub3Rlcywgc3VydmV5b3IsIHRpbWVzdGFtcCkuIFRoZSBzaXRlIHZp
c2l0IGZsYWcgd2lsbCBiZSBwcmVzZXJ2ZWQuXG5cblRoZSBwb2ludCB3aWxsIGJlIHJlc2V0IHRvIGl0cyBvcmlnaW5hbCBwcmUt
c3VydmV5IHN0YXRlLlxuXG5BcmUgeW91IHN1cmUgeW91IHdhbnQgdG8gY29udGludWU/IikpIHJldHVybjsKICAgIHNhdmluZ1Jl
Zi5jdXJyZW50ID0gdHJ1ZTsKICAgIHRyeSB7CiAgICAgIGNvbnN0IGIgPSBidWlsZGluZ3MuZmluZCh4ID0+IHgudWlkID09PSBz
ZWxlY3RlZCk7CiAgICAgIGNvbnN0IHBmID0gYj8ucHJlZmlsbCB8fCB7fTsKICAgICAgY29uc3QgY3VycmVudEZsYWdnZWQgPSAo
c3VydmV5c1JlZi5jdXJyZW50W3NlbGVjdGVkXSB8fCBwZikuZmxhZ2dlZCB8fCAiIjsKICAgICAgLy8gUmVzZXQgdG8gcHJlZmls
bCBkYXRhIHdpdGggZW1wdHkgc2F2ZWRBdC9zdXJ2ZXlvci9ub3RlcywgYnV0IHByZXNlcnZlIGZsYWcKICAgICAgY29uc3QgcmVz
ZXRFbnRyeSA9IHsKICAgICAgICBvY2NUeXBlOiBwZi5vY2NUeXBlIHx8ICIiLCBidWlsZGluZ1R5cGU6IHBmLmJ1aWxkaW5nVHlw
ZSB8fCAiIiwKICAgICAgICBudW1TdG9yaWVzOiBwZi5udW1TdG9yaWVzIHx8ICIiLCBhcmVhOiBwZi5hcmVhIHx8ICIiLAogICAg
ICAgIGZvdW5kYXRpb25UeXBlOiBwZi5mb3VuZGF0aW9uVHlwZSB8fCAiIiwgZmlyc3RGbG9vckhlaWdodDogcGYuZmlyc3RGbG9v
ckhlaWdodCB8fCAiIiwKICAgICAgICB5ZWFyQnVpbHQ6IHBmLnllYXJCdWlsdCB8fCAiIiwgZ3JvdW5kRWxldjogcGYuZ3JvdW5k
RWxldiB8fCAiIiwKICAgICAgICBhZGRyZXNzOiBwZi5hZGRyZXNzIHx8ICIiLAogICAgICAgIHN0cnVjdHVyZVZhbHVlOiBwZi5z
dHJ1Y3R1cmVWYWx1ZSB8fCAiIiwgY29udGVudFZhbHVlOiBwZi5jb250ZW50VmFsdWUgfHwgIiIsCiAgICAgICAgYmFzZW1lbnQ6
IHBmLmZvdW5kYXRpb25UeXBlID09PSAiQiIgPyAiWWVzIiA6ICJObyIsIG5vdGVzOiAiIiwgc3VydmV5b3I6ICIiLCBzYXZlZEF0
OiAiIiwgZmxhZ2dlZDogY3VycmVudEZsYWdnZWQsCiAgICAgIH07CiAgICAgIC8vIFVwZGF0ZSB0aGUgc2hlZXQgcm93IChrZWVw
IGJ1aWxkaW5nIGluIHNoZWV0IGJ1dCBjbGVhciB0aGUgc3VydmV5KQogICAgICBhd2FpdCBzYXZlU3VydmV5RW50cnkoc2VsZWN0
ZWQsIGIudHlwZSwgYi5uc2lJZCwgYi5sbmcsIGIubGF0LCByZXNldEVudHJ5LCBjdXJMb2NSZWYuY3VycmVudCk7CiAgICAgIHNl
dFN1cnZleXMocHJldiA9PiAoey4uLnByZXYsIFtzZWxlY3RlZF06IHJlc2V0RW50cnl9KSk7CiAgICAgIHNldEZvcm0oey4uLkVN
UFRZX0ZPUk0sIC4uLnJlc2V0RW50cnl9KTsKICAgICAgc2hvd1RvYXN0KCJTdXJ2ZXkgdW5kb25lIOKAlCBwb2ludCByZXNldCIs
ICJpbmZvIik7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgc2hvd1RvYXN0KCJVbmRvIGZhaWxlZDogIiArIGVyci5tZXNzYWdl
LCAiZXJyb3IiKTsKICAgIH0gZmluYWxseSB7CiAgICAgIHNhdmluZ1JlZi5jdXJyZW50ID0gZmFsc2U7CiAgICB9CiAgfTsKCiAg
Y29uc3QgbmF2aWdhdGVUb0J1aWxkaW5nID0gKCkgPT4gewogICAgaWYgKCFzZWxlY3RlZCkgcmV0dXJuOwogICAgY29uc3QgYiA9
IGJ1aWxkaW5ncy5maW5kKHggPT4geC51aWQgPT09IHNlbGVjdGVkKTsKICAgIHdpbmRvdy5vcGVuKGBodHRwczovL3d3dy5nb29n
bGUuY29tL21hcHMvZGlyLz9hcGk9MSZkZXN0aW5hdGlvbj0ke2IubGF0fSwke2IubG5nfSZ0cmF2ZWxtb2RlPWRyaXZpbmdgLCAi
X2JsYW5rIik7CiAgfTsKCiAgY29uc3QgZ29Ub05lYXJlc3QgPSAoKSA9PiB7CiAgICBpZiAoIW5hdmlnYXRvci5nZW9sb2NhdGlv
bikgeyBzaG93VG9hc3QoIk5vIGdlb2xvY2F0aW9uIiwgImVycm9yIik7IHJldHVybjsgfQogICAgbmF2aWdhdG9yLmdlb2xvY2F0
aW9uLmdldEN1cnJlbnRQb3NpdGlvbihwb3MgPT4gewogICAgICBjb25zdCB7bGF0aXR1ZGUsIGxvbmdpdHVkZX0gPSBwb3MuY29v
cmRzOwogICAgICBsZXQgbWluRCA9IEluZmluaXR5LCBuZWFyID0gbnVsbDsKICAgICAgYnVpbGRpbmdzLmZvckVhY2goYiA9PiB7
CiAgICAgICAgaWYgKHN1cnZleXNbYi51aWRdICYmIHN1cnZleXNbYi51aWRdLnNhdmVkQXQpIHJldHVybjsKICAgICAgICBjb25z
dCBkID0gTWF0aC5zcXJ0KChiLmxhdC1sYXRpdHVkZSkqKjIgKyAoYi5sbmctbG9uZ2l0dWRlKSoqMik7CiAgICAgICAgaWYgKGQg
PCBtaW5EKSB7IG1pbkQgPSBkOyBuZWFyID0gYi51aWQ7IH0KICAgICAgfSk7CiAgICAgIGlmIChuZWFyKSB3aW5kb3cuX19zZWwo
bmVhcik7CiAgICB9LCAoKSA9PiBzaG93VG9hc3QoIkxvY2F0aW9uIGRlbmllZCIsICJlcnJvciIpKTsKICB9OwoKICBjb25zdCBl
eHBvcnRYTFNYID0gKCkgPT4gewogICAgLy8gRXhwb3J0IGFsbCBidWlsZGluZ3MgRVhDRVBUOiBmbGFnZ2VkICJZZXMiLCBmbGFn
Z2VkICJEZW1vbGlzaGVkIiwKICAgIC8vIG9yIHVuc2F2ZWQgbmV3IChyZWQpIHBvaW50cyAoc3VydmV5X3R5cGUgPT09ICJzdXJ2
ZXkiIHdpdGggbm8gc2F2ZWRBdCkuCiAgICBjb25zdCBleHBvcnRCdWlsZGluZ3MgPSBidWlsZGluZ3MuZmlsdGVyKGIgPT4gewog
ICAgICBjb25zdCBzdiA9IHN1cnZleXNbYi51aWRdOwogICAgICBjb25zdCBwZiA9IGIucHJlZmlsbCB8fCB7fTsKICAgICAgY29u
c3QgZCA9IHN2IHx8IHBmOwogICAgICBpZiAoZC5mbGFnZ2VkID09PSAiWWVzIiB8fCBkLmZsYWdnZWQgPT09ICJEZW1vbGlzaGVk
IikgcmV0dXJuIGZhbHNlOwogICAgICBpZiAoYi50eXBlID09PSAic3VydmV5IiAmJiAhKHN2ICYmIHN2LnNhdmVkQXQpKSByZXR1
cm4gZmFsc2U7CiAgICAgIHJldHVybiB0cnVlOwogICAgfSk7CiAgICBpZiAoZXhwb3J0QnVpbGRpbmdzLmxlbmd0aCA9PT0gMCkg
eyBzaG93VG9hc3QoIk5vIGJ1aWxkaW5ncyB0byBleHBvcnQiLCAiZXJyb3IiKTsgcmV0dXJuOyB9CiAgICBjb25zdCBoZWFkZXJz
ID0gWyJJRCIsIm9jY3VwYW5jeV90eXBlIiwiYnVpbGRpbmdfdHlwZSIsIm51bWJlcl9vZl9zdG9yaWVzIiwiYXJlYSIsImZvdW5k
YXRpb25fdHlwZSIsImZvdW5kYXRpb25faGVpZ2h0IiwieWVhcl9idWlsdCIsImdyb3VuZF9lbGV2YXRpb24iLCJhZGRyZXNzIiwi
bG9uZ2l0dWRlIiwibGF0aXR1ZGUiLCJzdHJ1Y3R1cmVfdmFsdWUiLCJjb250ZW50X3ZhbHVlIl07CiAgICBjb25zdCBudW0gPSB2
ID0+IHsgY29uc3QgbiA9IHBhcnNlRmxvYXQodik7IHJldHVybiBpc0Zpbml0ZShuKSA/IG4gOiAiIjsgfTsKICAgIGNvbnN0IHJv
d3MgPSBleHBvcnRCdWlsZGluZ3MubWFwKGIgPT4gewogICAgICBjb25zdCBzYXZlZCA9IHN1cnZleXNbYi51aWRdOwogICAgICBj
b25zdCBwZiA9IGIucHJlZmlsbCB8fCB7fTsKICAgICAgY29uc3QgcyA9IHNhdmVkIHx8IHBmOwogICAgICByZXR1cm4gWwogICAg
ICAgIG51bShiLm5zaUlkIHx8ICIiKSwKICAgICAgICBzLm9jY1R5cGUgfHwgIiIsIHMuYnVpbGRpbmdUeXBlIHx8ICIiLAogICAg
ICAgIG51bShzLm51bVN0b3JpZXMgfHwgIiIpLCBudW0ocy5hcmVhIHx8ICIiKSwKICAgICAgICBzLmZvdW5kYXRpb25UeXBlIHx8
ICIiLCBudW0ocy5maXJzdEZsb29ySGVpZ2h0IHx8ICIiKSwgbnVtKHMueWVhckJ1aWx0IHx8ICIiKSwgbnVtKHMuZ3JvdW5kRWxl
diB8fCAiIiksCiAgICAgICAgcy5hZGRyZXNzIHx8ICIiLCBudW0oYi5sbmcpLCBudW0oYi5sYXQpLAogICAgICAgIG51bShzLnN0
cnVjdHVyZVZhbHVlIHx8ICIiKSwgbnVtKHMuY29udGVudFZhbHVlIHx8ICIiKQogICAgICBdOwogICAgfSk7CiAgICBjb25zdCB3
cyA9IFhMU1gudXRpbHMuYW9hX3RvX3NoZWV0KFtoZWFkZXJzLCAuLi5yb3dzXSk7CiAgICBjb25zdCB3YiA9IFhMU1gudXRpbHMu
Ym9va19uZXcoKTsKICAgIFhMU1gudXRpbHMuYm9va19hcHBlbmRfc2hlZXQod2IsIHdzLCAiU2hlZXQxIik7CiAgICBjb25zdCBs
b2NOYW1lID0gTE9DQVRJT05TW2N1ckxvY1JlZi5jdXJyZW50XS5uYW1lLnJlcGxhY2UoL1xzKy9nLCAiIik7CiAgICBYTFNYLndy
aXRlRmlsZSh3YiwgIkRERF9fXyIgKyBsb2NOYW1lICsgIl9fX05TSS54bHN4Iik7CiAgICBzaG93VG9hc3QoIkV4cG9ydGVkICIg
KyBleHBvcnRCdWlsZGluZ3MubGVuZ3RoICsgIiBidWlsZGluZ3MiKTsKICB9OwoKICBjb25zdCBbc2hlZXRTeW5jaW5nLCBzZXRT
aGVldFN5bmNpbmddID0gdXNlU3RhdGUoZmFsc2UpOwogIGNvbnN0IHN5bmNBbGxUb1NoZWV0ID0gYXN5bmMgKCkgPT4gewogICAg
aWYgKGJ1aWxkaW5ncy5sZW5ndGggPT09IDApIHsgc2hvd1RvYXN0KCJObyBidWlsZGluZ3MgdG8gc3luYyIsICJlcnJvciIpOyBy
ZXR1cm47IH0KICAgIGNvbnN0IHNjID0gYnVpbGRpbmdzLmZpbHRlcihiID0+IHN1cnZleXNbYi51aWRdICYmIHN1cnZleXNbYi51
aWRdLnNhdmVkQXQpLmxlbmd0aDsKICAgIGNvbnN0IHVjID0gYnVpbGRpbmdzLmxlbmd0aCAtIHNjOwogICAgaWYgKCFjb25maXJt
KCJQdXNoICIgKyBidWlsZGluZ3MubGVuZ3RoICsgIiBidWlsZGluZ3MgdG8gU2hlZXQ6XG7igKIgIiArIHNjICsgIiBzdXJ2ZXll
ZFxu4oCiICIgKyB1YyArICIgdW5zdXJ2ZXllZCAod2lsbCBoYXZlIHBhcnRpYWwgZGF0YSlcblxuQ29udGludWU/IikpIHJldHVy
bjsKICAgIHNldFNoZWV0U3luY2luZyh0cnVlKTsKICAgIHRyeSB7CiAgICAgIGNvbnN0IHJvd3MgPSBidWlsZGluZ3MubWFwKGIg
PT4gewogICAgICAgIGNvbnN0IHNhdmVkID0gc3VydmV5c1tiLnVpZF07CiAgICAgICAgY29uc3QgcGYgPSBiLnByZWZpbGwgfHwg
e307CiAgICAgICAgY29uc3QgcyA9IHNhdmVkIHx8IHBmOwogICAgICAgIHJldHVybiB7CiAgICAgICAgICB1aWQ6IGIudWlkLAog
ICAgICAgICAgc3VydmV5X3R5cGU6IGIudHlwZSwKICAgICAgICAgIElEOiBTdHJpbmcoYi5uc2lJZCB8fCAiIiksCiAgICAgICAg
ICBvY2N1cGFuY3lfdHlwZTogcy5vY2NUeXBlIHx8ICIiLAogICAgICAgICAgYnVpbGRpbmdfdHlwZTogcy5idWlsZGluZ1R5cGUg
fHwgIiIsCiAgICAgICAgICBudW1iZXJfb2Zfc3Rvcmllczogcy5udW1TdG9yaWVzIHx8ICIiLAogICAgICAgICAgYXJlYTogcy5h
cmVhIHx8ICIiLAogICAgICAgICAgZm91bmRhdGlvbl90eXBlOiBzLmZvdW5kYXRpb25UeXBlIHx8ICIiLAogICAgICAgICAgZm91
bmRhdGlvbl9oZWlnaHQ6IHMuZmlyc3RGbG9vckhlaWdodCB8fCAiIiwKICAgICAgICAgIHllYXJfYnVpbHQ6IHMueWVhckJ1aWx0
IHx8ICIiLAogICAgICAgICAgZ3JvdW5kX2VsZXZhdGlvbjogcy5ncm91bmRFbGV2IHx8ICIiLAogICAgICAgICAgYWRkcmVzczog
cy5hZGRyZXNzIHx8ICIiLAogICAgICAgICAgbG9uZ2l0dWRlOiBTdHJpbmcoYi5sbmcpLAogICAgICAgICAgbGF0aXR1ZGU6IFN0
cmluZyhiLmxhdCksCiAgICAgICAgICBzdHJ1Y3R1cmVfdmFsdWU6IHMuc3RydWN0dXJlVmFsdWUgfHwgIiIsCiAgICAgICAgICBj
b250ZW50X3ZhbHVlOiBzLmNvbnRlbnRWYWx1ZSB8fCAiIiwKICAgICAgICAgIGJhc2VtZW50OiAocy5mb3VuZGF0aW9uVHlwZSB8
fCAiIikudG9VcHBlckNhc2UoKSA9PT0gIkIiID8gIlllcyIgOiAiTm8iLAogICAgICAgICAgbm90ZXM6IHMubm90ZXMgfHwgIiIs
CiAgICAgICAgICBzdXJ2ZXlvcjogcy5zdXJ2ZXlvciB8fCAiIiwKICAgICAgICAgIHNhdmVkQXQ6IHMuc2F2ZWRBdCB8fCAiIiwK
ICAgICAgICAgIGZsYWdnZWQ6IHMuZmxhZ2dlZCB8fCAiIiwKICAgICAgICB9OwogICAgICB9KTsKICAgICAgY29uc3QgcmVzcCA9
IGF3YWl0IHBvc3RKc29uKHsgYWN0aW9uOiAiYnVsa1NhdmUiLCByb3dzLCBsb2NhdGlvbjogY3VyTG9jUmVmLmN1cnJlbnQgfSk7
CiAgICAgIGlmIChyZXNwLmVycm9yKSB0aHJvdyBuZXcgRXJyb3IocmVzcC5lcnJvcik7CiAgICAgIGNvbnN0IHN1cnZleWVkQ291
bnQgPSBidWlsZGluZ3MuZmlsdGVyKGIgPT4gc3VydmV5c1tiLnVpZF0gJiYgc3VydmV5c1tiLnVpZF0uc2F2ZWRBdCkubGVuZ3Ro
OwogICAgICBzaG93VG9hc3QoIlNoZWV0IHN5bmNlZCDigJQgIiArIGJ1aWxkaW5ncy5sZW5ndGggKyAiIGJ1aWxkaW5ncyAoIiAr
IHN1cnZleWVkQ291bnQgKyAiIHN1cnZleWVkKSBwdXNoZWQhIik7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgc2hvd1RvYXN0
KCJTeW5jIGZhaWxlZDogIiArIGVyci5tZXNzYWdlLCAiZXJyb3IiKTsKICAgIH0gZmluYWxseSB7CiAgICAgIHNldFNoZWV0U3lu
Y2luZyhmYWxzZSk7CiAgICB9CiAgfTsKCiAgY29uc3QgW3NlYXJjaElkLCBzZXRTZWFyY2hJZF0gPSB1c2VTdGF0ZSgiIik7CiAg
Y29uc3Qgc2VhcmNoQnlJZCA9ICgpID0+IHsKICAgIGNvbnN0IHEgPSBzZWFyY2hJZC50cmltKCk7CiAgICBpZiAoIXEpIHJldHVy
bjsKICAgIGNvbnN0IGIgPSBidWlsZGluZ3MuZmluZCh4ID0+IHgubnNpSWQgPT09IHEgfHwgeC51aWQgPT09IHEgfHwgeC51aWQg
PT09ICJuZXctIiArIHEgfHwgeC51aWQgPT09ICJuc2ktIiArIHEpOwogICAgaWYgKGIpIHsKICAgICAgd2luZG93Ll9fc2VsKGIu
dWlkKTsKICAgICAgc2V0U2VhcmNoSWQoIiIpOwogICAgfSBlbHNlIHsKICAgICAgc2hvd1RvYXN0KCJObyBidWlsZGluZyBmb3Vu
ZCB3aXRoIElEIFwiIiArIHEgKyAiXCIiLCAiZXJyb3IiKTsKICAgIH0KICB9OwoKICBjb25zdCBbcHVsbGluZywgc2V0UHVsbGlu
Z10gPSB1c2VTdGF0ZShmYWxzZSk7CiAgY29uc3QgW2ZldGNoaW5nQXJlYSwgc2V0RmV0Y2hpbmdBcmVhXSA9IHVzZVN0YXRlKGZh
bHNlKTsKICBjb25zdCBhdXRvRmV0Y2hBcmVhID0gYXN5bmMgKCkgPT4gewogICAgaWYgKCFzZWxlY3RlZCkgcmV0dXJuOwogICAg
Y29uc3QgYiA9IGJ1aWxkaW5ncy5maW5kKHggPT4geC51aWQgPT09IHNlbGVjdGVkKTsKICAgIGlmICghYikgcmV0dXJuOwogICAg
c2V0RmV0Y2hpbmdBcmVhKHRydWUpOwogICAgdHJ5IHsKICAgICAgY29uc3Qgc3FmdCA9IGF3YWl0IGZldGNoQnVpbGRpbmdGb290
cHJpbnRBcmVhKGIubG5nLCBiLmxhdCk7CiAgICAgIGlmIChzcWZ0ICE9PSBudWxsKSB7CiAgICAgICAgc2V0Rm9ybShmID0+ICh7
Li4uZiwgYXJlYTogU3RyaW5nKHNxZnQpfSkpOwogICAgICAgIHNob3dUb2FzdCgiQXJlYSBmZXRjaGVkOiAiICsgc3FmdC50b0xv
Y2FsZVN0cmluZygpICsgIiBzcWZ0Iik7CiAgICAgIH0gZWxzZSB7CiAgICAgICAgc2hvd1RvYXN0KCJObyBidWlsZGluZyBmb290
cHJpbnQgZm91bmQgYXQgdGhpcyBsb2NhdGlvbiIsICJlcnJvciIpOwogICAgICB9CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAg
c2hvd1RvYXN0KCJGb290cHJpbnQgZmV0Y2ggZmFpbGVkOiAiICsgZXJyLm1lc3NhZ2UsICJlcnJvciIpOwogICAgfSBmaW5hbGx5
IHsKICAgICAgc2V0RmV0Y2hpbmdBcmVhKGZhbHNlKTsKICAgIH0KICB9OwoKICAvLyDilIDilIDilIAgRHJhdyBwb2x5Z29uIHRv
IG1lYXN1cmUgZm9vdHByaW50IGFyZWEg4pSA4pSA4pSACiAgY29uc3QgW2RyYXdpbmdNb2RlLCBzZXREcmF3aW5nTW9kZV0gPSB1
c2VTdGF0ZShmYWxzZSk7CiAgY29uc3QgW2RyYXdQb2ludENvdW50LCBzZXREcmF3UG9pbnRDb3VudF0gPSB1c2VTdGF0ZSgwKTsK
ICBjb25zdCBkcmF3UG9pbnRzUmVmID0gdXNlUmVmKFtdKTsKICBjb25zdCBkcmF3TGF5ZXJzUmVmID0gdXNlUmVmKFtdKTsKCiAg
Y29uc3QgY2xlYXJEcmF3TGF5ZXJzID0gKCkgPT4gewogICAgZHJhd0xheWVyc1JlZi5jdXJyZW50LmZvckVhY2gobCA9PiBsLnJl
bW92ZSgpKTsKICAgIGRyYXdMYXllcnNSZWYuY3VycmVudCA9IFtdOwogIH07CgogIGNvbnN0IHN0YXJ0RHJhd2luZyA9ICgpID0+
IHsKICAgIGlmIChkZXZBY3Rpb24pIHsgc2hvd1RvYXN0KCJFeGl0IGRldiBtb2RlIGFjdGlvbiBmaXJzdCIsICJlcnJvciIpOyBy
ZXR1cm47IH0KICAgIGRyYXdQb2ludHNSZWYuY3VycmVudCA9IFtdOwogICAgc2V0RHJhd1BvaW50Q291bnQoMCk7CiAgICBjbGVh
ckRyYXdMYXllcnMoKTsKICAgIHNldERyYXdpbmdNb2RlKHRydWUpOwogICAgd2luZG93Ll9fZHJhd01vZGUgPSB0cnVlOwogIH07
CgogIGNvbnN0IGNhbmNlbERyYXdpbmcgPSAoKSA9PiB7CiAgICBkcmF3UG9pbnRzUmVmLmN1cnJlbnQgPSBbXTsKICAgIHNldERy
YXdQb2ludENvdW50KDApOwogICAgY2xlYXJEcmF3TGF5ZXJzKCk7CiAgICBzZXREcmF3aW5nTW9kZShmYWxzZSk7CiAgICB3aW5k
b3cuX19kcmF3TW9kZSA9IGZhbHNlOwogIH07CgogIGNvbnN0IGZpbmlzaERyYXdpbmcgPSAoKSA9PiB7CiAgICBjb25zdCBwdHMg
PSBkcmF3UG9pbnRzUmVmLmN1cnJlbnQ7CiAgICBpZiAocHRzLmxlbmd0aCA8IDMpIHsgc2hvd1RvYXN0KCJOZWVkIGF0IGxlYXN0
IDMgcG9pbnRzIHRvIGZvcm0gYSBwb2x5Z29uIiwgImVycm9yIik7IHJldHVybjsgfQogICAgLy8gQ2FsY3VsYXRlIGFyZWEgdXNp
bmcgU2hvZWxhY2Ugd2l0aCBnZW9kZXNpYyBjb3JyZWN0aW9uCiAgICBjb25zdCBtaWRMYXQgPSBwdHMucmVkdWNlKChzLCBwKSA9
PiBzICsgcFswXSwgMCkgLyBwdHMubGVuZ3RoOwogICAgY29uc3QgZGVnTG5nMm0gPSBNYXRoLmNvcyhtaWRMYXQgKiBNYXRoLlBJ
IC8gMTgwKSAqIDExMTMyMDsKICAgIGNvbnN0IGRlZ0xhdDJtID0gMTEwNTQwOwogICAgbGV0IGFyZWEgPSAwOwogICAgZm9yIChs
ZXQgaSA9IDA7IGkgPCBwdHMubGVuZ3RoOyBpKyspIHsKICAgICAgY29uc3QgaiA9IChpICsgMSkgJSBwdHMubGVuZ3RoOwogICAg
ICBjb25zdCB4aSA9IHB0c1tpXVsxXSAqIGRlZ0xuZzJtLCB5aSA9IHB0c1tpXVswXSAqIGRlZ0xhdDJtOwogICAgICBjb25zdCB4
aiA9IHB0c1tqXVsxXSAqIGRlZ0xuZzJtLCB5aiA9IHB0c1tqXVswXSAqIGRlZ0xhdDJtOwogICAgICBhcmVhICs9IHhpICogeWog
LSB4aiAqIHlpOwogICAgfQogICAgY29uc3Qgc3FNID0gTWF0aC5hYnMoYXJlYSkgLyAyOwogICAgY29uc3Qgc3FmdCA9IE1hdGgu
cm91bmQoc3FNICogMTAuNzYzOSk7CiAgICBzZXRGb3JtKGYgPT4gKHsuLi5mLCBhcmVhOiBTdHJpbmcoc3FmdCl9KSk7CiAgICBz
aG93VG9hc3QoIkRyYXduIGFyZWE6ICIgKyBzcWZ0LnRvTG9jYWxlU3RyaW5nKCkgKyAiIHNxZnQiKTsKICAgIGNhbmNlbERyYXdp
bmcoKTsKICB9OwoKICB1c2VFZmZlY3QoKCkgPT4gewogICAgd2luZG93Ll9fZHJhd0NsaWNrID0gKGxhdCwgbG5nKSA9PiB7CiAg
ICAgIGNvbnN0IHB0cyA9IGRyYXdQb2ludHNSZWYuY3VycmVudDsKICAgICAgcHRzLnB1c2goW2xhdCwgbG5nXSk7CiAgICAgIGRy
YXdQb2ludHNSZWYuY3VycmVudCA9IHB0czsKICAgICAgc2V0RHJhd1BvaW50Q291bnQocHRzLmxlbmd0aCk7CiAgICAgIGlmICgh
bWFwSW5zdC5jdXJyZW50IHx8ICF3aW5kb3cuTCkgcmV0dXJuOwogICAgICBjb25zdCBMID0gd2luZG93Lkw7CiAgICAgIC8vIEFk
ZCB2ZXJ0ZXggbWFya2VyCiAgICAgIGNvbnN0IG1hcmtlciA9IEwuY2lyY2xlTWFya2VyKFtsYXQsIGxuZ10sIHsKICAgICAgICBy
YWRpdXM6IDUsIGZpbGxDb2xvcjogIiNmNTllMGIiLCBjb2xvcjogIiNmZmYiLCB3ZWlnaHQ6IDIsIGZpbGxPcGFjaXR5OiAxLAog
ICAgICB9KS5hZGRUbyhtYXBJbnN0LmN1cnJlbnQpOwogICAgICBkcmF3TGF5ZXJzUmVmLmN1cnJlbnQucHVzaChtYXJrZXIpOwog
ICAgICAvLyBVcGRhdGUgcG9seWdvbiBwcmV2aWV3CiAgICAgIGlmIChwdHMubGVuZ3RoID49IDIpIHsKICAgICAgICAvLyBSZW1v
dmUgb2xkIHBvbHlnb24gcHJldmlldwogICAgICAgIGRyYXdMYXllcnNSZWYuY3VycmVudCA9IGRyYXdMYXllcnNSZWYuY3VycmVu
dC5maWx0ZXIobCA9PiB7CiAgICAgICAgICBpZiAobC5fZHJhd1BvbHkpIHsgbC5yZW1vdmUoKTsgcmV0dXJuIGZhbHNlOyB9CiAg
ICAgICAgICByZXR1cm4gdHJ1ZTsKICAgICAgICB9KTsKICAgICAgICBjb25zdCBwb2x5ID0gTC5wb2x5Z29uKHB0cywgewogICAg
ICAgICAgY29sb3I6ICIjZjU5ZTBiIiwgd2VpZ2h0OiAyLCBmaWxsQ29sb3I6ICIjZjU5ZTBiIiwgZmlsbE9wYWNpdHk6IDAuMTUs
IGRhc2hBcnJheTogIjYgNCIsCiAgICAgICAgfSkuYWRkVG8obWFwSW5zdC5jdXJyZW50KTsKICAgICAgICBwb2x5Ll9kcmF3UG9s
eSA9IHRydWU7CiAgICAgICAgZHJhd0xheWVyc1JlZi5jdXJyZW50LnB1c2gocG9seSk7CiAgICAgIH0KICAgIH07CiAgICByZXR1
cm4gKCkgPT4geyBkZWxldGUgd2luZG93Ll9fZHJhd0NsaWNrOyB9OwogIH0sIFtdKTsKCiAgdXNlRWZmZWN0KCgpID0+IHsgd2lu
ZG93Ll9fZHJhd01vZGUgPSBkcmF3aW5nTW9kZTsgfSwgW2RyYXdpbmdNb2RlXSk7CgogIGNvbnN0IGF1dG9Fc3RpbWF0ZUNvc3Qg
PSAoKSA9PiB7CiAgICBpZiAoIWZvcm0uYXJlYSB8fCAhZm9ybS5udW1TdG9yaWVzKSB7CiAgICAgIHNob3dUb2FzdCgiTmVlZCBm
b290cHJpbnQgYXJlYSBhbmQgc3RvcmllcyBmaXJzdCIsICJlcnJvciIpOyByZXR1cm47CiAgICB9CiAgICBjb25zdCByZXN1bHQg
PSBlc3RpbWF0ZUNvc3RSZWdyZXNzaW9uKGZvcm0uYXJlYSwgZm9ybS5udW1TdG9yaWVzLCBzdXJ2ZXlzLCBidWlsZGluZ3MsIGZv
cm0ub2NjVHlwZSk7CiAgICBpZiAocmVzdWx0KSB7CiAgICAgIHNldEZvcm0oZiA9PiAoey4uLmYsIHN0cnVjdHVyZVZhbHVlOiBT
dHJpbmcocmVzdWx0LnN0cnVjdHVyZSksIGNvbnRlbnRWYWx1ZTogU3RyaW5nKHJlc3VsdC5jb250ZW50KX0pKTsKICAgICAgY29u
c3Qgc2NvcGUgPSByZXN1bHQuZmlsdGVyZWQgPyByZXN1bHQub2NjUHJlZml4ICsgIiBidWlsZGluZ3MiIDogImFsbCBidWlsZGlu
Z3MiOwogICAgICBzaG93VG9hc3QocmVzdWx0Lm1ldGhvZCA9PT0gIk9MUyIKICAgICAgICA/ICJPTFMgZnJvbSAiICsgcmVzdWx0
LnJlZkNvdW50ICsgIiAiICsgc2NvcGUgKyAiOiAkIiArIHJlc3VsdC5zdHJ1Y3R1cmUudG9Mb2NhbGVTdHJpbmcoKQogICAgICAg
IDogIk1lZGlhbiBmcm9tICIgKyByZXN1bHQucmVmQ291bnQgKyAiICIgKyBzY29wZSArICI6ICQiICsgcmVzdWx0LnN0cnVjdHVy
ZS50b0xvY2FsZVN0cmluZygpCiAgICAgICk7CiAgICB9IGVsc2UgewogICAgICBzaG93VG9hc3QoIk5vdCBlbm91Z2ggcmVmZXJl
bmNlIGRhdGEg4oCUIG5lZWQgYXQgbGVhc3QgMiBidWlsZGluZ3Mgd2l0aCBrbm93biBhcmVhIGFuZCB2YWx1ZSBpbiB0aGlzIGxv
Y2F0aW9uIiwgImVycm9yIik7CiAgICB9CiAgfTsKCiAgY29uc3QgW2ZldGNoaW5nRWxldiwgc2V0RmV0Y2hpbmdFbGV2XSA9IHVz
ZVN0YXRlKGZhbHNlKTsKICBjb25zdCBhdXRvRmV0Y2hFbGV2YXRpb24gPSBhc3luYyAoKSA9PiB7CiAgICBpZiAoIXNlbGVjdGVk
KSByZXR1cm47CiAgICBjb25zdCBiID0gYnVpbGRpbmdzLmZpbmQoeCA9PiB4LnVpZCA9PT0gc2VsZWN0ZWQpOwogICAgaWYgKCFi
KSByZXR1cm47CiAgICBzZXRGZXRjaGluZ0VsZXYodHJ1ZSk7CiAgICB0cnkgewogICAgICBjb25zdCBlbGV2ID0gYXdhaXQgZmV0
Y2hVU0dTRWxldmF0aW9uKGIubG5nLCBiLmxhdCk7CiAgICAgIGlmIChlbGV2ICE9PSBudWxsKSB7CiAgICAgICAgc2V0Rm9ybShm
ID0+ICh7Li4uZiwgZ3JvdW5kRWxldjogU3RyaW5nKGVsZXYpfSkpOwogICAgICAgIHNob3dUb2FzdCgiRWxldmF0aW9uOiAiICsg
ZWxldiArICIgZnQgKE5BVkQ4OCkgZnJvbSBVU0dTIDNERVAiKTsKICAgICAgfSBlbHNlIHsKICAgICAgICBzaG93VG9hc3QoIk5v
IGVsZXZhdGlvbiBkYXRhIGF0IHRoaXMgbG9jYXRpb24iLCAiZXJyb3IiKTsKICAgICAgfQogICAgfSBjYXRjaCAoZXJyKSB7CiAg
ICAgIHNob3dUb2FzdCgiRWxldmF0aW9uIGZldGNoIGZhaWxlZDogIiArIGVyci5tZXNzYWdlLCAiZXJyb3IiKTsKICAgIH0gZmlu
YWxseSB7CiAgICAgIHNldEZldGNoaW5nRWxldihmYWxzZSk7CiAgICB9CiAgfTsKCiAgY29uc3QgcHVsbEZyb21TaGVldCA9IGFz
eW5jICgpID0+IHsKICAgIGlmICghY29uZmlybSgi4pqg77iPIFRoaXMgd2lsbCBvdmVyd3JpdGUgYWxsIGxvY2FsIGRhdGEgZm9y
IFwiIiArIExPQ0FUSU9OU1tjdXJMb2NSZWYuY3VycmVudF0ubmFtZSArICJcIiB3aXRoIHdoYXQncyBpbiB0aGUgR29vZ2xlIFNo
ZWV0LlxuXG5BbnkgbG9jYWwgcG9pbnRzIG9yIGNoYW5nZXMgbm90IGluIHRoZSBTaGVldCB3aWxsIGJlIGxvc3QuXG5cbkNvbnRp
bnVlPyIpKSByZXR1cm47CiAgICBzZXRQdWxsaW5nKHRydWUpOwogICAgc2V0U2VsZWN0ZWQobnVsbCk7CiAgICBzZXRGb3JtKEVN
UFRZX0ZPUk0pOwogICAgdHJ5IHsKICAgICAgLy8gUHVsbCBzdXJ2ZXlzIOKAlCB0aGlzIGlzIHRoZSBzaW5nbGUgc291cmNlIG9m
IHRydXRoCiAgICAgIGNvbnN0IHJlc3VsdCA9IGF3YWl0IGZldGNoU3VydmV5cyhjdXJMb2NSZWYuY3VycmVudCk7CiAgICAgIGlm
IChyZXN1bHQuZXJyb3IpIHRocm93IG5ldyBFcnJvcihyZXN1bHQuZXJyb3IpOwogICAgICBjb25zdCBzaGVldERhdGEgPSByZXN1
bHQuZGF0YSB8fCB7fTsKCiAgICAgIC8vIEJ1aWxkIGJ1aWxkaW5ncyBsaXN0IGVudGlyZWx5IGZyb20gc2hlZXQgcm93cwogICAg
ICBjb25zdCBzaGVldEJ1aWxkaW5ncyA9IHNoZWV0RGF0YVRvQnVpbGRpbmdzKHNoZWV0RGF0YSk7CgogICAgICAvLyBDbGVhciBk
ZXYgZWRpdHMg4oCUIHNoZWV0IGlzIG5vdyB0aGUgdHJ1dGgKICAgICAgY29uc3QgZnJlc2hEZXZFZGl0cyA9IHsgcmVtb3ZlZDog
W10sIG1vdmVkOiB7fSwgYWRkZWQ6IFtdLCBfdHM6IERhdGUubm93KCkgfTsKICAgICAgZGV2RWRpdHNSZWYuY3VycmVudCA9IGZy
ZXNoRGV2RWRpdHM7CiAgICAgIHNldERldkVkaXRzKGZyZXNoRGV2RWRpdHMpOwogICAgICBzYXZlTG9jYWxEZXYoZnJlc2hEZXZF
ZGl0cyk7CiAgICAgIGF3YWl0IHNhdmVEZXZFZGl0c1JlbW90ZShmcmVzaERldkVkaXRzLCBjdXJMb2NSZWYuY3VycmVudCk7Cgog
ICAgICBiYXNlQnVpbGRpbmdzUmVmLmN1cnJlbnQgPSBzaGVldEJ1aWxkaW5nczsKICAgICAgc2V0QnVpbGRpbmdzKHNoZWV0QnVp
bGRpbmdzKTsKICAgICAgc2V0U3VydmV5cyhzaGVldERhdGEpOwoKICAgICAgc2hvd1RvYXN0KCJQdWxsZWQgIiArIHNoZWV0QnVp
bGRpbmdzLmxlbmd0aCArICIgYnVpbGRpbmdzIGZyb20gU2hlZXQhIik7CiAgICB9IGNhdGNoIChlcnIpIHsKICAgICAgc2hvd1Rv
YXN0KCJQdWxsIGZhaWxlZDogIiArIGVyci5tZXNzYWdlLCAiZXJyb3IiKTsKICAgIH0gZmluYWxseSB7CiAgICAgIHNldFB1bGxp
bmcoZmFsc2UpOwogICAgfQogIH07CgogIGNvbnN0IHZlcmlmeUNvdW50ID0gYnVpbGRpbmdzLmZpbHRlcihiID0+IGIudHlwZT09
PSJ2ZXJpZnkiKS5sZW5ndGg7CiAgY29uc3Qgc3VydmV5Q291bnQgPSBidWlsZGluZ3MuZmlsdGVyKGIgPT4gYi50eXBlPT09InN1
cnZleSIpLmxlbmd0aDsKICBjb25zdCBkb25lQ291bnQgPSBidWlsZGluZ3MuZmlsdGVyKGIgPT4gc3VydmV5c1tiLnVpZF0gJiYg
c3VydmV5c1tiLnVpZF0uc2F2ZWRBdCkubGVuZ3RoOwogIGNvbnN0IHBjdCA9IGJ1aWxkaW5ncy5sZW5ndGggPyBNYXRoLnJvdW5k
KChkb25lQ291bnQgLyBidWlsZGluZ3MubGVuZ3RoKSAqIDEwMCkgOiAwOwogIGNvbnN0IHRvZG9Db3VudCA9IGJ1aWxkaW5ncy5m
aWx0ZXIoYiA9PiAhKHN1cnZleXNbYi51aWRdICYmIHN1cnZleXNbYi51aWRdLnNhdmVkQXQpKS5sZW5ndGg7CiAgY29uc3QgZmxh
Z2dlZENvdW50ID0gYnVpbGRpbmdzLmZpbHRlcihiID0+IHsgY29uc3Qgc3YgPSBzdXJ2ZXlzW2IudWlkXTsgY29uc3QgcGYgPSBi
LnByZWZpbGwgfHwge307IHJldHVybiAoc3YgfHwgcGYpLmZsYWdnZWQgPT09ICJZZXMiOyB9KS5sZW5ndGg7CiAgY29uc3QgZGVt
b2xpc2hlZENvdW50ID0gYnVpbGRpbmdzLmZpbHRlcihiID0+IHsgY29uc3Qgc3YgPSBzdXJ2ZXlzW2IudWlkXTsgY29uc3QgcGYg
PSBiLnByZWZpbGwgfHwge307IHJldHVybiAoc3YgfHwgcGYpLmZsYWdnZWQgPT09ICJEZW1vbGlzaGVkIjsgfSkubGVuZ3RoOwog
IGNvbnN0IHNlbEIgPSBzZWxlY3RlZCA/IGJ1aWxkaW5ncy5maW5kKHggPT4geC51aWQgPT09IHNlbGVjdGVkKSA6IG51bGw7CiAg
Ly8gRmlsdGVyLWF3YXJlIFVJRHMgZm9yIFByZXYvTmV4dCBuYXZpZ2F0aW9uCiAgY29uc3QgZmlsdGVyZWRVaWRzID0gYnVpbGRp
bmdzLmZpbHRlcihiID0+IHBhc3Nlc0ZpbHRlcnMoYikpLm1hcChiID0+IGIudWlkKTsKICBjb25zdCBkZXZFZGl0Q291bnQgPSBk
ZXZFZGl0cy5yZW1vdmVkLmxlbmd0aCArIE9iamVjdC5rZXlzKGRldkVkaXRzLm1vdmVkKS5sZW5ndGggKyBkZXZFZGl0cy5hZGRl
ZC5sZW5ndGg7CgogIHJldHVybiAoCiAgICA8ZGl2IHN0eWxlPXt7ZGlzcGxheToiZmxleCIsZmxleERpcmVjdGlvbjoiY29sdW1u
IixoZWlnaHQ6IjEwMCUiLGZvbnRGYW1pbHk6IidTZWdvZSBVSScsc3lzdGVtLXVpLHNhbnMtc2VyaWYiLGJhY2tncm91bmQ6IiMw
ZjE3MmEiLGNvbG9yOiIjZTJlOGYwIn19PgogICAgICB7dG9hc3QgJiYgKAogICAgICAgIDxkaXYgc3R5bGU9e3twb3NpdGlvbjoi
Zml4ZWQiLHRvcDoxNixsZWZ0OiI1MCUiLHRyYW5zZm9ybToidHJhbnNsYXRlWCgtNTAlKSIsekluZGV4OjEwMDAwLAogICAgICAg
ICAgcGFkZGluZzoiMTBweCAyNHB4Iixib3JkZXJSYWRpdXM6OCwKICAgICAgICAgIGJhY2tncm91bmQ6dG9hc3QudHlwZT09PSJl
cnJvciI/IiNkYzI2MjYiOnRvYXN0LnR5cGU9PT0iaW5mbyI/IiMyNTYzZWIiOiIjMTZhMzRhIiwKICAgICAgICAgIGNvbG9yOiIj
ZmZmIixmb250V2VpZ2h0OjYwMCxmb250U2l6ZToxNCxib3hTaGFkb3c6IjAgNHB4IDIwcHggcmdiYSgwLDAsMCwuNCkifX0+e3Rv
YXN0Lm1zZ308L2Rpdj4KICAgICAgKX0KCiAgICAgIHsvKiDilZDilZDilZAgVE9QIEJBUjogY29tcGFjdCBzaW5nbGUtcm93IHRv
b2xiYXIgKHdhcyBsZWZ0IHBhbmVsKSDilZDilZDilZAgKi99CiAgICAgIDxkaXYgc3R5bGU9e3t3aWR0aDoiMTAwJSIsYm9yZGVy
Qm90dG9tOiIxcHggc29saWQgIzFlMjkzYiIsYmFja2dyb3VuZDoibGluZWFyLWdyYWRpZW50KDE4MGRlZywjMGYxNzJhLCMxMTE4
MjcpIixmbGV4U2hyaW5rOjAscG9zaXRpb246InJlbGF0aXZlIix6SW5kZXg6MjAwMH19PgogICAgICAgIDxkaXYgc3R5bGU9e3tw
YWRkaW5nOiI3cHggMTJweCIsZGlzcGxheToiZmxleCIsZmxleERpcmVjdGlvbjoicm93IixmbGV4V3JhcDoid3JhcCIsYWxpZ25J
dGVtczoiY2VudGVyIixnYXA6IjhweCA5cHgifX0+CgogICAgICAgICAgey8qIExvY2F0aW9uIChyZWFkLW9ubHkg4oCUIGRyaXZl
biBieSB0aGUgQURBUFQgZ2xvYmFsIHJhaWwpICovfQogICAgICAgICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGFsaWdu
SXRlbXM6ImNlbnRlciIsZ2FwOjYsZmxleDoiMCAwIGF1dG8iLHBhZGRpbmc6IjVweCAxMHB4Iixib3JkZXJSYWRpdXM6NyxiYWNr
Z3JvdW5kOiIjMWUyOTNiIixib3JkZXI6IjFweCBzb2xpZCAjMzM0MTU1Iix3aGl0ZVNwYWNlOiJub3dyYXAifX0+CiAgICAgICAg
ICAgIDxzcGFuIHN0eWxlPXt7Zm9udFNpemU6MTN9fT7wn5ONPC9zcGFuPgogICAgICAgICAgICA8c3BhbiBzdHlsZT17e2ZvbnRT
aXplOjEyLGZvbnRXZWlnaHQ6NjAwLGNvbG9yOiIjZTJlOGYwIn19PntMT0NBVElPTlNbY3VyTG9jXS5uYW1lfTwvc3Bhbj4KICAg
ICAgICAgICAgPHNwYW4gc3R5bGU9e3tmb250U2l6ZToxMSxjb2xvcjoiIzY0NzQ4YiJ9fT7CtyB7YnVpbGRpbmdzLmxlbmd0aH08
L3NwYW4+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8ZGl2IHN0eWxlPXt0YkRpdmlkZXJ9Lz4KCiAgICAgICAgICB7Lyog
UHJvZ3Jlc3MgKyBzdGF0dXMgbGVnZW5kIChpbmxpbmUpICovfQogICAgICAgICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgi
LGFsaWduSXRlbXM6ImNlbnRlciIsZ2FwOjksZmxleDoiMCAwIGF1dG8iLHdoaXRlU3BhY2U6Im5vd3JhcCJ9fT4KICAgICAgICAg
ICAgPGRpdiBzdHlsZT17e3dpZHRoOjg4LGhlaWdodDo2LGJvcmRlclJhZGl1czozLGJhY2tncm91bmQ6IiMxZTI5M2IiLG92ZXJm
bG93OiJoaWRkZW4ifX0+CiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2hlaWdodDo2LGJvcmRlclJhZGl1czozLGJhY2tncm91
bmQ6cGN0PT09MTAwPyIjMjJjNTVlIjoibGluZWFyLWdyYWRpZW50KDkwZGVnLCMzYjgyZjYsIzhiNWNmNikiLHdpZHRoOmAke3Bj
dH0lYCx0cmFuc2l0aW9uOiJ3aWR0aCAuNHMifX0vPgogICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPHNwYW4gc3R5bGU9
e3tmb250U2l6ZToxMSxjb2xvcjoiI2NiZDVlMSJ9fT57ZG9uZUNvdW50fS97YnVpbGRpbmdzLmxlbmd0aH0gPGIgc3R5bGU9e3tj
b2xvcjpwY3Q9PT0xMDA/IiMyMmM1NWUiOiIjZjU5ZTBiIn19PntwY3R9JTwvYj48L3NwYW4+CiAgICAgICAgICAgIDxzcGFuIHN0
eWxlPXt7ZGlzcGxheToiZmxleCIsZ2FwOjgsZm9udFNpemU6MTEsY29sb3I6IiM5NGEzYjgifX0+CiAgICAgICAgICAgICAgPHNw
YW4gdGl0bGU9Ik5ldyAvIHVuc3VydmV5ZWQiPjxzcGFuIHN0eWxlPXt7ZGlzcGxheToiaW5saW5lLWJsb2NrIix3aWR0aDo4LGhl
aWdodDo4LGJvcmRlclJhZGl1czoiNTAlIixiYWNrZ3JvdW5kOiIjZWY0NDQ0IixtYXJnaW5SaWdodDozLHZlcnRpY2FsQWxpZ246
Im1pZGRsZSJ9fS8+e3N1cnZleUNvdW50fTwvc3Bhbj4KICAgICAgICAgICAgICA8c3BhbiB0aXRsZT0iVmVyaWZ5Ij48c3BhbiBz
dHlsZT17e2Rpc3BsYXk6ImlubGluZS1ibG9jayIsd2lkdGg6OCxoZWlnaHQ6OCxib3JkZXJSYWRpdXM6IjUwJSIsYmFja2dyb3Vu
ZDoiIzNiODJmNiIsbWFyZ2luUmlnaHQ6Myx2ZXJ0aWNhbEFsaWduOiJtaWRkbGUifX0vPnt2ZXJpZnlDb3VudH08L3NwYW4+CiAg
ICAgICAgICAgICAgPHNwYW4gdGl0bGU9IkRvbmUiPjxzcGFuIHN0eWxlPXt7ZGlzcGxheToiaW5saW5lLWJsb2NrIix3aWR0aDo4
LGhlaWdodDo4LGJvcmRlclJhZGl1czoiNTAlIixiYWNrZ3JvdW5kOiIjMjJjNTVlIixtYXJnaW5SaWdodDozLHZlcnRpY2FsQWxp
Z246Im1pZGRsZSJ9fS8+e2RvbmVDb3VudH08L3NwYW4+CiAgICAgICAgICAgICAge2ZsYWdnZWRDb3VudCA+IDAgJiYgPHNwYW4g
dGl0bGU9IkZsYWdnZWQiIHN0eWxlPXt7Y29sb3I6IiNmOTczMTYifX0+8J+aqXtmbGFnZ2VkQ291bnR9PC9zcGFuPn0KICAgICAg
ICAgICAgICB7ZGVtb2xpc2hlZENvdW50ID4gMCAmJiA8c3BhbiB0aXRsZT0iRGVtb2xpc2hlZCIgc3R5bGU9e3tjb2xvcjoiIzk0
YTNiOCJ9fT7inJV7ZGVtb2xpc2hlZENvdW50fTwvc3Bhbj59CiAgICAgICAgICAgIDwvc3Bhbj4KICAgICAgICAgIDwvZGl2PgoK
ICAgICAgICAgIDxkaXYgc3R5bGU9e3RiRGl2aWRlcn0vPgoKICAgICAgICAgIHsvKiBBZHZhbmNlZCBmaWx0ZXJzIHBvcG92ZXIg
Ki99CiAgICAgICAgICA8UG9wb3ZlciBsYWJlbD0i8J+UjiBGaWx0ZXJzIiBwYW5lbFdpZHRoPXsyNjR9IGFjY2VudD0iIzNiODJm
NiI+CiAgICAgICAgICAgIDxkaXYgc3R5bGU9e3ttYXJnaW5Cb3R0b206OX19PgogICAgICAgICAgICAgIDxsYWJlbCBzdHlsZT17
dGJMYmx9PlN1cnZleSBzdGF0dXM8L2xhYmVsPgogICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3tkaXNwbGF5OiJmbGV4IixnYXA6
M319PgogICAgICAgICAgICAgICAge1tbImFsbCIsIkFsbCIsYnVpbGRpbmdzLmxlbmd0aF0sWyJzdXJ2ZXkiLCJOZXciLHN1cnZl
eUNvdW50XSxbImRvbmUiLCJEb25lIixkb25lQ291bnRdLFsicGVuZGluZyIsIlRvZG8iLHRvZG9Db3VudF1dLm1hcCgoW2ssbCxu
XSk9PigKICAgICAgICAgICAgICAgICAgPGJ1dHRvbiBrZXk9e2t9IG9uQ2xpY2s9eygpPT5zZXRGaWx0ZXIoayl9IHN0eWxlPXt7
CiAgICAgICAgICAgICAgICAgICAgZmxleDoxLHBhZGRpbmc6IjVweCA0cHgiLGJvcmRlclJhZGl1czo2LGJvcmRlcjoibm9uZSIs
Y3Vyc29yOiJwb2ludGVyIixmb250U2l6ZToxMSxmb250V2VpZ2h0OjYwMCx3aGl0ZVNwYWNlOiJub3dyYXAiLAogICAgICAgICAg
ICAgICAgICAgIGJhY2tncm91bmQ6ZmlsdGVyPT09az8iIzNiODJmNiI6IiMxZTI5M2IiLGNvbG9yOmZpbHRlcj09PWs/IiNmZmYi
OiIjOTRhM2I4IiwKICAgICAgICAgICAgICAgICAgfX0+e2x9IDxzcGFuIHN0eWxlPXt7b3BhY2l0eTouNjV9fT57bn08L3NwYW4+
PC9idXR0b24+CiAgICAgICAgICAgICAgICApKX0KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPC9kaXY+CiAgICAg
ICAgICAgIDxkaXYgc3R5bGU9e3tkaXNwbGF5OiJncmlkIixncmlkVGVtcGxhdGVDb2x1bW5zOiIxZnIgMWZyIixnYXA6OH19Pgog
ICAgICAgICAgICAgIDxkaXY+CiAgICAgICAgICAgICAgICA8bGFiZWwgc3R5bGU9e3RiTGJsfT5TdGF0dXM8L2xhYmVsPgogICAg
ICAgICAgICAgICAgPHNlbGVjdCB2YWx1ZT17YWR2RmlsdGVycy5mbGFnZ2VkfSBvbkNoYW5nZT17ZT0+c2V0QWR2RmlsdGVycyhm
PT4oey4uLmYsZmxhZ2dlZDplLnRhcmdldC52YWx1ZX0pKX0gc3R5bGU9e3RiU2VsfT4KICAgICAgICAgICAgICAgICAgPG9wdGlv
biB2YWx1ZT0iYWxsIj5BbGw8L29wdGlvbj48b3B0aW9uIHZhbHVlPSJ5ZXMiPvCfmqkgRmxhZ2dlZDwvb3B0aW9uPjxvcHRpb24g
dmFsdWU9ImRlbW9saXNoZWQiPuKclSBEZW1vbGlzaGVkPC9vcHRpb24+PG9wdGlvbiB2YWx1ZT0ibm8iPlVuZmxhZ2dlZDwvb3B0
aW9uPgogICAgICAgICAgICAgICAgPC9zZWxlY3Q+CiAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgICAgPGRpdj4KICAg
ICAgICAgICAgICAgIDxsYWJlbCBzdHlsZT17dGJMYmx9Pk9jY3VwYW5jeTwvbGFiZWw+CiAgICAgICAgICAgICAgICA8c2VsZWN0
IHZhbHVlPXthZHZGaWx0ZXJzLm9jY0NsYXNzfSBvbkNoYW5nZT17ZT0+c2V0QWR2RmlsdGVycyhmPT4oey4uLmYsb2NjQ2xhc3M6
ZS50YXJnZXQudmFsdWV9KSl9IHN0eWxlPXt0YlNlbH0+CiAgICAgICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9ImFsbCI+QWxs
PC9vcHRpb24+PG9wdGlvbiB2YWx1ZT0iUkVTIj5SZXNpZGVudGlhbDwvb3B0aW9uPjxvcHRpb24gdmFsdWU9IkNPTSI+Q29tbWVy
Y2lhbDwvb3B0aW9uPjxvcHRpb24gdmFsdWU9IklORCI+SW5kdXN0cmlhbDwvb3B0aW9uPjxvcHRpb24gdmFsdWU9Ik9USEVSIj5P
dGhlcjwvb3B0aW9uPgogICAgICAgICAgICAgICAgPC9zZWxlY3Q+CiAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgICAg
PGRpdj4KICAgICAgICAgICAgICAgIDxsYWJlbCBzdHlsZT17dGJMYmx9PkZvdW5kYXRpb248L2xhYmVsPgogICAgICAgICAgICAg
ICAgPHNlbGVjdCB2YWx1ZT17YWR2RmlsdGVycy5mb3VuZGF0aW9ufSBvbkNoYW5nZT17ZT0+c2V0QWR2RmlsdGVycyhmPT4oey4u
LmYsZm91bmRhdGlvbjplLnRhcmdldC52YWx1ZX0pKX0gc3R5bGU9e3RiU2VsfT4KICAgICAgICAgICAgICAgICAgPG9wdGlvbiB2
YWx1ZT0iYWxsIj5BbGw8L29wdGlvbj48b3B0aW9uIHZhbHVlPSJTIj5TbGFiPC9vcHRpb24+PG9wdGlvbiB2YWx1ZT0iQyI+Q3Jh
d2w8L29wdGlvbj48b3B0aW9uIHZhbHVlPSJCIj5CYXNlbWVudDwvb3B0aW9uPjxvcHRpb24gdmFsdWU9IlAiPlBpZXI8L29wdGlv
bj48b3B0aW9uIHZhbHVlPSJXIj5XYWxsPC9vcHRpb24+CiAgICAgICAgICAgICAgICA8L3NlbGVjdD4KICAgICAgICAgICAgICA8
L2Rpdj4KICAgICAgICAgICAgICA8ZGl2PgogICAgICAgICAgICAgICAgPGxhYmVsIHN0eWxlPXt0YkxibH0+QmxkZyBUeXBlPC9s
YWJlbD4KICAgICAgICAgICAgICAgIDxzZWxlY3QgdmFsdWU9e2FkdkZpbHRlcnMuYmxkZ1R5cGV9IG9uQ2hhbmdlPXtlPT5zZXRB
ZHZGaWx0ZXJzKGY9Pih7Li4uZixibGRnVHlwZTplLnRhcmdldC52YWx1ZX0pKX0gc3R5bGU9e3RiU2VsfT4KICAgICAgICAgICAg
ICAgICAgPG9wdGlvbiB2YWx1ZT0iYWxsIj5BbGw8L29wdGlvbj48b3B0aW9uIHZhbHVlPSJXIj5Xb29kPC9vcHRpb24+PG9wdGlv
biB2YWx1ZT0iTSI+TWFzb25yeTwvb3B0aW9uPjxvcHRpb24gdmFsdWU9IkMiPkNvbmNyZXRlPC9vcHRpb24+PG9wdGlvbiB2YWx1
ZT0iUyI+U3RlZWw8L29wdGlvbj48b3B0aW9uIHZhbHVlPSJIIj5NYW51Zi48L29wdGlvbj4KICAgICAgICAgICAgICAgIDwvc2Vs
ZWN0PgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPGRpdiBzdHlsZT17e2Rpc3Bs
YXk6ImZsZXgiLGp1c3RpZnlDb250ZW50OiJzcGFjZS1iZXR3ZWVuIixhbGlnbkl0ZW1zOiJjZW50ZXIiLG1hcmdpblRvcDo5fX0+
CiAgICAgICAgICAgICAgPHNwYW4gc3R5bGU9e3tmb250U2l6ZToxMCxjb2xvcjoiIzk0YTNiOCJ9fT57ZmlsdGVyZWRVaWRzLmxl
bmd0aH0gLyB7YnVpbGRpbmdzLmxlbmd0aH0gc2hvd248L3NwYW4+CiAgICAgICAgICAgICAge09iamVjdC52YWx1ZXMoYWR2Rmls
dGVycykuc29tZSh2PT52IT09ImFsbCIpICYmICgKICAgICAgICAgICAgICAgIDxidXR0b24gb25DbGljaz17KCk9PnNldEFkdkZp
bHRlcnMoe2ZsYWdnZWQ6ImFsbCIsb2NjQ2xhc3M6ImFsbCIsZm91bmRhdGlvbjoiYWxsIixibGRnVHlwZToiYWxsIn0pfSBzdHls
ZT17e3BhZGRpbmc6IjNweCAxMHB4Iixib3JkZXJSYWRpdXM6NCxib3JkZXI6Im5vbmUiLGN1cnNvcjoicG9pbnRlciIsYmFja2dy
b3VuZDoiIzMzNDE1NSIsY29sb3I6IiNmOTczMTYiLGZvbnRTaXplOjEwLGZvbnRXZWlnaHQ6NjAwfX0+Q2xlYXI8L2J1dHRvbj4K
ICAgICAgICAgICAgICApfQogICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgIDwvUG9wb3Zlcj4KCiAgICAgICAgICA8ZGl2IHN0
eWxlPXt0YkRpdmlkZXJ9Lz4KCiAgICAgICAgICB7LyogQWN0aW9ucyAoaWNvbiArIGxhYmVsKSAqL30KICAgICAgICAgIDxkaXYg
c3R5bGU9e3tkaXNwbGF5OiJmbGV4IixmbGV4V3JhcDoid3JhcCIsZ2FwOjUsZmxleDoiMCAwIGF1dG8ifX0+CiAgICAgICAgICAg
IDxidXR0b24gdGl0bGU9IkdvIHRvIG5lYXJlc3QgdW5zdXJ2ZXllZCIgb25DbGljaz17Z29Ub05lYXJlc3R9IHN0eWxlPXt0YkJ0
bkwoIiMwZjc2NmUiKX0+8J+TjSBOZWFyZXN0PC9idXR0b24+CiAgICAgICAgICAgIDxidXR0b24gdGl0bGU9IkV4cG9ydCBzdXJ2
ZXkgZGF0YSB0byBYTFNYIiBvbkNsaWNrPXtleHBvcnRYTFNYfSBzdHlsZT17dGJCdG5MKCIjN2MzYWVkIil9PvCfk6UgRXhwb3J0
PC9idXR0b24+CiAgICAgICAgICAgIDxidXR0b24gdGl0bGU9Ik9wZW4gdGhlIEdvb2dsZSBTaGVldCIgb25DbGljaz17KCk9Pndp
bmRvdy5vcGVuKFNIRUVUX1VSTCwiX2JsYW5rIil9IHN0eWxlPXt0YkJ0bkwoIiMwMzY5YTEiKX0+8J+TiiBTaGVldDwvYnV0dG9u
PgogICAgICAgICAgICA8YnV0dG9uIHRpdGxlPSJPcGVuIHRoZSBkb2NzIiBvbkNsaWNrPXsoKT0+d2luZG93Lm9wZW4oUkVBRE1F
X1VSTCwiX2JsYW5rIil9IHN0eWxlPXt0YkJ0bkwoIiM0NzU1NjkiKX0+8J+TliBEb2NzPC9idXR0b24+CiAgICAgICAgICAgIDxi
dXR0b24gdGl0bGU9IlB1bGwgbGF0ZXN0IGZyb20gdGhlIFNoZWV0IiBvbkNsaWNrPXtwdWxsRnJvbVNoZWV0fSBkaXNhYmxlZD17
cHVsbGluZ30gc3R5bGU9e3RiQnRuTCgiIzFkNGVkOCIpfT57cHVsbGluZyA/ICLij7MgUHVsbCIgOiAi4qyHIFB1bGwifTwvYnV0
dG9uPgogICAgICAgICAgICA8YnV0dG9uIHRpdGxlPSJTeW5jIGFsbCByb3dzIHRvIHRoZSBTaGVldCIgb25DbGljaz17c3luY0Fs
bFRvU2hlZXR9IGRpc2FibGVkPXtzaGVldFN5bmNpbmd9IHN0eWxlPXt0YkJ0bkwoIiNiNDUzMDkiKX0+e3NoZWV0U3luY2luZyA/
ICLij7MgU3luYyIgOiAi4piB77iPIFN5bmMifTwvYnV0dG9uPgogICAgICAgICAgPC9kaXY+CgogICAgICAgICAgey8qIERldmVs
b3BlciBwb2ludC1tYW5hZ2VtZW50IHBvcG92ZXIgKi99CiAgICAgICAgICA8UG9wb3ZlciBsYWJlbD0i8J+boCBEZXYiIHBhbmVs
V2lkdGg9ezI1Mn0gYWxpZ249InJpZ2h0IiBhY2NlbnQ9IiNmNTllMGIiPgogICAgICAgICAgICA8ZGl2IHN0eWxlPXt7ZGlzcGxh
eToiZmxleCIsZmxleERpcmVjdGlvbjoiY29sdW1uIixnYXA6Nn19PgogICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3tkaXNwbGF5
OiJmbGV4IixnYXA6NX19PgogICAgICAgICAgICAgICAgPGJ1dHRvbiBvbkNsaWNrPXsoKT0+c2V0RGV2QWN0aW9uKGRldkFjdGlv
bj09PSJhZGQiP251bGw6ImFkZCIpfSBzdHlsZT17ewogICAgICAgICAgICAgICAgICBmbGV4OjEscGFkZGluZzoiNnB4Iixib3Jk
ZXJSYWRpdXM6Nixib3JkZXI6ZGV2QWN0aW9uPT09ImFkZCI/IjJweCBzb2xpZCAjMjJjNTVlIjoiMXB4IHNvbGlkICMzMzQxNTUi
LAogICAgICAgICAgICAgICAgICBjdXJzb3I6InBvaW50ZXIiLGJhY2tncm91bmQ6ZGV2QWN0aW9uPT09ImFkZCI/IiMxNDUzMmQi
OiIjMWUyOTNiIiwKICAgICAgICAgICAgICAgICAgY29sb3I6ZGV2QWN0aW9uPT09ImFkZCI/IiM0YWRlODAiOiIjOTRhM2I4Iixm
b250V2VpZ2h0OjYwMCxmb250U2l6ZToxMSwKICAgICAgICAgICAgICAgIH19PntkZXZBY3Rpb249PT0iYWRkIj8i8J+foiBDbGlj
ayBNYXAuLi4iOiLinpUgQWRkIn08L2J1dHRvbj4KICAgICAgICAgICAgICAgIDxidXR0b24gb25DbGljaz17KCk9PnsKICAgICAg
ICAgICAgICAgICAgaWYgKCFzZWxlY3RlZCkgeyBzaG93VG9hc3QoIlNlbGVjdCBhIHBvaW50IGZpcnN0IiwiZXJyb3IiKTsgcmV0
dXJuOyB9CiAgICAgICAgICAgICAgICAgIHNldERldkFjdGlvbihkZXZBY3Rpb249PT0ibW92ZSI/bnVsbDoibW92ZSIpOwogICAg
ICAgICAgICAgICAgfX0gc3R5bGU9e3sKICAgICAgICAgICAgICAgICAgZmxleDoxLHBhZGRpbmc6IjZweCIsYm9yZGVyUmFkaXVz
OjYsYm9yZGVyOmRldkFjdGlvbj09PSJtb3ZlIj8iMnB4IHNvbGlkICMzYjgyZjYiOiIxcHggc29saWQgIzMzNDE1NSIsCiAgICAg
ICAgICAgICAgICAgIGN1cnNvcjoicG9pbnRlciIsYmFja2dyb3VuZDpkZXZBY3Rpb249PT0ibW92ZSI/IiMxZTNhNWYiOiIjMWUy
OTNiIiwKICAgICAgICAgICAgICAgICAgY29sb3I6ZGV2QWN0aW9uPT09Im1vdmUiPyIjNjBhNWZhIjoiIzk0YTNiOCIsZm9udFdl
aWdodDo2MDAsZm9udFNpemU6MTEsCiAgICAgICAgICAgICAgICB9fT57ZGV2QWN0aW9uPT09Im1vdmUiPyLwn5S1IENsaWNrIE1h
cC4uLiI6IuKcpSBNb3ZlIn08L2J1dHRvbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IHN0eWxlPXt7
ZGlzcGxheToiZmxleCIsZ2FwOjV9fT4KICAgICAgICAgICAgICAgIDxidXR0b24gb25DbGljaz17KCk9PnsKICAgICAgICAgICAg
ICAgICAgaWYgKCFzZWxlY3RlZCkgeyBzaG93VG9hc3QoIlNlbGVjdCBhIHBvaW50IGZpcnN0IiwiZXJyb3IiKTsgcmV0dXJuOyB9
CiAgICAgICAgICAgICAgICAgIGRldkR1cGxpY2F0ZSgpOwogICAgICAgICAgICAgICAgfX0gc3R5bGU9e3tmbGV4OjEscGFkZGlu
ZzoiNnB4Iixib3JkZXJSYWRpdXM6Nixib3JkZXI6IjFweCBzb2xpZCAjMzM0MTU1IixjdXJzb3I6InBvaW50ZXIiLGJhY2tncm91
bmQ6IiMxZTI5M2IiLGNvbG9yOiIjYTc4YmZhIixmb250V2VpZ2h0OjYwMCxmb250U2l6ZToxMX19PgogICAgICAgICAgICAgICAg
ICDwn5OLIER1cGxpY2F0ZQogICAgICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9
eygpPT57CiAgICAgICAgICAgICAgICAgIGlmICghc2VsZWN0ZWQpIHsgc2hvd1RvYXN0KCJTZWxlY3QgYSBwb2ludCBmaXJzdCIs
ImVycm9yIik7IHJldHVybjsgfQogICAgICAgICAgICAgICAgICBkZXZSZW1vdmUoKTsKICAgICAgICAgICAgICAgIH19IHN0eWxl
PXt7ZmxleDoxLHBhZGRpbmc6IjZweCIsYm9yZGVyUmFkaXVzOjYsYm9yZGVyOiIxcHggc29saWQgIzdmMWQxZCIsY3Vyc29yOiJw
b2ludGVyIixiYWNrZ3JvdW5kOiIjMWUyOTNiIixjb2xvcjoiI2Y4NzE3MSIsZm9udFdlaWdodDo2MDAsZm9udFNpemU6MTF9fT4K
ICAgICAgICAgICAgICAgICAg8J+XkSBSZW1vdmUKICAgICAgICAgICAgICAgIDwvYnV0dG9uPgogICAgICAgICAgICAgIDwvZGl2
PgogICAgICAgICAgICAgIDxidXR0b24gb25DbGljaz17KCk9PnNldERldkFjdGlvbihkZXZBY3Rpb249PT0iZGVtb2xpc2hlZCI/
bnVsbDoiZGVtb2xpc2hlZCIpfSBzdHlsZT17ewogICAgICAgICAgICAgICAgd2lkdGg6IjEwMCUiLHBhZGRpbmc6IjZweCIsYm9y
ZGVyUmFkaXVzOjYsCiAgICAgICAgICAgICAgICBib3JkZXI6ZGV2QWN0aW9uPT09ImRlbW9saXNoZWQiPyIycHggc29saWQgIzk0
YTNiOCI6IjFweCBzb2xpZCAjMzM0MTU1IiwKICAgICAgICAgICAgICAgIGN1cnNvcjoicG9pbnRlciIsYmFja2dyb3VuZDpkZXZB
Y3Rpb249PT0iZGVtb2xpc2hlZCI/IiMzMzQxNTUiOiIjMWUyOTNiIiwKICAgICAgICAgICAgICAgIGNvbG9yOmRldkFjdGlvbj09
PSJkZW1vbGlzaGVkIj8iI2UyZThmMCI6IiM5NGEzYjgiLGZvbnRXZWlnaHQ6NjAwLGZvbnRTaXplOjExLAogICAgICAgICAgICAg
IH19PntkZXZBY3Rpb249PT0iZGVtb2xpc2hlZCI/IuKclSBDbGljayBNYXAgdG8gTWFyay4uLiI6IuKclSBNYXJrIERlbW9saXNo
ZWQifTwvYnV0dG9uPgogICAgICAgICAgICAgIHtkZXZFZGl0Q291bnQgPiAwICYmICgKICAgICAgICAgICAgICAgIDxkaXYgc3R5
bGU9e3tkaXNwbGF5OiJmbGV4IixnYXA6NX19PgogICAgICAgICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9eygpPT57IGlmIChj
b25maXJtKCJDbGVhciBhbGwgZWRpdHMgYW5kIHN5bmM/XG5cblN5bmNlZCBkZWxldGlvbnMgY2Fubm90IGJlIHJlc3RvcmVkLiBV
c2UgUHVsbCBTaGVldCB0byByZWxvYWQuIikpIGRldlJlc2V0QWxsKCk7IH19IHN0eWxlPXt7CiAgICAgICAgICAgICAgICAgICAg
ZmxleDoxLHBhZGRpbmc6IjVweCIsYm9yZGVyUmFkaXVzOjUsYm9yZGVyOiIxcHggc29saWQgIzMzNDE1NSIsY3Vyc29yOiJwb2lu
dGVyIixiYWNrZ3JvdW5kOiIjMWUyOTNiIixjb2xvcjoiI2Y1OWUwYiIsZm9udFdlaWdodDo2MDAsZm9udFNpemU6MTAsCiAgICAg
ICAgICAgICAgICAgIH19PuKGqSBDbGVhciAoe2RldkVkaXRDb3VudH0pPC9idXR0b24+CiAgICAgICAgICAgICAgICAgIHtkZXZE
aXJ0eSAmJiAoCiAgICAgICAgICAgICAgICAgICAgPGJ1dHRvbiBvbkNsaWNrPXtkZXZTeW5jVG9TZXJ2ZXJ9IGRpc2FibGVkPXtk
ZXZTeW5jaW5nfSBzdHlsZT17ewogICAgICAgICAgICAgICAgICAgICAgZmxleDoxLHBhZGRpbmc6IjVweCIsYm9yZGVyUmFkaXVz
OjUsYm9yZGVyOiJub25lIixjdXJzb3I6ZGV2U3luY2luZz8id2FpdCI6InBvaW50ZXIiLAogICAgICAgICAgICAgICAgICAgICAg
YmFja2dyb3VuZDpkZXZTeW5jaW5nPyIjMzM0MTU1IjoiIzE2YTM0YSIsY29sb3I6IiNmZmYiLGZvbnRXZWlnaHQ6NjAwLGZvbnRT
aXplOjEwLAogICAgICAgICAgICAgICAgICAgIH19PntkZXZTeW5jaW5nID8gIuKPsy4uLiIgOiAi4piB77iPIFN5bmMgTm93In08
L2J1dHRvbj4KICAgICAgICAgICAgICAgICAgKX0KICAgICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgICl9CiAgICAg
ICAgICAgICAgeyFkZXZEaXJ0eSAmJiBkZXZFZGl0Q291bnQgPiAwICYmICgKICAgICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3tm
b250U2l6ZToxMCxjb2xvcjoiIzRhZGU4MCIsdGV4dEFsaWduOiJjZW50ZXIifX0+4pyTIFN5bmNlZDwvZGl2PgogICAgICAgICAg
ICAgICl9CiAgICAgICAgICAgICAge3NlbGVjdGVkICYmIHNlbEIgJiYgKAogICAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2Zv
bnRTaXplOjEwLGNvbG9yOiIjYThhMjllIixsaW5lSGVpZ2h0OjEuNH19PgogICAgICAgICAgICAgICAgICBTZWxlY3RlZDogPGIg
c3R5bGU9e3tjb2xvcjoiI2ZiYmYyNCJ9fT57c2VsQi51aWR9PC9iPiBhdCB7c2VsQi5sYXQudG9GaXhlZCg2KX0sIHtzZWxCLmxu
Zy50b0ZpeGVkKDYpfQogICAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgICAgKX0KICAgICAgICAgICAgPC9kaXY+CiAg
ICAgICAgICA8L1BvcG92ZXI+CgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KCiAgICAgIHsvKiDilZDilZDilZAgQk9EWTog
TWFwICsgcmlnaHQgZGV0YWlsIHBhbmVsIOKVkOKVkOKVkCAqL30KICAgICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGZs
ZXhEaXJlY3Rpb246InJvdyIsZmxleDoxLG1pbkhlaWdodDowfX0+CiAgICAgIHsvKiDilZDilZDilZAgQ0VOVEVSOiBNYXAg4pWQ
4pWQ4pWQICovfQogICAgICA8ZGl2IHN0eWxlPXt7ZmxleDoxLHBvc2l0aW9uOiJyZWxhdGl2ZSJ9fT4KICAgICAgICA8ZGl2IHJl
Zj17bWFwUmVmfSBzdHlsZT17e3dpZHRoOiIxMDAlIixoZWlnaHQ6IjEwMCUifX0vPgogICAgICAgIHtkZXZBY3Rpb24gJiYgKAog
ICAgICAgICAgPGRpdiBzdHlsZT17ewogICAgICAgICAgICBwb3NpdGlvbjoiYWJzb2x1dGUiLHRvcDoxMixsZWZ0OiI1MCUiLHRy
YW5zZm9ybToidHJhbnNsYXRlWCgtNTAlKSIsekluZGV4OjEwMDAsCiAgICAgICAgICAgIHBhZGRpbmc6IjhweCAyMHB4Iixib3Jk
ZXJSYWRpdXM6OCwKICAgICAgICAgICAgYmFja2dyb3VuZDpkZXZBY3Rpb249PT0iYWRkIj8icmdiYSgyMiwxMDEsNTIsLjkpIjpk
ZXZBY3Rpb249PT0iZGVtb2xpc2hlZCI/InJnYmEoNTEsNjUsODUsLjkpIjoicmdiYSgzMCw1OCwxMzgsLjkpIiwKICAgICAgICAg
ICAgY29sb3I6IiNmZmYiLGZvbnRXZWlnaHQ6NjAwLGZvbnRTaXplOjEzLGJveFNoYWRvdzoiMCA0cHggMTZweCByZ2JhKDAsMCww
LC40KSIsCiAgICAgICAgICAgIGRpc3BsYXk6ImZsZXgiLGFsaWduSXRlbXM6ImNlbnRlciIsZ2FwOjEwLAogICAgICAgICAgfX0+
CiAgICAgICAgICAgIHtkZXZBY3Rpb249PT0iYWRkIiA/ICLinpUgQ2xpY2sgbWFwIHRvIHBsYWNlIG5ldyBwb2ludCIgOiBkZXZB
Y3Rpb249PT0iZGVtb2xpc2hlZCIgPyAi4pyVIENsaWNrIG1hcCB0byBtYXJrIGRlbW9saXNoZWQgc2l0ZSIgOiAi4pylIENsaWNr
IG1hcCB0byBtb3ZlIHNlbGVjdGVkIHBvaW50In0KICAgICAgICAgICAgPGJ1dHRvbiBvbkNsaWNrPXsoKT0+c2V0RGV2QWN0aW9u
KG51bGwpfSBzdHlsZT17e2JhY2tncm91bmQ6InJnYmEoMjU1LDI1NSwyNTUsLjIpIixib3JkZXI6Im5vbmUiLGNvbG9yOiIjZmZm
Iixib3JkZXJSYWRpdXM6NCxwYWRkaW5nOiIycHggOHB4IixjdXJzb3I6InBvaW50ZXIiLGZvbnRXZWlnaHQ6NzAwfX0+Q2FuY2Vs
PC9idXR0b24+CiAgICAgICAgICA8L2Rpdj4KICAgICAgICApfQogICAgICAgIHtkcmF3aW5nTW9kZSAmJiAoCiAgICAgICAgICA8
ZGl2IHN0eWxlPXt7CiAgICAgICAgICAgIHBvc2l0aW9uOiJhYnNvbHV0ZSIsdG9wOjEyLGxlZnQ6IjUwJSIsdHJhbnNmb3JtOiJ0
cmFuc2xhdGVYKC01MCUpIix6SW5kZXg6MTAwMCwKICAgICAgICAgICAgcGFkZGluZzoiOHB4IDIwcHgiLGJvcmRlclJhZGl1czo4
LAogICAgICAgICAgICBiYWNrZ3JvdW5kOiJyZ2JhKDEyMCw4MCwwLC45KSIsCiAgICAgICAgICAgIGNvbG9yOiIjZmZmIixmb250
V2VpZ2h0OjYwMCxmb250U2l6ZToxMyxib3hTaGFkb3c6IjAgNHB4IDE2cHggcmdiYSgwLDAsMCwuNCkiLAogICAgICAgICAgICBk
aXNwbGF5OiJmbGV4IixhbGlnbkl0ZW1zOiJjZW50ZXIiLGdhcDoxMCwKICAgICAgICAgIH19PgogICAgICAgICAgICDinI/vuI8g
Q2xpY2sgdG8gZHJhdyBwb2x5Z29uICh7ZHJhd1BvaW50Q291bnR9IHB0cykKICAgICAgICAgICAgPGJ1dHRvbiBvbkNsaWNrPXtm
aW5pc2hEcmF3aW5nfSBzdHlsZT17e2JhY2tncm91bmQ6IiMxNmEzNGEiLGJvcmRlcjoibm9uZSIsY29sb3I6IiNmZmYiLGJvcmRl
clJhZGl1czo0LHBhZGRpbmc6IjRweCAxMnB4IixjdXJzb3I6InBvaW50ZXIiLGZvbnRXZWlnaHQ6NzAwfX0+RG9uZTwvYnV0dG9u
PgogICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9e2NhbmNlbERyYXdpbmd9IHN0eWxlPXt7YmFja2dyb3VuZDoicmdiYSgyNTUs
MjU1LDI1NSwuMikiLGJvcmRlcjoibm9uZSIsY29sb3I6IiNmZmYiLGJvcmRlclJhZGl1czo0LHBhZGRpbmc6IjRweCA4cHgiLGN1
cnNvcjoicG9pbnRlciIsZm9udFdlaWdodDo3MDB9fT5DYW5jZWw8L2J1dHRvbj4KICAgICAgICAgIDwvZGl2PgogICAgICAgICl9
CiAgICAgICAge2xvYWRpbmcgJiYgKAogICAgICAgICAgPGRpdiBzdHlsZT17e3Bvc2l0aW9uOiJhYnNvbHV0ZSIsaW5zZXQ6MCxk
aXNwbGF5OiJmbGV4IixhbGlnbkl0ZW1zOiJjZW50ZXIiLGp1c3RpZnlDb250ZW50OiJjZW50ZXIiLGJhY2tncm91bmQ6InJnYmEo
MTUsMjMsNDIsLjgpIix6SW5kZXg6MTAwMH19PgogICAgICAgICAgICA8ZGl2IHN0eWxlPXt7Y29sb3I6IiM5NGEzYjgiLHRleHRB
bGlnbjoiY2VudGVyIn19PgogICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3tmb250U2l6ZToyNCxtYXJnaW5Cb3R0b206OCxhbmlt
YXRpb246InB1bHNlIDEuNXMgaW5maW5pdGUifX0+4piB77iPPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2ZvbnRT
aXplOjE0LGZvbnRXZWlnaHQ6NjAwfX0+TG9hZGluZyBmcm9tIEdvb2dsZSBTaGVldC4uLjwvZGl2PgogICAgICAgICAgICA8L2Rp
dj4KICAgICAgICAgIDwvZGl2PgogICAgICAgICl9CiAgICAgIDwvZGl2PgoKICAgICAgey8qIOKVkOKVkOKVkCBSSUdIVCBQQU5F
TDogQnVpbGRpbmcgRGV0YWlsIOKVkOKVkOKVkCAqL30KICAgICAgPGRpdiBzdHlsZT17e3dpZHRoOjM4MCxtaW5XaWR0aDozODAs
ZGlzcGxheToiZmxleCIsZmxleERpcmVjdGlvbjoiY29sdW1uIixib3JkZXJMZWZ0OiIxcHggc29saWQgIzFlMjkzYiIsYmFja2dy
b3VuZDoiIzBmMTcyYSIsb3ZlcmZsb3c6ImF1dG8ifX0+CiAgICAgICAgPGRpdiBzdHlsZT17e3BhZGRpbmc6IjEycHggMTRweCJ9
fT4KICAgICAgICAgIDxkaXYgc3R5bGU9e3tkaXNwbGF5OiJmbGV4IixnYXA6NCxtYXJnaW5Cb3R0b206MTJ9fT4KICAgICAgICAg
ICAgPGlucHV0IHR5cGU9InRleHQiIHBsYWNlaG9sZGVyPSJTZWFyY2ggYnkgSUQuLi4iIHZhbHVlPXtzZWFyY2hJZH0gb25DaGFu
Z2U9e2U9PnNldFNlYXJjaElkKGUudGFyZ2V0LnZhbHVlKX0gb25LZXlEb3duPXtlPT57aWYoZS5rZXk9PT0iRW50ZXIiKSBzZWFy
Y2hCeUlkKCk7fX0gc3R5bGU9e3tmbGV4OjEscGFkZGluZzoiN3B4IDEwcHgiLGJvcmRlclJhZGl1czo3LGJvcmRlcjoiMXB4IHNv
bGlkICMzMzQxNTUiLGJhY2tncm91bmQ6IiMxZTI5M2IiLGNvbG9yOiIjZTJlOGYwIixmb250U2l6ZToxMn19Lz4KICAgICAgICAg
ICAgPGJ1dHRvbiBvbkNsaWNrPXtzZWFyY2hCeUlkfSBzdHlsZT17e3BhZGRpbmc6IjdweCAxNHB4Iixib3JkZXJSYWRpdXM6Nyxi
b3JkZXI6Im5vbmUiLGN1cnNvcjoicG9pbnRlciIsYmFja2dyb3VuZDoiIzMzNDE1NSIsY29sb3I6IiNlMmU4ZjAiLGZvbnRXZWln
aHQ6NzAwLGZvbnRTaXplOjEyfX0+8J+UjTwvYnV0dG9uPgogICAgICAgICAgPC9kaXY+CiAgICAgICAgICB7YnVpbGRpbmdzLmxl
bmd0aCA9PT0gMCAmJiAhbG9hZGluZyA/ICgKICAgICAgICAgICAgPGRpdiBzdHlsZT17e3RleHRBbGlnbjoiY2VudGVyIixtYXJn
aW5Ub3A6NjAsY29sb3I6IiM0NzU1NjkifX0+CiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2ZvbnRTaXplOjQwLG1hcmdpbkJv
dHRvbTo4fX0+4piB77iPPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2ZvbnRTaXplOjEzLGZvbnRXZWlnaHQ6NjAw
LGNvbG9yOiIjOTRhM2I4In19Pk5vIGRhdGEgbG9hZGVkPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2ZvbnRTaXpl
OjExLG1hcmdpblRvcDo0LG1hcmdpbkJvdHRvbToxNH19PlB1bGwgZGF0YSBmcm9tIEdvb2dsZSBTaGVldDwvZGl2PgogICAgICAg
ICAgICAgIDxidXR0b24gb25DbGljaz17cHVsbEZyb21TaGVldH0gZGlzYWJsZWQ9e3B1bGxpbmd9IHN0eWxlPXt7cGFkZGluZzoi
OHB4IDIwcHgiLGJvcmRlclJhZGl1czo3LGJvcmRlcjoibm9uZSIsY3Vyc29yOiJwb2ludGVyIixiYWNrZ3JvdW5kOiIjMWQ0ZWQ4
Iixjb2xvcjoiI2ZmZiIsZm9udFdlaWdodDo3MDAsZm9udFNpemU6MTN9fT4KICAgICAgICAgICAgICAgIHtwdWxsaW5nID8gIuKP
syBQdWxsaW5nLi4uIiA6ICLirIcgUHVsbCBmcm9tIFNoZWV0In0KICAgICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgICAg
PC9kaXY+CiAgICAgICAgICApIDogIXNlbGVjdGVkID8gKAogICAgICAgICAgICA8ZGl2IHN0eWxlPXt7dGV4dEFsaWduOiJjZW50
ZXIiLG1hcmdpblRvcDo2MCxjb2xvcjoiIzQ3NTU2OSJ9fT4KICAgICAgICAgICAgICA8ZGl2IHN0eWxlPXt7Zm9udFNpemU6NDAs
bWFyZ2luQm90dG9tOjh9fT7wn5ONPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2ZvbnRTaXplOjEzLGZvbnRXZWln
aHQ6NjAwfX0+U2VsZWN0IGEgYnVpbGRpbmc8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IHN0eWxlPXt7Zm9udFNpemU6MTEsbWFy
Z2luVG9wOjR9fT5DbGljayBhIG1hcmtlciBvbiB0aGUgbWFwPC9kaXY+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgKSA6
ICgKICAgICAgICAgICAgPGRpdj4KICAgICAgICAgICAgICA8ZGl2IHN0eWxlPXt7ZGlzcGxheToiZmxleCIsanVzdGlmeUNvbnRl
bnQ6InNwYWNlLWJldHdlZW4iLGFsaWduSXRlbXM6ImNlbnRlciIsbWFyZ2luQm90dG9tOjh9fT4KICAgICAgICAgICAgICAgIDxk
aXY+CiAgICAgICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3tmb250U2l6ZToxMyxmb250V2VpZ2h0OjcwMCxjb2xvcjoiI2Y4ZmFm
YyJ9fT4KICAgICAgICAgICAgICAgICAgICBJRDoge3NlbEIubnNpSWQgfHwgc2VsQi51aWR9CiAgICAgICAgICAgICAgICAgICAg
PHNwYW4gc3R5bGU9e3ttYXJnaW5MZWZ0OjYsZm9udFNpemU6OSxwYWRkaW5nOiIycHggNnB4Iixib3JkZXJSYWRpdXM6NCwKICAg
ICAgICAgICAgICAgICAgICAgIGJhY2tncm91bmQ6c2VsQi50eXBlPT09InZlcmlmeSI/IiMxZTNhNWYiOiIjM2IxMjE5IiwKICAg
ICAgICAgICAgICAgICAgICAgIGNvbG9yOnNlbEIudHlwZT09PSJ2ZXJpZnkiPyIjNjBhNWZhIjoiI2ZjYTVhNSIKICAgICAgICAg
ICAgICAgICAgICB9fT57c2VsQi50eXBlPT09InZlcmlmeSI/IlZFUklGWSI6Ik5FVyJ9PC9zcGFuPgogICAgICAgICAgICAgICAg
ICAgIHtzdXJ2ZXlzW3NlbGVjdGVkXSAmJiBzdXJ2ZXlzW3NlbGVjdGVkXS5zYXZlZEF0ICYmIDxzcGFuIHN0eWxlPXt7bWFyZ2lu
TGVmdDo0LGZvbnRTaXplOjkscGFkZGluZzoiMnB4IDZweCIsYm9yZGVyUmFkaXVzOjQsYmFja2dyb3VuZDoiIzE2NjUzNCIsY29s
b3I6IiM0YWRlODAifX0+U0FWRUQ8L3NwYW4+fQogICAgICAgICAgICAgICAgICAgIHsoKHN1cnZleXNbc2VsZWN0ZWRdIHx8IChz
ZWxCICYmIHNlbEIucHJlZmlsbCkgfHwge30pLmZsYWdnZWQgPT09ICJZZXMiKSAmJiA8c3BhbiBzdHlsZT17e21hcmdpbkxlZnQ6
NCxmb250U2l6ZTo5LHBhZGRpbmc6IjJweCA2cHgiLGJvcmRlclJhZGl1czo0LGJhY2tncm91bmQ6IiM0MzE0MDciLGNvbG9yOiIj
ZmI5MjNjIn19PvCfmqk8L3NwYW4+fQogICAgICAgICAgICAgICAgICAgIHsoKHN1cnZleXNbc2VsZWN0ZWRdIHx8IChzZWxCICYm
IHNlbEIucHJlZmlsbCkgfHwge30pLmZsYWdnZWQgPT09ICJEZW1vbGlzaGVkIikgJiYgPHNwYW4gc3R5bGU9e3ttYXJnaW5MZWZ0
OjQsZm9udFNpemU6OSxwYWRkaW5nOiIycHggNnB4Iixib3JkZXJSYWRpdXM6NCxiYWNrZ3JvdW5kOiIjMzM0MTU1Iixjb2xvcjoi
Izk0YTNiOCJ9fT7inJUgREVNT0xJU0hFRDwvc3Bhbj59CiAgICAgICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgICAg
ICA8ZGl2IHN0eWxlPXt7Zm9udFNpemU6MTAsY29sb3I6IiM2NDc0OGIiLG1hcmdpblRvcDoxfX0+e3NlbEIubGF0LnRvRml4ZWQo
Nil9LCB7c2VsQi5sbmcudG9GaXhlZCg2KX08L2Rpdj4KICAgICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgICAgPGJ1
dHRvbiBvbkNsaWNrPXsoKT0+e3NldFNlbGVjdGVkKG51bGwpO3NldEZvcm0oRU1QVFlfRk9STSk7Y2FuY2VsRHJhd2luZygpO319
IHN0eWxlPXt7YmFja2dyb3VuZDoibm9uZSIsYm9yZGVyOiJub25lIixjb2xvcjoiIzY0NzQ4YiIsY3Vyc29yOiJwb2ludGVyIixm
b250U2l6ZToxNn19PuKclTwvYnV0dG9uPgogICAgICAgICAgICAgIDwvZGl2PgoKICAgICAgICAgICAgICA8ZGl2IHN0eWxlPXt7
ZGlzcGxheToiZmxleCIsZ2FwOjUsbWFyZ2luQm90dG9tOjh9fT4KICAgICAgICAgICAgICAgIDxidXR0b24gb25DbGljaz17bmF2
aWdhdGVUb0J1aWxkaW5nfSBzdHlsZT17e2ZsZXg6MSxwYWRkaW5nOiI2cHgiLGJvcmRlclJhZGl1czo3LGJvcmRlcjoibm9uZSIs
Y3Vyc29yOiJwb2ludGVyIixiYWNrZ3JvdW5kOiIjMWQ0ZWQ4Iixjb2xvcjoiI2ZmZiIsZm9udFdlaWdodDo2MDAsZm9udFNpemU6
MTIsZGlzcGxheToiZmxleCIsYWxpZ25JdGVtczoiY2VudGVyIixqdXN0aWZ5Q29udGVudDoiY2VudGVyIixnYXA6NH19PgogICAg
ICAgICAgICAgICAgICDwn6etIE5hdmlnYXRlCiAgICAgICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgICAgICAgIDxidXR0
b24gb25DbGljaz17KCk9PntpZighc2VsZWN0ZWQpcmV0dXJuO2NvbnN0IGI9YnVpbGRpbmdzLmZpbmQoeD0+eC51aWQ9PT1zZWxl
Y3RlZCk7d2luZG93Lm9wZW4oYGh0dHBzOi8vd3d3Lmdvb2dsZS5jb20vbWFwcy9AJHtiLmxhdH0sJHtiLmxuZ30sMTAwbS9kYXRh
PSEzbTEhMWUzYCwiX2JsYW5rIik7fX0gc3R5bGU9e3tmbGV4OjEscGFkZGluZzoiNnB4Iixib3JkZXJSYWRpdXM6Nyxib3JkZXI6
Im5vbmUiLGN1cnNvcjoicG9pbnRlciIsYmFja2dyb3VuZDoiIzAzNjlhMSIsY29sb3I6IiNmZmYiLGZvbnRXZWlnaHQ6NjAwLGZv
bnRTaXplOjEyLGRpc3BsYXk6ImZsZXgiLGFsaWduSXRlbXM6ImNlbnRlciIsanVzdGlmeUNvbnRlbnQ6ImNlbnRlciIsZ2FwOjR9
fT4KICAgICAgICAgICAgICAgICAg8J+MjSAzRCBWaWV3CiAgICAgICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgICAgICA8
L2Rpdj4KCiAgICAgICAgICAgICAgey8qIOKUgOKUgOKUgCBGbGFnICsgTm90ZXMgKGluZGVwZW5kZW50IG9mIFNhdmUpIOKUgOKU
gOKUgCAqL30KICAgICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9e3RvZ2dsZUZsYWd9IGRpc2FibGVkPXtmbGFnZ2luZ3x8c2F2
aW5nfSBzdHlsZT17ewogICAgICAgICAgICAgICAgd2lkdGg6IjEwMCUiLHBhZGRpbmc6IjdweCIsYm9yZGVyUmFkaXVzOjcsY3Vy
c29yOihmbGFnZ2luZ3x8c2F2aW5nKT8id2FpdCI6InBvaW50ZXIiLGZvbnRXZWlnaHQ6NzAwLGZvbnRTaXplOjExLAogICAgICAg
ICAgICAgICAgZGlzcGxheToiZmxleCIsYWxpZ25JdGVtczoiY2VudGVyIixqdXN0aWZ5Q29udGVudDoiY2VudGVyIixnYXA6NSxt
YXJnaW5Cb3R0b206NiwKICAgICAgICAgICAgICAgIGJvcmRlcjpmb3JtLmZsYWdnZWQ9PT0iWWVzIj8iMnB4IHNvbGlkICNmOTcz
MTYiOmZvcm0uZmxhZ2dlZD09PSJEZW1vbGlzaGVkIj8iMnB4IHNvbGlkICM2NDc0OGIiOiIxcHggc29saWQgIzMzNDE1NSIsCiAg
ICAgICAgICAgICAgICBiYWNrZ3JvdW5kOmZvcm0uZmxhZ2dlZD09PSJZZXMiPyIjNDMxNDA3Ijpmb3JtLmZsYWdnZWQ9PT0iRGVt
b2xpc2hlZCI/IiMxZTI5M2IiOiIjMWUyOTNiIiwKICAgICAgICAgICAgICAgIGNvbG9yOmZvcm0uZmxhZ2dlZD09PSJZZXMiPyIj
ZmI5MjNjIjpmb3JtLmZsYWdnZWQ9PT0iRGVtb2xpc2hlZCI/IiM5NGEzYjgiOiIjOTRhM2I4IiwKICAgICAgICAgICAgICB9fT57
ZmxhZ2dpbmc/IuKPsyBTeW5jaW5nLi4uIjpmb3JtLmZsYWdnZWQ9PT0iWWVzIj8i8J+aqSBGbGFnZ2VkIOKAlCBOZWVkcyBTaXRl
IFZpc2l0Ijpmb3JtLmZsYWdnZWQ9PT0iRGVtb2xpc2hlZCI/IuKclSBNYXJrZWQgYXMgRGVtb2xpc2hlZCI6IvCfj7PvuI8gRmxh
ZyBmb3IgU2l0ZSBWaXNpdCJ9PC9idXR0b24+CgogICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3ttYXJnaW5Cb3R0b206MTB9fT4K
ICAgICAgICAgICAgICAgIDxGTEQgbGFiZWw9Ik5vdGVzIj4KICAgICAgICAgICAgICAgICAgPHRleHRhcmVhIHZhbHVlPXtmb3Jt
Lm5vdGVzfSBvbkNoYW5nZT17ZT0+c2V0Rm9ybShmPT4oey4uLmYsbm90ZXM6ZS50YXJnZXQudmFsdWV9KSl9IHBsYWNlaG9sZGVy
PSJPYnNlcnZhdGlvbnMsIGNvbW1lbnRzLi4uIiByb3dzPXsyfSBzdHlsZT17ey4uLmlucCxyZXNpemU6InZlcnRpY2FsIixtYXJn
aW5Cb3R0b206NH19Lz4KICAgICAgICAgICAgICAgIDwvRkxEPgogICAgICAgICAgICAgICAgPGJ1dHRvbiBvbkNsaWNrPXtzYXZl
Tm90ZXN9IGRpc2FibGVkPXtzYXZpbmdOb3Rlc30gc3R5bGU9e3sKICAgICAgICAgICAgICAgICAgd2lkdGg6IjEwMCUiLHBhZGRp
bmc6IjZweCIsYm9yZGVyUmFkaXVzOjYsYm9yZGVyOiJub25lIixjdXJzb3I6c2F2aW5nTm90ZXM/IndhaXQiOiJwb2ludGVyIiwK
ICAgICAgICAgICAgICAgICAgYmFja2dyb3VuZDoiIzMzNDE1NSIsY29sb3I6IiNlMmU4ZjAiLGZvbnRXZWlnaHQ6NjAwLGZvbnRT
aXplOjExLAogICAgICAgICAgICAgICAgfX0+e3NhdmluZ05vdGVzPyLij7MgU2F2aW5nLi4uIjoi8J+TnSBTYXZlIE5vdGVzIn08
L2J1dHRvbj4KICAgICAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2JvcmRlclRvcDoiMXB4IHNv
bGlkICMxZTI5M2IiLHBhZGRpbmdUb3A6MTB9fT4KCiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGZs
ZXhEaXJlY3Rpb246ImNvbHVtbiIsZ2FwOjh9fT4KICAgICAgICAgICAgICAgIDxGTEQgbGFiZWw9IlN1cnZleW9yIj4KICAgICAg
ICAgICAgICAgICAgPGlucHV0IHZhbHVlPXtmb3JtLnN1cnZleW9yfSBvbkNoYW5nZT17ZT0+c2V0Rm9ybShmPT4oey4uLmYsc3Vy
dmV5b3I6ZS50YXJnZXQudmFsdWV9KSl9IHBsYWNlaG9sZGVyPSJZb3VyIG5hbWUiIHN0eWxlPXtpbnB9Lz4KICAgICAgICAgICAg
ICAgIDwvRkxEPgogICAgICAgICAgICAgICAgPEZMRCBsYWJlbD0iQWRkcmVzcyI+CiAgICAgICAgICAgICAgICAgIDxpbnB1dCB2
YWx1ZT17Zm9ybS5hZGRyZXNzfSBvbkNoYW5nZT17ZT0+c2V0Rm9ybShmPT4oey4uLmYsYWRkcmVzczplLnRhcmdldC52YWx1ZX0p
KX0gcGxhY2Vob2xkZXI9IlN0cmVldCBhZGRyZXNzIiBzdHlsZT17aW5wfS8+CiAgICAgICAgICAgICAgICA8L0ZMRD4KICAgICAg
ICAgICAgICAgIDxGTEQgbGFiZWw9IlN0b3JpZXMgKiI+CiAgICAgICAgICAgICAgICAgIDxpbnB1dCB0eXBlPSJudW1iZXIiIG1p
bj0iMSIgbWF4PSI5OSIgdmFsdWU9e2Zvcm0ubnVtU3Rvcmllc30gb25DaGFuZ2U9e2U9PnNldEZvcm0oZj0+KHsuLi5mLG51bVN0
b3JpZXM6ZS50YXJnZXQudmFsdWV9KSl9IHN0eWxlPXtpbnB9Lz4KICAgICAgICAgICAgICAgIDwvRkxEPgogICAgICAgICAgICAg
ICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGdhcDo2fX0+CiAgICAgICAgICAgICAgICAgIDxGTEQgbGFiZWw9Ik9jY3Vw
YW5jeSBUeXBlICoiIHM9e3tmbGV4OjJ9fT4KICAgICAgICAgICAgICAgICAgICA8c2VsZWN0IHZhbHVlPXtmb3JtLm9jY1R5cGV9
IG9uQ2hhbmdlPXtlPT5zZXRGb3JtKGY9Pih7Li4uZixvY2NUeXBlOmUudGFyZ2V0LnZhbHVlfSkpfSBzdHlsZT17ey4uLmlucCxh
cHBlYXJhbmNlOiJhdXRvIn19PgogICAgICAgICAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0iIj7igJQgU2VsZWN0IOKAlDwv
b3B0aW9uPgogICAgICAgICAgICAgICAgICAgICAgPG9wdGdyb3VwIGxhYmVsPSJSZXNpZGVudGlhbCI+CiAgICAgICAgICAgICAg
ICAgICAgICAgIHtPQ0NfVFlQRVMuZmlsdGVyKG89Pm8uY29kZS5zdGFydHNXaXRoKCJSRVMiKSkubWFwKG89PjxvcHRpb24ga2V5
PXtvLmNvZGV9IHZhbHVlPXtvLmNvZGV9PntvLmxhYmVsfTwvb3B0aW9uPil9CiAgICAgICAgICAgICAgICAgICAgICA8L29wdGdy
b3VwPgogICAgICAgICAgICAgICAgICAgICAgPG9wdGdyb3VwIGxhYmVsPSJDb21tZXJjaWFsIj4KICAgICAgICAgICAgICAgICAg
ICAgICAge09DQ19UWVBFUy5maWx0ZXIobz0+by5jb2RlLnN0YXJ0c1dpdGgoIkNPTSIpKS5tYXAobz0+PG9wdGlvbiBrZXk9e28u
Y29kZX0gdmFsdWU9e28uY29kZX0+e28ubGFiZWx9PC9vcHRpb24+KX0KICAgICAgICAgICAgICAgICAgICAgIDwvb3B0Z3JvdXA+
CiAgICAgICAgICAgICAgICAgICAgICA8b3B0Z3JvdXAgbGFiZWw9IkluZHVzdHJpYWwiPgogICAgICAgICAgICAgICAgICAgICAg
ICB7T0NDX1RZUEVTLmZpbHRlcihvPT5vLmNvZGUuc3RhcnRzV2l0aCgiSU5EIikpLm1hcChvPT48b3B0aW9uIGtleT17by5jb2Rl
fSB2YWx1ZT17by5jb2RlfT57by5sYWJlbH08L29wdGlvbj4pfQogICAgICAgICAgICAgICAgICAgICAgPC9vcHRncm91cD4KICAg
ICAgICAgICAgICAgICAgICAgIDxvcHRncm91cCBsYWJlbD0iT3RoZXIiPgogICAgICAgICAgICAgICAgICAgICAgICB7T0NDX1RZ
UEVTLmZpbHRlcihvPT4hL14oUkVTfENPTXxJTkQpLy50ZXN0KG8uY29kZSkpLm1hcChvPT48b3B0aW9uIGtleT17by5jb2RlfSB2
YWx1ZT17by5jb2RlfT57by5sYWJlbH08L29wdGlvbj4pfQogICAgICAgICAgICAgICAgICAgICAgPC9vcHRncm91cD4KICAgICAg
ICAgICAgICAgICAgICA8L3NlbGVjdD4KICAgICAgICAgICAgICAgICAgPC9GTEQ+CiAgICAgICAgICAgICAgICAgIDxGTEQgbGFi
ZWw9IkJsZGcgVHlwZSAqIiBzPXt7ZmxleDoxfX0+CiAgICAgICAgICAgICAgICAgICAgPHNlbGVjdCB2YWx1ZT17Zm9ybS5idWls
ZGluZ1R5cGV9IG9uQ2hhbmdlPXtlPT5zZXRGb3JtKGY9Pih7Li4uZixidWlsZGluZ1R5cGU6ZS50YXJnZXQudmFsdWV9KSl9IHN0
eWxlPXt7Li4uaW5wLGFwcGVhcmFuY2U6ImF1dG8ifX0+CiAgICAgICAgICAgICAgICAgICAgICA8b3B0aW9uIHZhbHVlPSIiPuKA
lDwvb3B0aW9uPgogICAgICAgICAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0iVyI+VyDigJQgV29vZDwvb3B0aW9uPgogICAg
ICAgICAgICAgICAgICAgICAgPG9wdGlvbiB2YWx1ZT0iTSI+TSDigJQgTWFzb25yeTwvb3B0aW9uPgogICAgICAgICAgICAgICAg
ICAgICAgPG9wdGlvbiB2YWx1ZT0iQyI+QyDigJQgQ29uY3JldGU8L29wdGlvbj4KICAgICAgICAgICAgICAgICAgICAgIDxvcHRp
b24gdmFsdWU9IlMiPlMg4oCUIFN0ZWVsPC9vcHRpb24+CiAgICAgICAgICAgICAgICAgICAgICA8b3B0aW9uIHZhbHVlPSJIIj5I
IOKAlCBNYW51ZmFjdHVyZWQ8L29wdGlvbj4KICAgICAgICAgICAgICAgICAgICA8L3NlbGVjdD4KICAgICAgICAgICAgICAgICAg
PC9GTEQ+CiAgICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICAgIDxkaXYgc3R5bGU9e3tkaXNwbGF5OiJmbGV4Iixn
YXA6Nn19PgogICAgICAgICAgICAgICAgICA8RkxEIGxhYmVsPSJGb3VuZGF0aW9uIFR5cGUgKiIgcz17e2ZsZXg6MX19PgogICAg
ICAgICAgICAgICAgICAgIDxzZWxlY3QgdmFsdWU9e2Zvcm0uZm91bmRhdGlvblR5cGV9IG9uQ2hhbmdlPXtlPT5zZXRGb3JtKGY9
Pih7Li4uZixmb3VuZGF0aW9uVHlwZTplLnRhcmdldC52YWx1ZX0pKX0gc3R5bGU9e3suLi5pbnAsYXBwZWFyYW5jZToiYXV0byJ9
fT4KICAgICAgICAgICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IiI+4oCUIFNlbGVjdCDigJQ8L29wdGlvbj4KICAgICAgICAg
ICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IlMiPlMg4oCUIFNsYWIgb24gR3JhZGU8L29wdGlvbj4KICAgICAgICAgICAgICAg
ICAgICAgIDxvcHRpb24gdmFsdWU9IkMiPkMg4oCUIENyYXdsc3BhY2U8L29wdGlvbj4KICAgICAgICAgICAgICAgICAgICAgIDxv
cHRpb24gdmFsdWU9IkIiPkIg4oCUIEJhc2VtZW50PC9vcHRpb24+CiAgICAgICAgICAgICAgICAgICAgICA8b3B0aW9uIHZhbHVl
PSJQIj5QIOKAlCBQaWVyL1BpbGU8L29wdGlvbj4KICAgICAgICAgICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IlciPlcg4oCU
IFNvbGlkIFdhbGw8L29wdGlvbj4KICAgICAgICAgICAgICAgICAgICAgIDxvcHRpb24gdmFsdWU9IlUiPlUg4oCUIFVua25vd248
L29wdGlvbj4KICAgICAgICAgICAgICAgICAgICA8L3NlbGVjdD4KICAgICAgICAgICAgICAgICAgPC9GTEQ+CiAgICAgICAgICAg
ICAgICAgIDxGTEQgbGFiZWw9IjFzdCBGbG9vciBIdCAoZnQpICoiIHM9e3tmbGV4OjF9fT4KICAgICAgICAgICAgICAgICAgICA8
aW5wdXQgdHlwZT0ibnVtYmVyIiBtaW49IjAiIG1heD0iMzAiIHN0ZXA9IjAuNSIgdmFsdWU9e2Zvcm0uZmlyc3RGbG9vckhlaWdo
dH0gb25DaGFuZ2U9e2U9PnNldEZvcm0oZj0+KHsuLi5mLGZpcnN0Rmxvb3JIZWlnaHQ6ZS50YXJnZXQudmFsdWV9KSl9IHN0eWxl
PXtpbnB9Lz4KICAgICAgICAgICAgICAgICAgPC9GTEQ+CiAgICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICAgIDxk
aXYgc3R5bGU9e3tkaXNwbGF5OiJmbGV4IixnYXA6MyxhbGlnbkl0ZW1zOiJmbGV4LWVuZCJ9fT4KICAgICAgICAgICAgICAgICAg
PEZMRCBsYWJlbD0iRm9vdHByaW50IChzcWZ0KSAqIiBzPXt7ZmxleDoxfX0+CiAgICAgICAgICAgICAgICAgICAgPGlucHV0IHR5
cGU9Im51bWJlciIgdmFsdWU9e2Zvcm0uYXJlYX0gb25DaGFuZ2U9e2U9PnNldEZvcm0oZj0+KHsuLi5mLGFyZWE6ZS50YXJnZXQu
dmFsdWV9KSl9IHN0eWxlPXtpbnB9Lz4KICAgICAgICAgICAgICAgICAgPC9GTEQ+CiAgICAgICAgICAgICAgICAgIDxidXR0b24g
b25DbGljaz17YXV0b0ZldGNoQXJlYX0gZGlzYWJsZWQ9e2ZldGNoaW5nQXJlYXx8ZHJhd2luZ01vZGV9IHRpdGxlPSJNaWNyb3Nv
ZnQgQnVpbGRpbmcgRm9vdHByaW50cyIgc3R5bGU9e3twYWRkaW5nOiI3cHggOHB4Iixib3JkZXJSYWRpdXM6Nixib3JkZXI6IjFw
eCBzb2xpZCAjMzM0MTU1IixjdXJzb3I6ZmV0Y2hpbmdBcmVhPyJ3YWl0IjoicG9pbnRlciIsYmFja2dyb3VuZDpmZXRjaGluZ0Fy
ZWE/IiMzMzQxNTUiOiIjMWUzYTVmIixjb2xvcjoiIzYwYTVmYSIsZm9udFdlaWdodDo3MDAsZm9udFNpemU6OSx3aGl0ZVNwYWNl
OiJub3dyYXAifX0+e2ZldGNoaW5nQXJlYSA/ICLij7MiIDogIvCfj6AgQXV0byJ9PC9idXR0b24+CiAgICAgICAgICAgICAgICAg
IDxidXR0b24gb25DbGljaz17ZHJhd2luZ01vZGUgPyBjYW5jZWxEcmF3aW5nIDogc3RhcnREcmF3aW5nfSBkaXNhYmxlZD17ISFk
ZXZBY3Rpb259IHRpdGxlPSJEcmF3IHBvbHlnb24gb24gbWFwIiBzdHlsZT17e3BhZGRpbmc6IjdweCA4cHgiLGJvcmRlclJhZGl1
czo2LGJvcmRlcjpkcmF3aW5nTW9kZT8iMnB4IHNvbGlkICNmNTllMGIiOiIxcHggc29saWQgIzMzNDE1NSIsY3Vyc29yOiJwb2lu
dGVyIixiYWNrZ3JvdW5kOmRyYXdpbmdNb2RlPyIjNDIyMDA2IjoiIzFlM2E1ZiIsY29sb3I6ZHJhd2luZ01vZGU/IiNmYmJmMjQi
OiIjNjBhNWZhIixmb250V2VpZ2h0OjcwMCxmb250U2l6ZTo5LHdoaXRlU3BhY2U6Im5vd3JhcCJ9fT57ZHJhd2luZ01vZGUgPyAi
4pyVIiA6ICLinI/vuI8gRHJhdyJ9PC9idXR0b24+CiAgICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICAgIDxkaXYg
c3R5bGU9e3tkaXNwbGF5OiJmbGV4IixnYXA6Nn19PgogICAgICAgICAgICAgICAgICA8RkxEIGxhYmVsPSJZZWFyIEJ1aWx0IiBz
PXt7ZmxleDoxfX0+CiAgICAgICAgICAgICAgICAgICAgPGlucHV0IHR5cGU9Im51bWJlciIgdmFsdWU9e2Zvcm0ueWVhckJ1aWx0
fSBvbkNoYW5nZT17ZT0+c2V0Rm9ybShmPT4oey4uLmYseWVhckJ1aWx0OmUudGFyZ2V0LnZhbHVlfSkpfSBzdHlsZT17aW5wfS8+
CiAgICAgICAgICAgICAgICAgIDwvRkxEPgogICAgICAgICAgICAgICAgICA8RkxEIGxhYmVsPSJHcm91bmQgRWxldiAoZnQpICoi
IHM9e3tmbGV4OjF9fT4KICAgICAgICAgICAgICAgICAgICA8ZGl2IHN0eWxlPXt7ZGlzcGxheToiZmxleCIsZ2FwOjN9fT4KICAg
ICAgICAgICAgICAgICAgICAgIDxpbnB1dCB0eXBlPSJudW1iZXIiIHN0ZXA9IjAuMDEiIHZhbHVlPXtmb3JtLmdyb3VuZEVsZXZ9
IG9uQ2hhbmdlPXtlPT5zZXRGb3JtKGY9Pih7Li4uZixncm91bmRFbGV2OmUudGFyZ2V0LnZhbHVlfSkpfSBzdHlsZT17ey4uLmlu
cCxmbGV4OjF9fS8+CiAgICAgICAgICAgICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9e2F1dG9GZXRjaEVsZXZhdGlvbn0gZGlz
YWJsZWQ9e2ZldGNoaW5nRWxldn0gdGl0bGU9IlVTR1MgM0RFUCIgc3R5bGU9e3twYWRkaW5nOiI1cHggOHB4Iixib3JkZXJSYWRp
dXM6Nixib3JkZXI6IjFweCBzb2xpZCAjMzM0MTU1IixjdXJzb3I6ZmV0Y2hpbmdFbGV2PyJ3YWl0IjoicG9pbnRlciIsYmFja2dy
b3VuZDpmZXRjaGluZ0VsZXY/IiMzMzQxNTUiOiIjMWUzYTVmIixjb2xvcjoiIzYwYTVmYSIsZm9udFdlaWdodDo3MDAsZm9udFNp
emU6OSx3aGl0ZVNwYWNlOiJub3dyYXAifX0+e2ZldGNoaW5nRWxldiA/ICLij7MiIDogIvCfk5AifTwvYnV0dG9uPgogICAgICAg
ICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgICAgICA8L0ZMRD4KICAgICAgICAgICAgICAgIDwvZGl2PgogICAgICAg
ICAgICAgICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGdhcDo2LGFsaWduSXRlbXM6ImZsZXgtZW5kIn19PgogICAgICAg
ICAgICAgICAgICA8RkxEIGxhYmVsPSJTdHJ1Y3R1cmUgVmFsdWUgKCQpICoiIHM9e3tmbGV4OjF9fT4KICAgICAgICAgICAgICAg
ICAgICA8aW5wdXQgdHlwZT0ibnVtYmVyIiB2YWx1ZT17Zm9ybS5zdHJ1Y3R1cmVWYWx1ZX0gb25DaGFuZ2U9e2U9PnNldEZvcm0o
Zj0+KHsuLi5mLHN0cnVjdHVyZVZhbHVlOmUudGFyZ2V0LnZhbHVlfSkpfSBzdHlsZT17aW5wfS8+CiAgICAgICAgICAgICAgICAg
IDwvRkxEPgogICAgICAgICAgICAgICAgICA8RkxEIGxhYmVsPSJDb250ZW50IFZhbHVlICgkKSAqIiBzPXt7ZmxleDoxfX0+CiAg
ICAgICAgICAgICAgICAgICAgPGlucHV0IHR5cGU9Im51bWJlciIgdmFsdWU9e2Zvcm0uY29udGVudFZhbHVlfSBvbkNoYW5nZT17
ZT0+c2V0Rm9ybShmPT4oey4uLmYsY29udGVudFZhbHVlOmUudGFyZ2V0LnZhbHVlfSkpfSBzdHlsZT17aW5wfS8+CiAgICAgICAg
ICAgICAgICAgIDwvRkxEPgogICAgICAgICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9e2F1dG9Fc3RpbWF0ZUNvc3R9IHRpdGxl
PSJPTFMgcmVncmVzc2lvbiBlc3RpbWF0ZSIgc3R5bGU9e3twYWRkaW5nOiI3cHggOHB4Iixib3JkZXJSYWRpdXM6Nixib3JkZXI6
IjFweCBzb2xpZCAjMzM0MTU1IixjdXJzb3I6InBvaW50ZXIiLGJhY2tncm91bmQ6IiMxZTNhNWYiLGNvbG9yOiIjNjBhNWZhIixm
b250V2VpZ2h0OjcwMCxmb250U2l6ZTo5LHdoaXRlU3BhY2U6Im5vd3JhcCJ9fT7wn5KwPC9idXR0b24+CiAgICAgICAgICAgICAg
ICA8L2Rpdj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICAgICAgPGRpdiBzdHls
ZT17e2Rpc3BsYXk6ImZsZXgiLGdhcDo2LG1hcmdpblRvcDoxMH19PgogICAgICAgICAgICAgICAgPGJ1dHRvbiBvbkNsaWNrPXto
YW5kbGVTYXZlfSBkaXNhYmxlZD17c2F2aW5nfHxmbGFnZ2luZ30gc3R5bGU9e3sKICAgICAgICAgICAgICAgICAgZmxleDoxLHBh
ZGRpbmc6IjlweCIsYm9yZGVyUmFkaXVzOjgsYm9yZGVyOiJub25lIixjdXJzb3I6KHNhdmluZ3x8ZmxhZ2dpbmcpPyJ3YWl0Ijoi
cG9pbnRlciIsCiAgICAgICAgICAgICAgICAgIGJhY2tncm91bmQ6ImxpbmVhci1ncmFkaWVudCgxMzVkZWcsIzE2YTM0YSwjMTU4
MDNkKSIsY29sb3I6IiNmZmYiLGZvbnRXZWlnaHQ6NzAwLGZvbnRTaXplOjEyLAogICAgICAgICAgICAgICAgfX0+e3NhdmluZz8i
U2F2aW5nLi4uIjoi8J+SviBTYXZlIn08L2J1dHRvbj4KICAgICAgICAgICAgICAgIHtzdXJ2ZXlzW3NlbGVjdGVkXSAmJiBzdXJ2
ZXlzW3NlbGVjdGVkXS5zYXZlZEF0ICYmIDxidXR0b24gb25DbGljaz17aGFuZGxlVW5kb1NhdmV9IHN0eWxlPXt7cGFkZGluZzoi
OXB4IDEycHgiLGJvcmRlclJhZGl1czo4LGJvcmRlcjoiMXB4IHNvbGlkICM3ZjFkMWQiLGN1cnNvcjoicG9pbnRlciIsYmFja2dy
b3VuZDoiIzFlMjkzYiIsY29sb3I6IiNmODcxNzEiLGZvbnRXZWlnaHQ6NjAwLGZvbnRTaXplOjEyfX0+8J+XkSBDbGVhcjwvYnV0
dG9uPn0KICAgICAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGdhcDo2
LG1hcmdpblRvcDo2LHBhZGRpbmdCb3R0b206MTZ9fT4KICAgICAgICAgICAgICAgIDxidXR0b24gb25DbGljaz17KCk9Pntjb25z
dCBpPWZpbHRlcmVkVWlkcy5pbmRleE9mKHNlbGVjdGVkKTtjb25zdCBwPWk+MD9pLTE6ZmlsdGVyZWRVaWRzLmxlbmd0aC0xO3dp
bmRvdy5fX3NlbChmaWx0ZXJlZFVpZHNbcF0pO319IHN0eWxlPXthYnRuKCIjMzM0MTU1Iil9PuKGkCBQcmV2PC9idXR0b24+CiAg
ICAgICAgICAgICAgICA8YnV0dG9uIG9uQ2xpY2s9eygpPT57Y29uc3QgaT1maWx0ZXJlZFVpZHMuaW5kZXhPZihzZWxlY3RlZCk7
Y29uc3Qgbj1pPGZpbHRlcmVkVWlkcy5sZW5ndGgtMT9pKzE6MDt3aW5kb3cuX19zZWwoZmlsdGVyZWRVaWRzW25dKTt9fSBzdHls
ZT17YWJ0bigiIzMzNDE1NSIpfT5OZXh0IOKGkjwvYnV0dG9uPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8L2Rp
dj4KICAgICAgICAgICl9CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KCiAgICAgIDxzdHlsZT57YAog
ICAgICAgIC5sZWFmbGV0LWNvbnRhaW5lciB7IGJhY2tncm91bmQ6ICMwZjE3MmEgIWltcG9ydGFudDsgfQogICAgICAgICR7KGRl
dkFjdGlvbiB8fCBkcmF3aW5nTW9kZSkgPyAnLmxlYWZsZXQtY29udGFpbmVyIHsgY3Vyc29yOiBjcm9zc2hhaXIgIWltcG9ydGFu
dDsgfScgOiAnJ30KICAgICAgICAkeyhkcmF3aW5nTW9kZSB8fCBkZXZBY3Rpb24gPT09ICJtb3ZlIiB8fCBkZXZBY3Rpb24gPT09
ICJkZW1vbGlzaGVkIikgPyAnLmxlYWZsZXQtbWFya2VyLXBhbmUsIC5sZWFmbGV0LW92ZXJsYXktcGFuZSBjaXJjbGUsIC5sZWFm
bGV0LW92ZXJsYXktcGFuZSBwYXRoIHsgcG9pbnRlci1ldmVudHM6IG5vbmUgIWltcG9ydGFudDsgfScgOiAnJ30KICAgICAgICBA
a2V5ZnJhbWVzIHB1bHNlIHsgMCUsMTAwJSB7IG9wYWNpdHk6MTsgfSA1MCUgeyBvcGFjaXR5OjAuNzsgfSB9CiAgICAgICAgOjot
d2Via2l0LXNjcm9sbGJhciB7IHdpZHRoOiA1cHg7IH0KICAgICAgICA6Oi13ZWJraXQtc2Nyb2xsYmFyLXRyYWNrIHsgYmFja2dy
b3VuZDogIzBmMTcyYTsgfQogICAgICAgIC5sZWFmbGV0LWNvbnRyb2wtbGF5ZXJzIHsgYmFja2dyb3VuZDogcmdiYSgxNSwyMyw0
MiwuOSkgIWltcG9ydGFudDsgYm9yZGVyOiAxcHggc29saWQgIzMzNDE1NSAhaW1wb3J0YW50OyBib3JkZXItcmFkaXVzOiA4cHgg
IWltcG9ydGFudDsgcGFkZGluZzogOHB4IDEycHggIWltcG9ydGFudDsgY29sb3I6ICNlMmU4ZjAgIWltcG9ydGFudDsgZm9udC1z
aXplOiAxMnB4ICFpbXBvcnRhbnQ7IGZvbnQtd2VpZ2h0OiA2MDAgIWltcG9ydGFudDsgYmFja2Ryb3AtZmlsdGVyOiBibHVyKDhw
eCk7IGJveC1zaGFkb3c6IDAgNHB4IDEycHggcmdiYSgwLDAsMCwuNCkgIWltcG9ydGFudDsgfQogICAgICAgIC5sZWFmbGV0LWNv
bnRyb2wtbGF5ZXJzIGxhYmVsIHsgY29sb3I6ICNlMmU4ZjAgIWltcG9ydGFudDsgY3Vyc29yOiBwb2ludGVyOyBkaXNwbGF5OiBm
bGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBnYXA6IDZweDsgcGFkZGluZzogMnB4IDA7IH0KICAgICAgICAubGVhZmxldC1jb250
cm9sLWxheWVycy1zZXBhcmF0b3IgeyBib3JkZXItdG9wOiAxcHggc29saWQgIzMzNDE1NSAhaW1wb3J0YW50OyB9CiAgICAgICAg
Ojotd2Via2l0LXNjcm9sbGJhci10aHVtYiB7IGJhY2tncm91bmQ6ICMzMzQxNTU7IGJvcmRlci1yYWRpdXM6IDNweDsgfQogICAg
ICBgfTwvc3R5bGU+CiAgICA8L2Rpdj4KICApOwp9CgpmdW5jdGlvbiBGTEQoe2xhYmVsLGNoaWxkcmVuLHN9KSB7CiAgcmV0dXJu
IDxkaXYgc3R5bGU9e3N9PjxsYWJlbCBzdHlsZT17e2Rpc3BsYXk6ImJsb2NrIixmb250U2l6ZToxMSxmb250V2VpZ2h0OjYwMCxj
b2xvcjoiIzk0YTNiOCIsbWFyZ2luQm90dG9tOjMsdGV4dFRyYW5zZm9ybToidXBwZXJjYXNlIixsZXR0ZXJTcGFjaW5nOi41fX0+
e2xhYmVsfTwvbGFiZWw+e2NoaWxkcmVufTwvZGl2PjsKfQoKY29uc3QgaW5wID0ge3dpZHRoOiIxMDAlIixwYWRkaW5nOiI4cHgg
MTBweCIsYm9yZGVyUmFkaXVzOjcsYm9yZGVyOiIxcHggc29saWQgIzMzNDE1NSIsYmFja2dyb3VuZDoiIzFlMjkzYiIsY29sb3I6
IiNlMmU4ZjAiLGZvbnRTaXplOjEzLG91dGxpbmU6Im5vbmUiLGJveFNpemluZzoiYm9yZGVyLWJveCJ9OwoKLy8g4pSA4pSA4pSA
IENvbXBhY3QgdG9wLWJhciBoZWxwZXJzIChOU0kgdGFiIHRvb2xiYXIpIOKUgOKUgOKUgAovLyBMaWdodHdlaWdodCBjbGljay1v
dXRzaWRlIHBvcG92ZXIgc28gdGhlIGRlbnNlIGNvbnRyb2xzIChhZHZhbmNlZCBmaWx0ZXJzLAovLyBkZXZlbG9wZXIgcG9pbnQt
bWFuYWdlbWVudCkgY29sbGFwc2UgaW50byBvbi1kZW1hbmQgcGFuZWxzLCBrZWVwaW5nIHRoZQovLyB0b29sYmFyIHRvIGEgc2lu
Z2xlIGxvdyByb3cuCmZ1bmN0aW9uIFBvcG92ZXIoeyBsYWJlbCwgcGFuZWxXaWR0aCA9IDI2MCwgYWxpZ24gPSAibGVmdCIsIGFj
Y2VudCA9ICIjMzM0MTU1IiwgY2hpbGRyZW4gfSkgewogIGNvbnN0IFtvcGVuLCBzZXRPcGVuXSA9IFJlYWN0LnVzZVN0YXRlKGZh
bHNlKTsKICBjb25zdCByZWYgPSBSZWFjdC51c2VSZWYobnVsbCk7CiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHsKICAgIGlmICgh
b3BlbikgcmV0dXJuOwogICAgY29uc3QgaCA9IChlKSA9PiB7IGlmIChyZWYuY3VycmVudCAmJiAhcmVmLmN1cnJlbnQuY29udGFp
bnMoZS50YXJnZXQpKSBzZXRPcGVuKGZhbHNlKTsgfTsKICAgIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoIm1vdXNlZG93biIs
IGgpOwogICAgcmV0dXJuICgpID0+IGRvY3VtZW50LnJlbW92ZUV2ZW50TGlzdGVuZXIoIm1vdXNlZG93biIsIGgpOwogIH0sIFtv
cGVuXSk7CiAgcmV0dXJuICgKICAgIDxkaXYgcmVmPXtyZWZ9IHN0eWxlPXt7IHBvc2l0aW9uOiAicmVsYXRpdmUiLCBmbGV4OiAi
MCAwIGF1dG8iIH19PgogICAgICA8YnV0dG9uIG9uQ2xpY2s9eygpID0+IHNldE9wZW4obyA9PiAhbyl9IHN0eWxlPXt7CiAgICAg
ICAgcGFkZGluZzogIjZweCAxMXB4IiwgYm9yZGVyUmFkaXVzOiA3LCBib3JkZXI6ICIxcHggc29saWQgIiArIChvcGVuID8gYWNj
ZW50IDogIiMzMzQxNTUiKSwKICAgICAgICBiYWNrZ3JvdW5kOiBvcGVuID8gIiMxZTI5M2IiIDogIiMxMTE4MjciLCBjb2xvcjog
IiNlMmU4ZjAiLAogICAgICAgIGZvbnRXZWlnaHQ6IDYwMCwgZm9udFNpemU6IDEyLCBjdXJzb3I6ICJwb2ludGVyIiwgd2hpdGVT
cGFjZTogIm5vd3JhcCIsCiAgICAgICAgZGlzcGxheTogImZsZXgiLCBhbGlnbkl0ZW1zOiAiY2VudGVyIiwgZ2FwOiA2LAogICAg
ICB9fT57bGFiZWx9PHNwYW4gc3R5bGU9e3sgZm9udFNpemU6IDgsIGNvbG9yOiAiIzk0YTNiOCIgfX0+e29wZW4gPyAi4payIiA6
ICLilrwifTwvc3Bhbj48L2J1dHRvbj4KICAgICAge29wZW4gJiYgKAogICAgICAgIDxkaXYgc3R5bGU9e3sKICAgICAgICAgIHBv
c2l0aW9uOiAiYWJzb2x1dGUiLCB0b3A6ICJjYWxjKDEwMCUgKyA2cHgpIiwgW2FsaWduXTogMCwgekluZGV4OiAzMDAwLAogICAg
ICAgICAgd2lkdGg6IHBhbmVsV2lkdGgsIGJhY2tncm91bmQ6ICIjMGYxNzJhIiwgYm9yZGVyOiAiMXB4IHNvbGlkICMzMzQxNTUi
LAogICAgICAgICAgYm9yZGVyUmFkaXVzOiAxMCwgcGFkZGluZzogMTIsIGJveFNoYWRvdzogIjAgMTRweCAzOHB4IHJnYmEoMCww
LDAsLjU1KSIsCiAgICAgICAgfX0+e2NoaWxkcmVufTwvZGl2PgogICAgICApfQogICAgPC9kaXY+CiAgKTsKfQoKY29uc3QgdGJE
aXZpZGVyID0geyB3aWR0aDogMSwgYWxpZ25TZWxmOiAic3RyZXRjaCIsIG1pbkhlaWdodDogMjIsIGJhY2tncm91bmQ6ICIjMWUy
OTNiIiwgbWFyZ2luOiAiMCAxcHgiIH07CmNvbnN0IHRiSWNvbkJ0biA9IChiZykgPT4gKHsKICB3aWR0aDogMzQsIGhlaWdodDog
MzEsIGJvcmRlclJhZGl1czogNywgYm9yZGVyOiAibm9uZSIsIGN1cnNvcjogInBvaW50ZXIiLAogIGJhY2tncm91bmQ6IGJnLCBj
b2xvcjogIiNmZmYiLCBmb250U2l6ZTogMTQsIGxpbmVIZWlnaHQ6IDEsIGRpc3BsYXk6ICJmbGV4IiwKICBhbGlnbkl0ZW1zOiAi
Y2VudGVyIiwganVzdGlmeUNvbnRlbnQ6ICJjZW50ZXIiLCBmbGV4OiAiMCAwIGF1dG8iLAp9KTsKLy8gTGFiZWxlZCBjb21wYWN0
IGJ1dHRvbiAoaWNvbiArIHRleHQpIGZvciB0aGUgY29sb3JmdWwgYWN0aW9uIGNvbnRyb2xzLgpjb25zdCB0YkJ0bkwgPSAoYmcp
ID0+ICh7CiAgaGVpZ2h0OiAzMSwgcGFkZGluZzogIjAgMTFweCIsIGJvcmRlclJhZGl1czogNywgYm9yZGVyOiAibm9uZSIsIGN1
cnNvcjogInBvaW50ZXIiLAogIGJhY2tncm91bmQ6IGJnLCBjb2xvcjogIiNmZmYiLCBmb250U2l6ZTogMTEuNSwgZm9udFdlaWdo
dDogNjAwLCBsaW5lSGVpZ2h0OiAxLAogIGRpc3BsYXk6ICJmbGV4IiwgYWxpZ25JdGVtczogImNlbnRlciIsIGdhcDogNSwgd2hp
dGVTcGFjZTogIm5vd3JhcCIsIGZsZXg6ICIwIDAgYXV0byIsCn0pOwpjb25zdCB0YkxibCA9IHsgZm9udFNpemU6IDksIGZvbnRX
ZWlnaHQ6IDcwMCwgY29sb3I6ICIjOTRhM2I4IiwgdGV4dFRyYW5zZm9ybTogInVwcGVyY2FzZSIsIGxldHRlclNwYWNpbmc6IC41
LCBkaXNwbGF5OiAiYmxvY2siLCBtYXJnaW5Cb3R0b206IDMgfTsKY29uc3QgdGJTZWwgPSB7IHdpZHRoOiAiMTAwJSIsIHBhZGRp
bmc6ICI1cHggNnB4IiwgYm9yZGVyUmFkaXVzOiA1LCBib3JkZXI6ICIxcHggc29saWQgIzMzNDE1NSIsIGJhY2tncm91bmQ6ICIj
MGYxNzJhIiwgY29sb3I6ICIjZTJlOGYwIiwgZm9udFNpemU6IDExLCBhcHBlYXJhbmNlOiAiYXV0byIgfTsKCmZ1bmN0aW9uIGFi
dG4oYmcpIHsKICByZXR1cm4ge2ZsZXg6MSxwYWRkaW5nOiI2cHggMCIsYm9yZGVyUmFkaXVzOjYsYm9yZGVyOiJub25lIixjdXJz
b3I6InBvaW50ZXIiLGJhY2tncm91bmQ6YmcsY29sb3I6IiNlMmU4ZjAiLGZvbnRXZWlnaHQ6NjAwLGZvbnRTaXplOjExLHRleHRB
bGlnbjoiY2VudGVyIn07Cn0KCgogIAoKZnVuY3Rpb24gUGFzc3dvcmRHYXRlKHsgb25VbmxvY2sgfSkgewogIGNvbnN0IFtwdywg
c2V0UHddID0gUmVhY3QudXNlU3RhdGUoIiIpOwogIGNvbnN0IFtlcnJvciwgc2V0RXJyb3JdID0gUmVhY3QudXNlU3RhdGUoZmFs
c2UpOwoKICBSZWFjdC51c2VFZmZlY3QoKCkgPT4gewogICAgY29uc3Qgc2F2ZWQgPSBzZXNzaW9uU3RvcmFnZS5nZXRJdGVtKCJu
c2ktYXV0aCIpOwogICAgaWYgKHNhdmVkID09PSAib2siKSBvblVubG9jaygpOwogIH0sIFtdKTsKCiAgY29uc3QgaGFuZGxlU3Vi
bWl0ID0gKCkgPT4gewogICAgaWYgKHB3ID09PSBhdG9iKCJUbGt5TURJMlZrRT0iKSkgewogICAgICBzZXNzaW9uU3RvcmFnZS5z
ZXRJdGVtKCJuc2ktYXV0aCIsICJvayIpOwogICAgICBvblVubG9jaygpOwogICAgfSBlbHNlIHsKICAgICAgc2V0RXJyb3IodHJ1
ZSk7CiAgICAgIHNldFRpbWVvdXQoKCkgPT4gc2V0RXJyb3IoZmFsc2UpLCAyMDAwKTsKICAgIH0KICB9OwoKICByZXR1cm4gKAog
ICAgPGRpdiBzdHlsZT17e2Rpc3BsYXk6ImZsZXgiLGFsaWduSXRlbXM6ImNlbnRlciIsanVzdGlmeUNvbnRlbnQ6ImNlbnRlciIs
aGVpZ2h0OiIxMDB2aCIsYmFja2dyb3VuZDoiIzBmMTcyYSIsZm9udEZhbWlseToiJ1NlZ29lIFVJJyxzeXN0ZW0tdWksc2Fucy1z
ZXJpZiIscGFkZGluZzoyMH19PgogICAgICA8ZGl2IHN0eWxlPXt7bWF4V2lkdGg6NDAwLHdpZHRoOiIxMDAlIix0ZXh0QWxpZ246
ImNlbnRlciJ9fT4KICAgICAgICA8ZGl2IHN0eWxlPXt7Zm9udFNpemU6NDgsbWFyZ2luQm90dG9tOjEyfX0+8J+UkjwvZGl2Pgog
ICAgICAgIDxoMSBzdHlsZT17e2NvbG9yOiIjZjhmYWZjIixmb250U2l6ZToyMCxmb250V2VpZ2h0OjcwMCxtYXJnaW46IjAgMCA0
cHgifX0+TlNJIEZpZWxkIFN1cnZleSBUb29sPC9oMT4KICAgICAgICA8cCBzdHlsZT17e2NvbG9yOiIjNjQ3NDhiIixmb250U2l6
ZToxMyxtYXJnaW5Cb3R0b206MjR9fT5FbnRlciBwYXNzd29yZCB0byBjb250aW51ZTwvcD4KICAgICAgICA8ZGl2IHN0eWxlPXt7
ZGlzcGxheToiZmxleCIsZ2FwOjh9fT4KICAgICAgICAgIDxpbnB1dCB0eXBlPSJwYXNzd29yZCIgdmFsdWU9e3B3fQogICAgICAg
ICAgICBvbkNoYW5nZT17ZT0+e3NldFB3KGUudGFyZ2V0LnZhbHVlKTtzZXRFcnJvcihmYWxzZSk7fX0KICAgICAgICAgICAgb25L
ZXlEb3duPXtlPT5lLmtleT09PSJFbnRlciImJmhhbmRsZVN1Ym1pdCgpfQogICAgICAgICAgICBwbGFjZWhvbGRlcj0iUGFzc3dv
cmQiIGF1dG9Gb2N1cwogICAgICAgICAgICBzdHlsZT17e2ZsZXg6MSxwYWRkaW5nOiIxMnB4IDE2cHgiLGJvcmRlclJhZGl1czo4
LGJvcmRlcjplcnJvcj8iMnB4IHNvbGlkICNkYzI2MjYiOiIxcHggc29saWQgIzMzNDE1NSIsYmFja2dyb3VuZDoiIzFlMjkzYiIs
Y29sb3I6IiNlMmU4ZjAiLGZvbnRTaXplOjE1LG91dGxpbmU6Im5vbmUiLGJveFNpemluZzoiYm9yZGVyLWJveCJ9fQogICAgICAg
ICAgLz4KICAgICAgICAgIDxidXR0b24gb25DbGljaz17aGFuZGxlU3VibWl0fSBzdHlsZT17e3BhZGRpbmc6IjEycHggMjRweCIs
Ym9yZGVyUmFkaXVzOjgsYm9yZGVyOiJub25lIixjdXJzb3I6InBvaW50ZXIiLGJhY2tncm91bmQ6IiMzYjgyZjYiLGNvbG9yOiIj
ZmZmIixmb250V2VpZ2h0OjcwMCxmb250U2l6ZToxNH19PkdvPC9idXR0b24+CiAgICAgICAgPC9kaXY+CiAgICAgICAge2Vycm9y
ICYmIDxkaXYgc3R5bGU9e3ttYXJnaW5Ub3A6MTAsY29sb3I6IiNmODcxNzEiLGZvbnRTaXplOjEzLGZvbnRXZWlnaHQ6NjAwfX0+
SW5jb3JyZWN0IHBhc3N3b3JkPC9kaXY+fQogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogICk7Cn0KCmZ1bmN0aW9uIFJvb3QoKSB7
CiAgLy8gUGFzc3dvcmQgZ2F0ZSByZW1vdmVkIGZvciB0aGUgZW1iZWRkZWQgQURBUFQgdGFiOiBhcHAxIGFscmVhZHkKICAvLyBh
dXRoZW50aWNhdGVzIHRoZSB1c2VyIGJlZm9yZSB0aGlzIGNvbXBvbmVudCBpcyBldmVyIHJlbmRlcmVkLgogIC8vIFRvIHJlc3Rv
cmUgdGhlIHN0YW5kYWxvbmUgZ2F0ZSwgcmV2ZXJ0IHRvIHRoZSBjb21tZW50ZWQgdmVyc2lvbiBiZWxvdy4KICByZXR1cm4gPEFw
cCAvPjsKICAvLyBjb25zdCBbYXV0aGVkLCBzZXRBdXRoZWRdID0gUmVhY3QudXNlU3RhdGUoZmFsc2UpOwogIC8vIGlmICghYXV0
aGVkKSByZXR1cm4gPFBhc3N3b3JkR2F0ZSBvblVubG9jaz17KCkgPT4gc2V0QXV0aGVkKHRydWUpfSAvPjsKICAvLyByZXR1cm4g
PEFwcCAvPjsKfQoKY29uc3Qgcm9vdCA9IFJlYWN0RE9NLmNyZWF0ZVJvb3QoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoInJvb3Qi
KSk7CnJvb3QucmVuZGVyKDxSb290IC8+KTsKCiAgPC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPg==
"""


def load_nsi_tool_html():
    """Return the NSI Field Survey tool HTML for the 'NSI dataset' tab.

    NOTE: intentionally NOT cached with st.cache_data. That cache keys on the
    function's code, not on the embedded _NSI_TOOL_HTML_B64 constant, so a
    cached result would keep serving an old build whenever only the constant
    changes. Decoding ~100 KB of base64 per run is negligible.

    Prefers an on-disk `nsi_tool.html` next to app.py (easy override for
    editing); otherwise decodes the embedded base64 copy, so the app runs as
    a single self-contained file with no external asset. b64decode ignores
    the newlines used to wrap the constant."""
    import base64
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'nsi_tool.html'),
                 os.path.join('.', 'nsi_tool.html')):
        if os.path.exists(cand):
            with open(cand, 'r', encoding='utf-8') as fh:
                return fh.read()
    return base64.b64decode(_NSI_TOOL_HTML_B64).decode('utf-8')


# ----------------------------------------------------------------------------
# CSV-bundle loader (current ADAPT data format)
# ----------------------------------------------------------------------------
# The upstream pipeline ships a per-location bundle of seven files:
#
#   {LOCATION}_metadata.csv                            - key/value metadata
#   {LOCATION}_bldg_lookup.csv                         - analysis-ready bldg attrs
#   {LOCATION}_bldg_CumulativeDamage.csv               - bldg × (year, action, slr) × pcts
#   {LOCATION}_CumulativeDamage_categories.csv         - 4 leaf categories × ... × pcts
#   {LOCATION}_skipped_buildings.csv                   - provenance log (optional)
#   DDD___{LOCATION}___NSI.xlsx                        - full NSI descriptors
#   DDD___{LOCATION}_MC_annual_max_waterlevels_P50.csv - Year × MC_0001..MC_1000
#   DDD___{LOCATION}_MC_annual_max_waterlevels_P90.csv - same for high-end SLR
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
#     runtime - community aggregates are now built from the per-building
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
    """Combine bldg_lookup (analysis-ready) with NSI (inventory source of truth).

    The NSI file is the maintained building inventory, so it is authoritative
    for every attribute it carries - structure/content value, occupancy,
    ground elevation, lon/lat, and the descriptive fields (building_type,
    number_of_stories, area, foundation_type, foundation_height, year_built,
    address). For those, the NSI value is used where present and the lookup
    fills only the gaps. Fields the NSI file does NOT carry but that drove the
    damage calculation - FFE_ft, DFE_Status, SOID, occupancy_group - always
    come from the lookup.
    """
    # 1. Map the analysis lookup to the app's lowercase schema. FFE_ft,
    #    DFE_Status, and SOID keep their original case because the UI looks
    #    them up by those exact names.
    rename = {
        'BuildingID':         'id',
        'OccupancyType':      'occupancy_type',
        'OccupancyGroup':     'occupancy_group',
        'StructureValue':     'structure_value',
        'ContentValue':       'content_value',
        'GroundElevation_ft': 'ground_elevation',
        'Longitude':          'longitude',
        'Latitude':           'latitude',
        # Legacy bundles ship 'Floodplain_Status' ('In/Out of floodplain');
        # canonical bundles ship 'DFE_Status' ('Above/Under DFE'). Accept the
        # legacy name and normalize values via convert_floodplain_status below.
        'Floodplain_Status':  'DFE_Status',
    }
    out = lookup_df.rename(columns=rename).copy()

    # 2. NSI already uses the lowercase schema; align it by id and let it win.
    nsi = nsi_df.rename(columns={'ID': 'id'}).copy()
    if 'id' in nsi.columns and 'id' in out.columns:
        out = out.set_index('id')
        nsi = nsi.set_index('id')
        nsi = nsi[~nsi.index.duplicated(keep='first')]
        # 3. Overlay every NSI-supplied column: the NSI value wins where it is
        #    present (non-null), the lookup fills the gaps, and NSI-only
        #    columns (building_type, stories, area, …) are added outright.
        for col in nsi.columns:
            nsi_col = nsi[col].reindex(out.index)
            if col in out.columns:
                out[col] = nsi_col.combine_first(out[col])
            else:
                out[col] = nsi_col
        out = out.reset_index()

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
# FLOOD-MAP (BATHTUB) SUPPORT - inlined so the app is a single file.
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
# Blue-only ramp (light = shallow → dark = deep) for the road-flooding maps.
WS_COLORS_BLUE = [
    (198, 219, 239), (158, 202, 225), (107, 174, 214),
    (66, 146, 198), (33, 113, 181), (8, 69, 148),
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


def depth_to_rgba(depth_ft, alpha=217, colors=None):
    """Depth-in-feet array -> (H,W,4) uint8 RGBA (discrete bins; dry transparent).
    `colors` defaults to the multi-hue ramp; pass WS_COLORS_BLUE for blue-only."""
    cols = colors if colors is not None else WS_COLORS
    h, w = depth_ft.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    n = len(cols)
    for b, (r, g, bl) in enumerate(cols):
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
    overlay = Image.fromarray(depth_to_rgba(depth_ft, alpha=150), "RGBA").resize(base.size, Image.NEAREST)
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


def classify_roads_access(Zm, ext, roads, wl_ft, sample_m=8.0,
                          source="boundary", snap_dp=6, boundary_tol_m=35.0,
                          entry_points=None, entrance_reach_m=300.0):
    """Topological road-accessibility classifier (replaces the proximity buffer).

    A road is classified by whether you can still REACH it on dry roads, not by
    how close it sits to water. Status codes match classify_roads so the same
    renderer/legend plumbing works:

        0 = dry & accessible   (a dry path to the outside world still exists)
        1 = dry & INACCESSIBLE (dry itself, but every dry route out is severed
                                by flooding - e.g. both ends flooded, or one end
                                flooded and the other a dead-end, or stranded
                                behind a flooded road as part of a cut-off cluster)
        2 = flooded            (the road surface is below the water level)

    Method. OSM ways carry their topology as *shared vertex coordinates* at
    intersections (and bridges/overpasses share no node, so they don't connect).
    We rebuild the network graph by snapping coordinates, mark each span dry or
    flooded against the same bathtub mask as the Flood Maps tab, then ask a pure
    reachability question:

        a dry span is INACCESSIBLE  <=>  it is reachable from a SOURCE in the
        full network (flooded roads treated as passable) but NOT reachable once
        flooded roads are removed.

    The "full-network" baseline is the causality guard: it ensures we only blame
    flooding for what flooding actually severed, never roads that were already
    isolated in the raw OSM data (those are returned as status 0 and tallied in
    counts['predisc']).

    source: 'boundary' (default) treats roads leaving the map as exits to the
    wider world - the most defensible anchor when the map is a sub-area of a
    larger network. 'largest' instead treats the biggest dry component as the
    mainland; used automatically as a fallback if no boundary exits are found.

    entry_points: optional list of (lon, lat) gateways for a designated main
    entrance that should stay open even when the entrance road itself floods
    (e.g. a low causeway that is the site's only access and floods first). For
    each gateway we take the nearest network node, walk the CONTIGUOUS flooded
    blob outward from it up to entrance_reach_m (network metres) - capturing the
    flooded entrance and its flooded gateway intersection but not a separate
    interior flood - reconnect that blob, and treat the gateway as connected to
    the outside. The entrance road still renders red where flooded; it just no
    longer severs the whole site. Interior floods that form their own separate
    blobs keep stranding roads normally, so this never over-rescues.

    Returns (segments, counts) where segments is a list of
    ((lon0,lat0),(lon1,lat1),status), drawn at the same ~sample_m resolution as
    classify_roads so the map detail is unchanged.
    """
    from collections import defaultdict

    lon_min, lat_min, lon_max, lat_max = ext
    nlat, nlon = Zm.shape
    wl_m = float(wl_ft) * 0.3048
    Zmask = np.where(np.isnan(Zm), 9999.0, Zm)
    flood = (wl_m - Zmask > 0) & (Zmask > 0)          # same flood mask as the flood-map tab

    def flooded_fn(lons, lats):
        # nearest-cell lookup, matching classify_roads' to_rc()
        c = np.clip(((np.asarray(lons, float) - lon_min) / max(lon_max - lon_min, 1e-9)
                     * (nlon - 1)).round().astype(int), 0, nlon - 1)
        r = np.clip(((lat_max - np.asarray(lats, float)) / max(lat_max - lat_min, 1e-9)
                     * (nlat - 1)).round().astype(int), 0, nlat - 1)
        return flood[r, c]

    tol_lat = boundary_tol_m / 111320.0
    midlat = 0.5 * (lat_min + lat_max)
    tol_lon = boundary_tol_m / (111320.0 * max(math.cos(math.radians(midlat)), 1e-6))

    def nkey(lon, lat):
        return (round(lon, snap_dp), round(lat, snap_dp))

    def is_boundary(lon, lat):
        on_lr = ((abs(lon - lon_min) <= tol_lon or abs(lon - lon_max) <= tol_lon)
                 and (lat_min - tol_lat <= lat <= lat_max + tol_lat))
        on_tb = ((abs(lat - lat_min) <= tol_lat or abs(lat - lat_max) <= tol_lat)
                 and (lon_min - tol_lon <= lon <= lon_max + tol_lon))
        return on_lr or on_tb

    def densify(coords):
        """Sample ~every sample_m along a way, forcing the original OSM vertices in
        (so intersections snap) and reporting which samples are real vertices."""
        lons, lats = coords[:, 0], coords[:, 1]
        meanlat = float(np.mean(lats))
        dlat = np.diff(lats) * 111320.0
        dlon = np.diff(lons) * 111320.0 * math.cos(math.radians(meanlat))
        cum = np.concatenate([[0.0], np.cumsum(np.sqrt(dlat ** 2 + dlon ** 2))])
        total = float(cum[-1])
        if total < 1.0:
            return None
        fill = np.linspace(0.0, total, max(int(total / sample_m), 1) + 1)
        d = np.unique(np.concatenate([cum, fill]))
        slon = np.interp(d, cum, lons)
        slat = np.interp(d, cum, lats)
        return slon, slat, np.isin(d, cum)

    dry_edges = []          # (u, v, [subsegs])
    all_edges = []          # (u, v) for the baseline (flood-as-passable) graph
    flooded_subsegs = []
    flooded_uv = []         # (u, v) node keys of flooded spans, for the entrance walk
    node_xy = {}

    for rd in roads:
        coords = np.asarray(rd["coords"], dtype=float)
        if len(coords) < 2:
            continue
        if not np.any((coords[:, 0] >= lon_min) & (coords[:, 0] <= lon_max) &
                      (coords[:, 1] >= lat_min) & (coords[:, 1] <= lat_max)):
            continue
        dz = densify(coords)
        if dz is None:
            continue
        slon, slat, is_vtx = dz
        fl = np.asarray(flooded_fn(slon, slat), dtype=bool)
        n = len(slon)
        for i in range(n):
            node_xy.setdefault(nkey(slon[i], slat[i]), (slon[i], slat[i]))
        brk = np.zeros(n, dtype=bool)
        brk[0] = brk[-1] = True
        brk[is_vtx] = True                       # split at intersections
        trans = np.where(fl[:-1] != fl[1:])[0]   # split at dry/flood transitions
        brk[trans] = True
        brk[trans + 1] = True
        bidx = np.where(brk)[0]
        for a, b in zip(bidx[:-1], bidx[1:]):
            u = nkey(slon[a], slat[a]); v = nkey(slon[b], slat[b])
            if u == v:
                continue
            span_flooded = bool(fl[a:b + 1].any())
            subs = [((float(slon[j]), float(slat[j])),
                     (float(slon[j + 1]), float(slat[j + 1]))) for j in range(a, b)]
            all_edges.append((u, v))
            if span_flooded:
                flooded_subsegs.extend(subs)
                flooded_uv.append((u, v))
            else:
                dry_edges.append((u, v, subs))

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    parent_all = {}

    def find_a(x):
        parent_all.setdefault(x, x)
        r = x
        while parent_all[r] != r:
            r = parent_all[r]
        while parent_all[x] != r:
            parent_all[x], x = r, parent_all[x]
        return r

    def union_a(a, b):
        ra, rb = find_a(a), find_a(b)
        if ra != rb:
            parent_all[ra] = rb

    for u, v, _ in dry_edges:
        union(u, v)
    for u, v in all_edges:
        union_a(u, v)

    boundary_nodes = [k for k, (lo, la) in node_xy.items() if is_boundary(lo, la)]
    used_source = source
    if source == "boundary" and boundary_nodes:
        dry_src_roots = set(find(k) for k in boundary_nodes)
        all_src_roots = set(find_a(k) for k in boundary_nodes)
    else:
        used_source = "largest" if source == "boundary" else source
        comp_size = defaultdict(int)
        for u, v, _ in dry_edges:
            comp_size[find(u)] += 1
        main = max(comp_size, key=comp_size.get) if comp_size else None
        dry_src_roots = {main} if main is not None else set()
        all_src_roots = set()
        if main is not None:
            for u, v, _ in dry_edges:
                if find(u) == main:
                    all_src_roots.add(find_a(u)); break

    # Designated main-entrance exemption. Access can come through the entrance road
    # and its (often flooded) gateway intersection, so for each gateway we take the
    # nearest node, walk the contiguous flooded blob outward up to entrance_reach_m
    # (network metres) - capturing the flooded entrance + flooded gateway but not a
    # separate interior flood - reconnect that blob, and make the gateway a source.
    n_entry = 0
    if entry_points:
        import heapq
        cosl = math.cos(math.radians(midlat))

        def _mlen(a, b):
            return math.hypot((node_xy[a][0] - node_xy[b][0]) * 111320.0 * cosl,
                              (node_xy[a][1] - node_xy[b][1]) * 111320.0)

        fl_adj = defaultdict(list)
        for u, v in flooded_uv:
            L = _mlen(u, v)
            fl_adj[u].append((v, L)); fl_adj[v].append((u, L))
        for ep in entry_points:
            if not node_xy:
                break
            G = min(node_xy, key=lambda k: ((node_xy[k][0] - ep[0]) * cosl) ** 2
                    + (node_xy[k][1] - ep[1]) ** 2)
            reached = {G: 0.0}; pq = [(0.0, G)]
            while pq:
                d, nn = heapq.heappop(pq)
                if d > reached.get(nn, 1e18):
                    continue
                for w, L in fl_adj[nn]:
                    nd = d + L
                    if nd <= entrance_reach_m and nd < reached.get(w, 1e18):
                        reached[w] = nd; heapq.heappush(pq, (nd, w))
            for nn in reached:
                union(G, nn); union_a(G, nn)
            dry_src_roots.add(find(G)); all_src_roots.add(find_a(G)); n_entry += 1

    segs = []
    nf = na = ni = npd = 0
    for s in flooded_subsegs:
        segs.append((s[0], s[1], 2)); nf += 1
    for u, v, subs in dry_edges:
        if find(u) in dry_src_roots:
            status = 0; na += len(subs)
        elif find_a(u) in all_src_roots:
            status = 1; ni += len(subs)
        else:
            status = 0; npd += len(subs)        # isolated in raw data; not flood-caused
        for s in subs:
            segs.append((s[0], s[1], status))

    tot = nf + na + ni + npd
    counts = {
        "flood": nf, "inacc": ni, "dry": na, "predisc": npd, "total": tot,
        "pct_flood": 100.0 * nf / tot if tot else 0.0,
        "pct_inacc": 100.0 * ni / tot if tot else 0.0,
        "pct_dry": 100.0 * (na + npd) / tot if tot else 0.0,
        "source_used": used_source, "n_boundary_nodes": len(boundary_nodes),
        "n_entry": n_entry,
        # back-compat alias so any caller reading pct_prox still works
        "prox": ni, "pct_prox": 100.0 * ni / tot if tot else 0.0,
    }
    return segs, counts


def _entrance_nodes_edges(roads, snap_dp=6):
    """Flood-agnostic road graph for entrance detection: node_key -> (lon,lat),
    adjacency, a per-node 'on a main road' flag (from OSM highway class), and
    node degree. Main roads are the through-classes (primary/secondary/...)."""
    from collections import defaultdict
    MAIN = {"motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}

    def nk(lo, la):
        return (round(lo, snap_dp), round(la, snap_dp))

    adj = defaultdict(set); xy = {}; ismain = defaultdict(bool)
    for rd in roads:
        c = np.asarray(rd["coords"], dtype=float)
        if len(c) < 2:
            continue
        is_main = rd.get("highway", "") in MAIN
        for i in range(len(c) - 1):
            u = nk(*c[i]); v = nk(*c[i + 1])
            xy[u] = (float(c[i][0]), float(c[i][1]))
            xy[v] = (float(c[i + 1][0]), float(c[i + 1][1]))
            if u != v:
                adj[u].add(v); adj[v].add(u)
                if is_main:
                    ismain[u] = True; ismain[v] = True
    deg = {n: len(adj[n]) for n in adj}
    return xy, adj, ismain, deg


def detect_main_entrance(roads, ext, site_lonlat, snap_dp=6, boundary_tol_m=40.0):
    """Auto-detect the site's main entrance gateway: the site-side end of the road
    that connects the building cluster to the OUTSIDE WORLD. Returns [(lon, lat)] to
    pass as entry_points, or [] if it can't be determined.

    The connection to the outside is anchored on the building cluster's OWN
    component: a classified main road if one is present in that component, else the
    component's boundary exits (roads leaving the map). This is deliberately NOT
    anchored on the global classified main road - at sites like Pamunkey the
    reservation network is a separate OSM component whose nearest classified road
    (King William Rd) is ~1 km away and never shares a node, so a main-road anchor
    would find no path and return nothing. The gateway is the first junction
    (degree >= 3) reached after leaving the anchor on the path toward the cluster,
    i.e. where the single entrance corridor meets the local network - placing the
    exemption at the entrance, not deep inside the site."""
    from collections import deque
    xy, adj, ismain, deg = _entrance_nodes_edges(roads, snap_dp)
    if not xy:
        return []
    lon_min, lat_min, lon_max, lat_max = ext
    cosl = math.cos(math.radians(0.5 * (lat_min + lat_max)))
    tol = boundary_tol_m / 111320.0
    tolx = boundary_tol_m / (111320.0 * max(cosl, 1e-6))

    def d2(a, b):
        return ((a[0] - b[0]) * cosl) ** 2 + (a[1] - b[1]) ** 2

    def is_boundary(lo, la):
        return ((abs(lo - lon_min) <= tolx or abs(lo - lon_max) <= tolx)
                and lat_min - tol <= la <= lat_max + tol) or \
               ((abs(la - lat_min) <= tol or abs(la - lat_max) <= tol)
                and lon_min - tolx <= lo <= lon_max + tolx)

    # nearest node to the building centroid, and the component it belongs to
    N_site = min(xy, key=lambda n: d2(xy[n], site_lonlat))
    comp = {N_site}; dq = deque([N_site])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w not in comp:
                comp.add(w); dq.append(w)

    # anchors = how THIS component reaches the outside: a main road within the
    # component if present, else the component's boundary exits.
    main_in = [n for n in comp if ismain[n]]
    bexits = [n for n in comp if is_boundary(*xy[n])]
    anchors = main_in if main_in else bexits
    if not anchors:                              # component has no outside link
        anchors = ([n for n in xy if ismain[n]]
                   or [n for n, (lo, la) in xy.items() if is_boundary(lo, la)])
    if not anchors:
        return []
    anchors = set(anchors)

    pred = {}; seen = set(anchors); dq = deque(anchors)
    for a in anchors:
        pred[a] = None
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w not in seen:
                seen.add(w); pred[w] = u; dq.append(w)
    if N_site not in seen:
        return []
    path = []; x = N_site
    while x is not None:
        path.append(x); x = pred[x]
    path.reverse()                               # anchor -> site
    gateway = None
    for n in path:
        if n in anchors:
            continue
        if deg.get(n, 0) >= 3:
            gateway = n; break
    if gateway is None:
        cands = [n for n in path if n not in anchors]
        gateway = min(cands, key=lambda n: d2(xy[n], site_lonlat)) if cands else N_site
    return [xy[gateway]]


def compose_road_png(basemap_rgb, depth_ft, segments, ext):
    """Basemap + flood-depth overlay + colored road segments -> PNG bytes.
    Segment colors: green dry/accessible, violet inaccessible (cut off by
    flooding), red flooded."""
    import io as _io
    from PIL import Image, ImageDraw
    base = Image.fromarray(np.asarray(basemap_rgb, dtype=np.uint8), "RGB").convert("RGBA")
    W, H = base.size
    overlay = Image.fromarray(depth_to_rgba(depth_ft, alpha=150, colors=WS_COLORS_BLUE), "RGBA").resize((W, H), Image.NEAREST)
    base = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(base)
    lon_min, lat_min, lon_max, lat_max = ext

    def px(lon, lat):
        x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * (W - 1)
        y = (lat_max - lat) / max(lat_max - lat_min, 1e-9) * (H - 1)
        return (x, y)

    colors = {0: (34, 139, 34, 235), 1: (138, 43, 226, 245), 2: (220, 20, 20, 250)}
    base_w = max(2, int(round(W / 450)))
    widths = {0: base_w, 1: base_w + 1, 2: base_w + 2}
    for status in (0, 1, 2):                       # dry, then inaccessible, then flooded on top
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
    classify_roads_access=classify_roads_access,
    detect_main_entrance=detect_main_entrance,
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
def load_bundle(data_folder, location_slug, file_sig=None):
    """Load a single-location CSV bundle and return the data_store entry.

    `file_sig` is a fingerprint (per-file mtime+size) of the bundle's files.
    It is unused in the body but participates in the cache key, so editing
    any bundle file (e.g. the NSI xlsx) invalidates the cache and the bundle
    is re-read on the next run - no manual "Clear cache" needed.
    """
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

    # 2. NSI (271 rows incl. the building that gets skipped - joined later)
    nsi = pd.read_excel(join(f'DDD___{location_slug}___NSI.xlsx'))

    # 3. Building lookup (270 rows - only buildings that were actually analyzed)
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
    # sets `Dmg(:,:,4,si) = dmg_baseline` - i.e. Elevate equals No mitigation
    # exactly, by construction. The DFE_Status flag is itself derived
    # from the building's first-floor elevation (FFE) being above the design
    # flood elevation (DFE = BFE + 2 ft), but FFE = ground + foundation height,
    # so a building can sit on low ground (yard/garage flood frequently) and
    # still be tagged Above-DFE because its raised foundation lifts
    # the FFE above the DFE. The combined effect: a non-trivial number of
    # Above-DFE buildings have substantial baseline damage AND a no-op'd Elevate
    # column, which makes them look like "Residual: even elevation can't help"
    # on the Adaptation Effectiveness map - when in reality elevation simply
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
        # equality across ALL percentile columns simultaneously - anything
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
        # Also require at least one column to be non-zero - a real
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
    # trajectories, and every aggregate downstream - one filter, one
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

    # Manufactured housing (RES2) sits on piers: no basement, no conditioned
    # upper floor to relocate utilities to, and no separable 1st-floor
    # envelope to dry/wet-floodproof. The only physically meaningful retrofit
    # is raising (elevating) the whole home, so drop Raise Utilities, WFP
    # Basement, and WFP 1st Floor for RES2 buildings. This makes those actions
    # vanish from the hover, the map (including the Adaptation-Effectiveness
    # yellow "cheap retrofit" bucket), the distributions, and every aggregate
    # - a RES2 home can only ever be classified under Elevate.
    if 'occupancy_type' in df_buildings.columns:
        _is_res2 = (df_buildings['occupancy_type']
                    .fillna('').astype(str).str.upper().str.startswith('RES2'))
        _keep &= ~(_is_res2 & _action.isin(['Raise Utilities', 'WFP B', 'WFP 1st']))

    # Manufactured-housing-dominant inventory (e.g., Pamunkey): if basements
    # are essentially absent, drop Wet-Floodproof-Basement for the whole
    # location so it never surfaces as an option anywhere.
    _n_bldg_total = df_buildings['id'].nunique()
    _n_basement_bldg = df_buildings.loc[_is_basement, 'id'].nunique()
    _basement_share = (_n_basement_bldg / _n_bldg_total) if _n_bldg_total else 0.0
    if _basement_share < 0.10:
        _keep &= ~(_action == 'WFP B')

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
    # rows zeroed out - which makes the Summary metric read $0 while the
    # retrofit rows still hold real values, and the "Damage Reduction"
    # chart then plots NEGATIVE reductions because it computes
    # `baseline − retrofit = 0 − $235M`. To keep the Summary, the box-
    # plot, and the Map/Details visualizations all consistent, we now
    # build the aggregate **directly from the per-building damage table**
    # (the same source the per-building visuals read from). The
    # statistical concession is that we use sum-of-percentiles rather
    # than percentile-of-sum for the tails (P05/P95), which slightly
    # overstates community tails - but consistency is far more important
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
    # is now canonical - the comparison just helps when debugging
    # data-pipeline issues for a new location.
    cat_csv_path = join(f'{location_slug}_CumulativeDamage_categories.csv')
    cat = None
    try:
        cat = pd.read_csv(cat_csv_path)
        cat['TargetYear'] = _bundle_normalize_target_year(cat['TargetYear'])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    # 7. Water levels - load raw MC + pre-compute percentile shim
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
    # only trust the result when it actually inserted a space - otherwise
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
    file are skipped silently - we don't claim partial bundles.
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
            # Fingerprint every bundle file (mtime + size) so an edit to any of
            # them - e.g. correcting a structure value in the NSI xlsx -
            # invalidates load_bundle's cache and forces a fresh read.
            _sig = tuple(
                (os.path.getmtime(p), os.path.getsize(p))
                for p in [meta_path] + [os.path.join(data_folder, r) for r in required]
            )
        except OSError:
            _sig = None
        try:
            entry = load_bundle(data_folder, slug, file_sig=_sig)
        except Exception as e:
            # Don't kill the app on a malformed bundle - log and skip.
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
    the bins themselves - which makes year-to-year comparisons meaningful.
    
    The upper-tail percentile used here is **P90**, matching the workshop
    convention and the Distributions tab's "Building Counts by Adaptation
    Effectiveness" classifier - so the same building gets the same
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


@st.cache_data(show_spinner=False)
def compute_flood_occurrences(_mc_df, _ffe_by_id, sig):
    """Per-building flood-occurrence counts from the MC water-level ensemble.

    For each Monte-Carlo realization (one of the up-to-1,000 `MC_*` columns)
    we count, across every year from the first MC year through the selected
    horizon, how many years the simulated annual-maximum water level exceeds
    the building's first-floor elevation (FFE). That yields one occurrence
    count per MC column per building - up to 1,000 numbers - and we then
    report percentiles of that distribution (P10 / P25 / P50 / P75 / P90),
    mirroring the percentile reporting used throughout the tool.

    Parameters
    ----------
    _mc_df : DataFrame
        Raw MC sheet for one SLR scenario: a 'Year' column plus MC_0001..
        Underscore-prefixed so Streamlit does NOT hash it (it's large).
    _ffe_by_id : dict[int, float | None]
        First-floor elevation (ft NAVD88) per building id. Underscore-
        prefixed for the same reason.
    sig : tuple
        (location_name, scenario, horizon_year) - the ONLY hashed argument.
        It uniquely determines `_mc_df` and `_ffe_by_id`, so the cache key is
        correct even though the big inputs are skipped. Keep this computed
        over ALL location buildings (not the occupancy/DFE-filtered subset)
        so the cache isn't poisoned by a narrower selection.

    Returns
    -------
    DataFrame with columns id, occ_P10, occ_P25, occ_P50, occ_P75, occ_P90,
    occ_mean, n_years, horizon - or None when there's nothing to compute.
    """
    location_name, scenario, horizon_year = sig
    mc_cols = [c for c in _mc_df.columns if c.startswith('MC_')]
    sub = _mc_df[_mc_df['Year'] <= int(horizon_year)]
    if sub.empty or not mc_cols:
        return None
    M = sub[mc_cols].to_numpy(dtype=float)            # (n_years, n_mc)
    n_years, n_mc = M.shape

    ids = [int(i) for i, f in _ffe_by_id.items()
           if f is not None and np.isfinite(f)]
    if not ids:
        return None
    ffe = np.array([float(_ffe_by_id[i]) for i in ids], dtype=float)   # (N,)

    # A building with FFE f floods in (year, realization) iff WL > f. For a
    # fixed realization column, the occurrence count as a function of FFE is a
    # step function, so we sort the column's water levels once and resolve all
    # buildings with a single searchsorted:
    #     count(WL > f) = n_years - searchsorted(sorted_col, f, side='right')
    # 'right' makes the comparison strictly greater-than (water above the
    # first floor), matching the building-depth convention elsewhere.
    occ = np.empty((len(ids), n_mc), dtype=np.int32)
    for j in range(n_mc):
        col_sorted = np.sort(M[:, j])
        occ[:, j] = n_years - np.searchsorted(col_sorted, ffe, side='right')

    # Nearest-rank (integer) percentiles: the occurrence count is an integer
    # number of years per realization, so we report an actual realized count
    # ("in the median realization it floods N years") rather than a linearly
    # interpolated fraction. Version-robust across numpy's method/interpolation
    # keyword rename.
    def _pctile_int(arr, p):
        try:
            return np.percentile(arr, p, axis=1, method='nearest')
        except TypeError:
            return np.percentile(arr, p, axis=1, interpolation='nearest')

    out = pd.DataFrame({'id': ids})
    for p in (10, 25, 50, 75, 90):
        out[f'occ_P{p:02d}'] = _pctile_int(occ, p).astype(int)
    out['occ_mean'] = occ.mean(axis=1)
    out['n_years']  = int(n_years)
    out['horizon']  = int(horizon_year)
    return out


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


# ESRI World Imagery raster tile source - overlaid on a white-bg base
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
        The per-building map dataframe - used to recompute the bbox-based
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
        three are optional and independent - if only some are passed,
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

    # Tick whisker height - fixed pixels, converted to paper-y units
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
    # Mapbox-layer scale bar removed - keeping mapbox.layers empty so the
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
    title_main = f"Building-level Flood Risk - {location}"
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
        ("ADAPT - Assessment of Damage and Adaptation Planning Tool   |   "
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
        # Scale-bar label - centered above the bar at its midpoint
        dict(text=f"<b>{scale_label}</b>",
             xref="paper", yref="paper",
             x=(sb_x0_paper + sb_x1_paper) / 2,
             y=sb_y_paper + tick_h_paper_y * 4,
             xanchor="center", yanchor="bottom",
             showarrow=False,
             font=dict(family="Arial", size=14, color="#0f172a")),
        # North arrow - paper-coord glyph in the upper-right corner
        dict(text="<b>N</b><br>▲",
             xref="paper", yref="paper",
             x=0.96, y=0.96, xanchor="center", yanchor="top",
             showarrow=False,
             align="center",
             bgcolor="rgba(255,255,255,0.9)",
             bordercolor="rgba(0,0,0,0.3)", borderwidth=1, borderpad=6,
             font=dict(family="Arial Black", size=18, color="#0f172a")),
    ]

    # Existing legend - move it to a less-cluttered corner if possible,
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
                            lower_pct=0.05, upper_pct=0.95,
                            median_label_shift=0.0):
    """Construct a Plotly box-and-whisker panel with grouped pairs of boxes.

    Parameters
    ----------
    group_labels : list[str]
        X-axis category labels (one per group, e.g. one per strategy).
    scenario_data : dict[slr_key -> list[tuple or None]]
        Per-group statistics for each SLR scenario. Use ``None`` for
        missing groups. Each tuple is one of:
          * ``(p05, p50, p95)`` - 3-tuple. Q1/Q3 are estimated by
            linear-CDF interpolation between the supplied bounds.
          * ``(p05, p25, p50, p75, p95)`` - 5-tuple. Q1 and Q3 are taken
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
            # Two accepted shapes - see the docstring above.
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
                # Unrecognized shape - skip rather than crash.
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
            x=x_num,                        # NUMERIC x - exact positions
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
        # missing - but the message we want is "this strategy reduces damage
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
    # Median-label vertical offset. 0 (default) sits the label right at the
    # median line; positive lifts it above the line, negative pushes it below.
    # Expressed as a fraction of the y-axis span so it scales with the plot.
    med_label_gap = float(median_label_shift) * y_span

    # Annotations: every label sits directly above its reference line -
    #   * Median label sits at the median line by default (yanchor='bottom'),
    #     offset up/down by med_label_gap (driven by median_label_shift).
    #   * P95 label sits just above the upper whisker
    # The lower whisker (P05) is intentionally NOT labeled.
    # X position is the exact numeric center of each box, so labels never
    # drift away from their own box regardless of plot width.
    for x_center, p05_v, p50_v, p95_v, line_clr in annot_records:
        med_text = fmt_money_rounded(p50_v) if abs(p50_v) >= label_zero_thresh else "$0"
        # Median - bold colored text on a semi-transparent white pill,
        # positioned relative to the median line by median_label_shift.
        fig.add_annotation(
            x=x_center, y=p50_v + med_label_gap,
            text=f"<b>{med_text}</b>",
            showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color=line_clr),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=2, bordercolor='rgba(0,0,0,0)',
        )
        if abs(p95_v) >= label_zero_thresh:
            # P95 - bold colored text just above the upper whisker
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
    # NAVIGATION RAIL  +  GLOBAL FILTER STATE  +  PER-PAGE INLINE SETTINGS
    # ------------------------------------------------------------------------
    # Right-hand vertical nav. There is no separate Settings page: each
    # analysis page renders only the global controls it needs as a compact
    # one-line row beneath the page title. Committed values live in cv_*
    # session keys (never widget keys, so never purged when a control isn't
    # shown on the current page). Inline widgets use w_* keys plus an
    # on_change callback that copies w_* -> cv_* at the start of the rerun,
    # so the pipeline below (which reads cv_*) always reflects the latest
    # selection with no one-rerun lag.
    # ========================================================================
    import re as _re
    import streamlit.components.v1 as _components

    ss = st.session_state

    # ---- Stop Streamlit's clear-cache shortcut from firing on Ctrl/Cmd+C.
    # The component iframe is same-origin (srcdoc), so it can add a
    # capture-phase keydown listener on the parent document that swallows the
    # shortcut for a copy gesture WITHOUT calling preventDefault (native copy
    # still works). Installed once via a window flag. ----
    _components.html(
        """
        <script>
        (function(){
          try {
            var d = window.parent.document;
            if (d.__adapt_copyfix) return;
            d.__adapt_copyfix = true;
            d.addEventListener('keydown', function(e){
              var k = (e.key || '').toLowerCase();
              if (k === 'c' && (e.ctrlKey || e.metaKey)) { e.stopImmediatePropagation(); }
            }, true);
          } catch (err) {}
        })();
        </script>
        """,
        height=0,
    )

    # View identifiers (constants so labels live in one place).
    V_FLOOD = "\U0001f30a Flood maps"
    V_MAP = "\U0001f5fa\ufe0f Dynamic maps"
    V_ROADS = "\U0001f6e3\ufe0f Road maps"
    V_OVERVIEW = "\U0001f4ca Overview"
    V_DIST = "\U0001f4e6 Damage distributions"
    V_RES = "\U0001f3e0 Residential example"
    V_NONRES = "\U0001f3e2 Non-residential example"
    V_NSI = "\U0001f5c2\ufe0f NSI dataset"
    V_FRAG = "\U0001f4c8 Fragility curves"
    VIEWS = [V_OVERVIEW, V_MAP, V_FLOOD, V_ROADS, V_DIST, V_RES, V_NONRES, V_NSI, V_FRAG]

    def _keyed_container(_key):
        try:
            return st.container(key=_key)
        except TypeError:
            return st.container()

    # ---- Navigation rail (rendered in the sidebar; CSS below flips it to
    #      the right edge and styles it as the dark ADAPT rail). Using the
    #      sidebar gives a stable selector across Streamlit versions. ----
    # Location is a GLOBAL control: it lives in the rail directly under the
    # brand and drives every tab (including the NSI dataset tab), so it is no
    # longer rendered as a per-page setting. cv_location is initialized here
    # (before the widget) so the sidebar selectbox can bind to it.
    def _cb_loc():
        ss.cv_location = ss.w_location
        # Mobile-homes-dominated default follows the location (on for Pamunkey).
        ss.cv_mobile = (ss.w_location == "Pamunkey")

    def _cb_mobile():
        ss.cv_mobile = ss.w_mobile

    if available_locations:
        if "cv_location" not in ss:
            ss.cv_location = "Mastic Beach" if "Mastic Beach" in available_locations else available_locations[0]
        if ss.cv_location not in available_locations:
            ss.cv_location = available_locations[0]
        if "cv_mobile" not in ss:
            ss.cv_mobile = (ss.cv_location == "Pamunkey")

    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        else:
            st.markdown(
                '<div class="adapt-rail-brand">'
                '<span class="adapt-rail-word">ADAPT</span>'
                '<span class="adapt-rail-sub">Assessment of Damage &amp; '
                'Adaptation Planning Tool</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        # Global location selector - directly under the brand.
        if available_locations:
            ss.w_location = ss.cv_location
            st.selectbox("\U0001f4cd Location", available_locations,
                         key="w_location", on_change=_cb_loc)
            ss.w_mobile = ss.cv_mobile
            st.checkbox(
                "Mobile-homes-dominated area", key="w_mobile", on_change=_cb_mobile,
                help="When on, the adaptation analysis for this area considers only raising "
                     "(elevating) homes - the realistic retrofit for manufactured/mobile housing - "
                     "and compares it against the no-mitigation baseline. The Overview, "
                     "Distributions, and the map's Adaptation-Effectiveness view reflect this; "
                     "the integrated baseline results are still shown for comparison. Default on "
                     "for Pamunkey.",
            )
        active = st.radio(
            "Navigation", options=VIEWS, key="adapt_active_view",
            label_visibility="collapsed",
        )

    # ---- No-data guard (precedes the pipeline, which indexes
    #      available_locations[0]) ----
    if len(available_locations) == 0:
        st.markdown(
            "<h1 style='text-align:center; color:#0f172a; font-weight:800; "
            "font-size:1.9rem; line-height:1.2; margin: 0.25rem 0 1.25rem 0;'>"
            "Building-level flood damage assessment under climate change scenarios"
            "</h1>",
            unsafe_allow_html=True,
        )
        st.error("\u26a0\ufe0f No data files found. Please ensure `.xlsx` result files (e.g., `Shinnecock_Results_ALL.xlsx`) are in the same directory as `app.py`.")
        st.stop()

    # ========================================================================
    # COMMITTED FILTER STATE (cv_*) + on_change callbacks
    # (cv_location + _cb_loc are set up earlier, next to the global rail
    # selector, so location can be committed before the sidebar renders.)
    # ========================================================================
    def _cb_occ():  ss.cv_occ = ss.w_occ
    def _cb_year(): ss.cv_year = ss.w_year
    def _cb_scn():  ss.cv_scn = ss.w_scn
    def _cb_dfe():  ss.cv_dfe = ss.w_dfe
    def _cb_zero(): ss.cv_showzero = ss.w_showzero

    # --- Location is committed by the global rail selector above. ---
    selected_location = ss.cv_location

    # Mobile-homes-dominated area: restrict the adaptation comparison to raising
    # (elevating) homes vs the no-mitigation baseline across the integrated views.
    mobile_raise_only = bool(ss.get("cv_mobile", False))
    _RAISE_ONLY_ACTIONS = ('No mitigation', 'Elevate')

    def _restrict_to_raise_only(df):
        """Keep only the no-mitigation baseline and the Elevate (raise-home)
        action. Returns df unchanged if it has no 'Action' column."""
        if df is None or 'Action' not in getattr(df, 'columns', []):
            return df
        return df[df['Action'].isin(_RAISE_ONLY_ACTIONS)].copy()

    ss.setdefault("cv_occ", "All")
    if ss.cv_occ not in ("All", "Residential", "Non-Residential"):
        ss.cv_occ = "All"
    selected_occupancy = ss.cv_occ

    # --- Load + occupancy filter + baseline-year drop (unchanged logic) ---
    df_agg_raw = None
    df_buildings_raw = None
    loc_entry = None
    if selected_location and selected_location in data_store:
        loc_entry = data_store[selected_location]
        df_agg_raw = loc_entry.get('agg')
        df_buildings_raw = loc_entry.get('buildings')

    df_buildings = filter_by_occupancy(df_buildings_raw, selected_occupancy)

    BASELINE_YEAR_TO_DROP = 2025
    if df_buildings is not None and 'TargetYear' in df_buildings.columns:
        df_buildings = df_buildings[df_buildings['TargetYear'] != BASELINE_YEAR_TO_DROP].copy()
    if df_agg_raw is not None and 'TargetYear' in df_agg_raw.columns:
        df_agg_raw = df_agg_raw[df_agg_raw['TargetYear'] != BASELINE_YEAR_TO_DROP].copy()

    preloaded_agg = None
    if loc_entry is not None and loc_entry.get('agg_by_occ'):
        preloaded_agg = loc_entry['agg_by_occ'].get(selected_occupancy)
        if preloaded_agg is not None and 'TargetYear' in preloaded_agg.columns:
            preloaded_agg = preloaded_agg[
                preloaded_agg['TargetYear'] != BASELINE_YEAR_TO_DROP
            ].copy()

    # --- Year options + label formatter (unchanged logic) ---
    available_years = [2040, 2055, 2060, 2100]
    if df_buildings is not None and 'TargetYear' in df_buildings.columns:
        available_years = sorted(df_buildings['TargetYear'].unique())
    elif df_agg_raw is not None and 'TargetYear' in df_agg_raw.columns:
        available_years = sorted(df_agg_raw['TargetYear'].unique())

    target_year_labels = {}
    if loc_entry is not None and isinstance(loc_entry.get('target_year_labels'), dict):
        target_year_labels = loc_entry['target_year_labels']

    def _format_target_year(y):
        label = target_year_labels.get(int(y), str(int(y)))
        if label != str(int(y)) and label != 'Potential':
            return f"{label} ({int(y)})"
        return str(int(y))

    _default_year = available_years[0] if len(available_years) > 0 else None
    if len(available_years) > 1:
        for _y in available_years:
            if target_year_labels.get(int(_y)) not in (None, 'Potential'):
                _default_year = _y
                break
    if "cv_year" not in ss:
        ss.cv_year = _default_year
    if ss.cv_year not in list(available_years):
        ss.cv_year = _default_year
    target_year = ss.cv_year

    # --- SLR scenario options (unchanged logic) ---
    available_scenarios = ['50th-percentile', '90th-percentile']
    if df_buildings is not None and 'SLR' in df_buildings.columns:
        available_scenarios = sorted(df_buildings['SLR'].unique())
    elif df_agg_raw is not None and 'SLR' in df_agg_raw.columns:
        available_scenarios = sorted(df_agg_raw['SLR'].unique())
    if "cv_scn" not in ss:
        ss.cv_scn = available_scenarios[0] if available_scenarios else '50th-percentile'
    if ss.cv_scn not in list(available_scenarios):
        ss.cv_scn = available_scenarios[0] if available_scenarios else '50th-percentile'
    scenario = ss.cv_scn

    # --- DFE status filter options (unchanged logic) ---
    if df_buildings is not None and 'DFE_Status' in df_buildings.columns:
        fp_options = df_buildings['DFE_Status'].dropna().unique().tolist()
    else:
        fp_options = []
    if "cv_dfe" not in ss:
        ss.cv_dfe = list(fp_options)
    ss.cv_dfe = [d for d in ss.cv_dfe if d in fp_options]
    if not ss.cv_dfe and fp_options:
        ss.cv_dfe = list(fp_options)
    dfe_filter = ss.cv_dfe if fp_options else None

    ss.setdefault("cv_showzero", True)
    show_zero_damage = ss.cv_showzero

    location_name = selected_location if selected_location else ""
    occupancy_label = selected_occupancy if selected_occupancy != "All" else "All Buildings"

    # ========================================================================
    # GLOBAL CSS -- nav in the sidebar, flipped to the RIGHT, white labels
    # ========================================================================
    _css = (
        "<style>\n"
        # Nav lives in the sidebar (left by default).
        "[data-testid=\"stAppViewContainer\"] { display: flex; }\n"
        # Dark ADAPT rail styling on the sidebar.
        "section[data-testid=\"stSidebar\"] {\n"
        "    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;\n"
        "    border-right: 1px solid rgba(148,163,184,0.18);\n"
        "    min-width: 248px !important; max-width: 320px !important;\n"
        "}\n"
        "section[data-testid=\"stSidebar\"] > div,\n"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stSidebarContent\"],\n"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stSidebarUserContent\"] {\n"
        "    background: transparent !important;\n"
        "}\n"
        # Brand block.
        ".adapt-rail-brand {\n"
        "    display: flex; flex-direction: column; align-items: flex-start;\n"
        "    padding: 0 0.35rem 0.9rem 0.35rem; margin: 0 0 0.6rem 0;\n"
        "    border-bottom: 1px solid rgba(148,163,184,0.22);\n"
        "}\n"
        ".adapt-rail-word { font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px;\n"
        "    color: #38bdf8; line-height: 1; }\n"
        ".adapt-rail-sub { font-size: 0.62rem; color: #cbd5e1; font-weight: 500;\n"
        "    margin-top: 4px; line-height: 1.25; }\n"
        # Nav radio: no radio dots, pill for selected, smooth transitions.
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] { gap: 5px; }\n"
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label {\n"
        "    display: flex; align-items: center; padding: 0.62rem 0.8rem; margin: 0; width: 100%;\n"
        "    border-radius: 10px; cursor: pointer; background: transparent;\n"
        "    transition: background-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;\n"
        "}\n"
        # Hide the little radio circle on the left of each nav item. The
        # visible ring is a <div>, but the native <input> is the actual first
        # child of the label, so `div:first-child` matched nothing on this
        # Streamlit build and the dot stayed (turning red/#FF4B4B when the tab
        # was selected). `div:first-of-type` targets the first DIV child
        # specifically, i.e. the circle wrapper, while leaving the label's
        # text (a separate markdown div) untouched. Hiding the input too is a
        # safety net in case a future Streamlit build reorders the children.
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label > div:first-of-type,\n"
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label input[type=\"radio\"] { display: none !important; }\n"
        # FORCE plain white on the label text, whatever the inner element is.
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label,\n"
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label * {\n"
        "    color: #ffffff !important; font-weight: 600; font-size: 0.95rem;\n"
        "}\n"
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label:hover {\n"
        "    background: rgba(148,163,184,0.16); transform: translateX(3px);\n"
        "}\n"
        "section[data-testid=\"stSidebar\"] [role=\"radiogroup\"] > label:has(input:checked) {\n"
        "    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);\n"
        "    box-shadow: 0 6px 16px rgba(14,165,233,0.40); transform: translateX(3px);\n"
        "}\n"
        # Global Location selector under the brand: white label + value on the
        # dark rail (the base stylesheet sets sidebar selectbox labels dark).
        "section[data-testid=\"stSidebar\"] .stSelectbox label,\n"
        "section[data-testid=\"stSidebar\"] .stSelectbox label p,\n"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stWidgetLabel\"] p {\n"
        "    color: #ffffff !important; font-weight: 600 !important;\n"
        "}\n"
        "</style>"
    )
    st.markdown(_css, unsafe_allow_html=True)

    # ========================================================================
    # PAGE TITLE
    # ========================================================================
    if selected_location:
        page_title = f"Building-level flood damage assessment for {selected_location}"
        if selected_occupancy != "All":
            page_title += f" - {selected_occupancy}"
    else:
        page_title = "Building-level flood damage assessment under climate change scenarios"
    st.markdown(
        "<h1 style='text-align:center; color:#0f172a; font-weight:800; "
        "font-size:1.9rem; line-height:1.2; margin: 0.25rem 0 1.0rem 0;'>"
        f"{page_title}"
        "</h1>",
        unsafe_allow_html=True,
    )

    # ========================================================================
    # NO-BUILDINGS GUARD
    # ========================================================================
    if df_buildings is None or len(df_buildings) == 0:
        st.warning(f"No {selected_occupancy.lower()} buildings found in the data for {selected_location}.")
        st.stop()

    # ========================================================================
    # COMPUTE AGGREGATED DATA (unchanged logic)
    # ========================================================================
    df_agg = None
    if preloaded_agg is not None and not preloaded_agg.empty:
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
    # PER-PAGE INLINE SETTINGS (compact one-line row beneath the title)
    # ========================================================================
    def _occ_fmt(x):
        return ("\U0001f3d8\ufe0f\U0001f3e2 All Buildings" if x == "All"
                else "\U0001f3d8\ufe0f Residential" if x == "Residential"
                else "\U0001f3e2 Non-Residential")

    def _scn_fmt(x):
        return ('Median SLR (50th pct)' if x == '50th-percentile'
                else 'High-End SLR (90th pct)' if x == '90th-percentile' else x)

    def render_inline_settings(items):
        items = [i for i in items if (i != "dfe" or fp_options)]
        if not items:
            return
        cols = st.columns(len(items))
        for _col, name in zip(cols, items):
            with _col:
                if name == "occ":
                    ss.w_occ = ss.cv_occ
                    st.selectbox("\U0001f3e0 Occupancy", ["All", "Residential", "Non-Residential"],
                                 key="w_occ", on_change=_cb_occ, format_func=_occ_fmt)
                elif name == "year":
                    ss.w_year = ss.cv_year
                    st.selectbox("\U0001f4c5 Target Year", available_years,
                                 key="w_year", on_change=_cb_year, format_func=_format_target_year)
                elif name == "scn":
                    ss.w_scn = ss.cv_scn
                    st.selectbox("\U0001f30a SLR Scenario", available_scenarios,
                                 key="w_scn", on_change=_cb_scn, format_func=_scn_fmt)
                elif name == "dfe":
                    ss.w_dfe = ss.cv_dfe
                    st.multiselect("DFE status", fp_options, key="w_dfe", on_change=_cb_dfe)
                elif name == "showzero":
                    ss.w_showzero = ss.cv_showzero
                    st.checkbox("Show $0 buildings", key="w_showzero", on_change=_cb_zero)

    _PAGE_SETTINGS = {
        V_MAP:      ["occ", "year", "scn"],
        V_OVERVIEW: ["occ", "year", "scn"],
        V_DIST:     ["occ", "year", "scn"],
        V_RES:      ["year", "scn"],
        # V_NONRES previously showed only the location selector, which is now
        # global; it therefore has no per-page settings row.
        # V_FLOOD, V_ROADS, V_NSI, V_FRAG -> no global settings row
    }
    # Short, web-sourced flood-context blurbs shown on the Overview page.
    # Sources: National Trust for Historic Preservation, NOAA, USGS, VIMS,
    # pamunkey.org (2025); sealevelrise.org / Woods Hole (Chesapeake).
    _LOC_BLURB = {
        "Pamunkey": (
            "The Pamunkey Indian Reservation sits on a low-lying ~1,600-acre peninsula on a bend "
            "of the Pamunkey River near West Point, Virginia, ringed by river and tidal marsh on "
            "three sides. It faces compounding flood pressure from tidal and storm flooding, "
            "stormwater, and land subsidence, with NOAA projecting roughly 3–6 ft of sea-level "
            "rise by 2100 - risk that led the National Trust to name it one of the 11 Most "
            "Endangered Historic Places of 2025."
        ),
        "West Point": (
            "West Point, Virginia occupies a low, flat point at the confluence of the Mattaponi "
            "and Pamunkey rivers (which join to form the York River) in the tidal Chesapeake Bay "
            "region - among the fastest sea-level-rise rates on the U.S. East Coast, where land "
            "subsidence compounds tidal and storm-surge flooding."
        ),
    }
    _LOC_BLURB_DEFAULT = (
        "A low-lying coastal community increasingly exposed to rising sea level and more "
        "frequent tidal and storm-surge flooding."
    )

    def _location_blurb(_loc):
        if not _loc:
            return _LOC_BLURB_DEFAULT
        if _loc in _LOC_BLURB:
            return _LOC_BLURB[_loc]
        for _k, _v in _LOC_BLURB.items():
            if _k.lower() in str(_loc).lower():
                return _v
        return _LOC_BLURB_DEFAULT

    # ---- Per-page description: rendered once, directly below the page title
    #      and above any per-page settings row (and, on the Overview, above the
    #      location blurb). Each page's body no longer renders its own copy. ----
    _PAGE_DESC = {
        V_OVERVIEW: (
            '<p class="tab-description">Aggregated community-wide damage statistics '
            'comparing all adaptation strategies, separated by buildings Under DFE and '
            'Above DFE.</p>'
        ),
        V_MAP: (
            '<p class="tab-description">Interactive map showing building-level flood risk. '
            'Use the <b>Map View</b> selector to switch between damage intensity, adaptation '
            'effectiveness, binned damage, and per-building <b>flood occurrences</b> (how often '
            'each home floods through a chosen horizon). Hover any building for details.</p>'
        ),
        V_FLOOD: (
            '<p class="tab-description">Bathtub flood-inundation maps for water levels you specify. '
            'Enter present-day flood levels (ft NAVD88); the app adds projected sea-level rise for each '
            'planning horizon and maps the resulting inundation depth. Terrain is the USGS 3DEP 1/3 arc-second '
            '(~10&nbsp;m) DEM, read on demand for this area only.</p>'
        ),
        V_ROADS: (
            '<p class="tab-description">OpenStreetMap roads classified against the same bathtub flood levels '
            'as the Flood Maps tab. Each road is sampled along its length, its ground elevation read from the '
            'terrain, and every segment flagged <b style="color:#dc1414">flooded</b> (surface below the water '
            'level), <b style="color:#8a2be2">inaccessible</b> (dry but cut off from the road network by flooding), or '
            '<b style="color:#228b22">dry</b>.</p>'
        ),
        V_DIST: (
            '<p class="tab-description">Distribution of cumulative damage <b>across individual buildings</b> '
            'and counts of buildings by adaptation effectiveness. <b>Both SLR scenarios</b> are shown '
            'side-by-side for the selected target year, regardless of the SLR Scenario chosen below. '
            'For the community-aggregated distribution (community totals across Monte Carlo realizations), '
            'see the Community Summary tab.</p>'
        ),
        V_NONRES: (
            '<p class="tab-description">Flood depth at a single building. Pick a structure, enter the same '
            'flood levels used in the map tabs, and the app reports the projected water level by year (with '
            'sea-level rise) three ways: depth above the <b>ground</b>, depth above the <b>first floor</b>, '
            'and the absolute <b>NAVD88</b> water surface. Positive = water above that reference; '
            'negative = below it (no flooding).</p>'
        ),
        V_RES: (
            '<p class="tab-description">Select an individual building to view detailed damage projections '
            'across time horizons and compare adaptation options.</p>'
        ),
        V_NSI: (
            '<p class="tab-description">National Structure Inventory (NSI) field-survey '
            'tool - walk the map to add, verify, move, or flag buildings and record their '
            'structural attributes. Edits sync to the shared Google Sheet backend.</p>'
        ),
        V_FRAG: (
            '<p class="tab-description">Explore the FEMA/Hazus depth-damage (fragility) curves '
            'behind the damage model. Pick an occupancy, flood zone, and basement condition, and '
            'the tool overlays the structure- and content-damage curves for each number of stories '
            'on one plot. The curves are read directly from the FAST tables shipped with the app. '
            'In the example-building tabs, the building&#39;s own curve is highlighted.</p>'
        ),
    }
    if active in _PAGE_DESC:
        st.markdown(_PAGE_DESC[active], unsafe_allow_html=True)

    if active == V_OVERVIEW:
        # Location flood-context blurb - shown above the settings row.
        st.markdown(
            '<div style="background:#eff6ff; border-left:4px solid #0ea5e9; '
            'border-radius:6px; padding:0.7rem 0.95rem; margin:0.1rem 0 0.9rem 0; '
            'color:#0f172a; font-size:0.92rem; line-height:1.45;">'
            f'<span style="font-weight:700;">📍 {location_name}</span> &nbsp;·&nbsp; '
            f'{_location_blurb(selected_location)}'
            '</div>',
            unsafe_allow_html=True,
        )

    if active in _PAGE_SETTINGS:
        render_inline_settings(_PAGE_SETTINGS[active])

    # ========================================================================
    # FRAGILITY CURVES (FEMA / Hazus depth-damage functions)
    # Shared renderer: used by the "Fragility curves" view and embedded, per
    # selected building, in the Residential / Non-residential example tabs.
    # Curve CSVs are uploaded in-app (FAST structure- and content-damage
    # function tables) and cached in session_state.
    # ========================================================================
    _OCC_FULL = {
        "RES1": "Single Family Dwelling", "RES2": "Manufactured Housing",
        "RES3A": "Multi-Family Duplex", "RES3B": "Multi-Family (3-4 Units)",
        "RES3C": "Multi-Family (5-9 Units)", "RES3D": "Multi-Family (10-19 Units)",
        "RES3E": "Multi-Family (20-49 Units)", "RES3F": "Multi-Family (50+ Units)",
        "RES4": "Temporary Lodging", "RES5": "Institutional Dormitory", "RES6": "Nursing Home",
        "COM1": "Retail Trade", "COM2": "Wholesale Trade", "COM3": "Personal & Repair Services",
        "COM4": "Professional/Technical", "COM5": "Banks/Financial", "COM6": "Hospital",
        "COM7": "Medical Office/Clinic", "COM8": "Entertainment & Recreation", "COM9": "Theaters",
        "COM10": "Parking", "IND1": "Heavy Industrial", "IND2": "Light Industrial",
        "IND3": "Food/Drugs/Chemicals", "IND4": "Metals/Minerals Processing", "IND5": "High Technology",
        "IND6": "Construction", "AGR1": "Agriculture", "REL1": "Church/Non-Profit",
        "GOV1": "General Government", "GOV2": "Emergency Response", "EDU1": "Schools (K-12)",
        "EDU2": "Colleges/Universities",
    }

    # --- Parse a FAST curve Description into (stories, basement, zone) ---
    def _frag_parse_desc(desc):
        d = str(desc).lower()
        if "split" in d:
            stories = "Split level"
        elif "three or more" in d or "3 or more" in d:
            stories = "3+ stories"
        elif "two floor" in d or "two stor" in d or "2 floor" in d:
            stories = "2 stories"
        elif "one floor" in d or "one stor" in d or "1 floor" in d:
            stories = "1 story"
        elif "1to2" in d or "1 to 2" in d:
            stories = "1\u20132 stories"
        else:
            stories = None
        if "w/ basement" in d or "with basement" in d or "sub-grade" in d or "subgrade" in d:
            basement = "With basement"
        elif "no basement" in d or "slab" in d or "no_basement" in d:
            basement = "No basement"
        else:
            basement = None
        if "coastal a or v" in d:
            zone = "Coastal A/V"
        elif "v-zone" in d or "v zone" in d or "coastal v" in d:
            zone = "V-Zone"
        elif "a-zone" in d or "a zone" in d or "coastal a" in d:
            zone = "A-Zone"
        else:
            zone = None
        return stories, basement, zone

    def _frag_story_key(x):
        s = str(x)
        m = _re.match(r"(\d+)", s)
        if m:
            return (0, int(m.group(1)))
        if "split" in s.lower():
            return (1, 0)
        return (2, s)

    def _frag_curve_long(_df):
        """A FAST DmgFn table -> tidy rows [fnid, occ, stories, basement, zone, desc, depth, pct]."""
        idcol = next((c for c in _df.columns if str(c).strip().lower().endswith("dmgfnid")),
                     _df.columns[0])
        depth_cols = {}
        for c in _df.columns:
            cs = str(c).strip().lower()
            mm = _re.fullmatch(r"m(\d+)", cs)
            pp = _re.fullmatch(r"p(\d+)", cs)
            if mm:
                depth_cols[c] = -int(mm.group(1))
            elif pp:
                depth_cols[c] = int(pp.group(1))
        occ_col = next((c for c in _df.columns if str(c).strip().lower() == "occupancy"), None)
        desc_col = next((c for c in _df.columns if str(c).strip().lower() == "description"), None)
        rows = []
        for _, r in _df.iterrows():
            occ = str(r[occ_col]).strip() if occ_col else ""
            desc = str(r[desc_col]).strip() if desc_col else ""
            fnid = r[idcol]
            stories, basement, zone = _frag_parse_desc(desc)
            for c, d in depth_cols.items():
                try:
                    v = float(r[c])
                except Exception:
                    continue
                rows.append((fnid, occ, stories, basement, zone, desc, d, v))
        return pd.DataFrame(
            rows, columns=["fnid", "occ", "stories", "basement", "zone", "desc", "depth", "pct"]
        )

    def _frag_map_lookup(_map, soid, idcol, zonecol):
        """SOID + hazard column -> DmgFnId (mirrors the MATLAB calc_damage_pcts)."""
        if _map is None or not soid:
            return None
        try:
            keycol = _map.columns[0]  # SOccupId
            m = _map[_map[keycol].astype(str).str.strip().str.upper() == str(soid).strip().upper()]
            if zonecol in m.columns:
                m = m[pd.to_numeric(m[zonecol], errors="coerce") == 1]
            idc = next((c for c in _map.columns if str(c).strip().lower() == idcol.lower()), None)
            if (not m.empty) and idc is not None:
                return m.iloc[0][idc]
        except Exception:
            pass
        return None

    def _try_frag_autoload():
        """Find the FEMA/Hazus FAST CSVs already sitting in the app folder and load
        them, so nothing needs to be uploaded. Pulls the two depth-damage function
        tables (…DmgFn) and the two SOID->FnId mapping tables (…DmgFinal)."""
        if ss.get("_frag_autoload_tried"):
            return
        ss["_frag_autoload_tried"] = True
        import glob
        seen, cands = set(), []
        for _d in (".", "data", "Data", "FAST", "fast", "curves", "depth_damage", "fragility"):
            for _f in glob.glob(os.path.join(_d, "*.csv")):
                if _f not in seen:
                    seen.add(_f)
                    cands.append(_f)
        sfn = cfn = smap = cmap = None
        for _p in cands:
            try:
                _head = pd.read_csv(_p, nrows=4)
            except Exception:
                continue
            cols = [str(c).strip().lower() for c in _head.columns]
            ndepth = sum(1 for c in cols if _re.fullmatch(r"[mp]\d+", c))
            is_cont = "cont" in os.path.basename(_p).lower()
            if ndepth >= 3 and "occupancy" in cols:          # …DmgFn (curves)
                if is_cont:
                    cfn = cfn or _p
                else:
                    sfn = sfn or _p
            elif "soccupid" in cols:                          # …DmgFinal (mapping)
                if is_cont:
                    cmap = cmap or _p
                else:
                    smap = smap or _p
        if sfn and cfn:
            try:
                ss["_frag_S"] = _frag_curve_long(pd.read_csv(sfn))
                ss["_frag_C"] = _frag_curve_long(pd.read_csv(cfn))
                ss["_frag_Smap"] = pd.read_csv(smap) if smap else None
                ss["_frag_Cmap"] = pd.read_csv(cmap) if cmap else None
                ss["_frag_src"] = (os.path.basename(sfn), os.path.basename(cfn))
            except Exception as _e:
                ss["_frag_err"] = str(_e)

    def render_fragility_curves(building_row=None, ctx="frag"):
        # Auto-load the curves already shipped with the app.
        if ss.get("_frag_S") is None or ss.get("_frag_C") is None:
            _try_frag_autoload()
        S = ss.get("_frag_S")
        C = ss.get("_frag_C")
        if S is None or C is None or S.empty:
            st.info(
                "The FEMA/Hazus depth-damage curve files weren't found in the app folder. "
                "Expected the FAST tables `flBldgStructDmgFn.csv` and `flBldgContDmgFn.csv` "
                "(and optionally the `…DmgFinal.csv` mapping tables) next to `app.py`. "
                "You can also load them here once."
            )
            cA, cB = st.columns(2)
            with cA:
                upS = st.file_uploader("Structure depth-damage CSV", type=["csv"], key=f"{ctx}_upS")
            with cB:
                upC = st.file_uploader("Content depth-damage CSV", type=["csv"], key=f"{ctx}_upC")
            if upS is not None and upC is not None:
                try:
                    ss["_frag_S"] = _frag_curve_long(pd.read_csv(upS))
                    ss["_frag_C"] = _frag_curve_long(pd.read_csv(upC))
                    st.rerun()
                except Exception as _e:
                    st.error(f"Could not parse the curve files: {_e}")
            return

        occs = sorted(S["occ"].dropna().unique())
        if not occs:
            st.warning("No occupancy classes were detected in the curve files.")
            return

        def _occ_label(o):
            return f"{o} - {_OCC_FULL[o]}" if o in _OCC_FULL else o

        # Defaults / exact curve when embedded for a specific building.
        bld_occ = bld_zone = bld_base = None
        hl_S = hl_C = None
        if building_row is not None:
            _raw = str(building_row.get("occupancy_type", "") or "").strip().upper()
            bld_occ = _raw if _raw in occs else next((o for o in occs if _raw.startswith(o)), None)
            _soid = str(building_row.get("SOID", "") or "").strip()
            if _soid:
                hl_S = _frag_map_lookup(ss.get("_frag_Smap"), _soid, "BldgDmgFnId", "HazardCA")
                hl_C = _frag_map_lookup(ss.get("_frag_Cmap"), _soid, "ContDmgFnId", "HazardCA")
            # Default the selectors to the resolved curve's own attributes so the
            # building's exact curve is actually inside the filtered overlay.
            if hl_S is not None and (S["fnid"] == hl_S).any():
                _hrow = S[S["fnid"] == hl_S].iloc[0]
                bld_occ = _hrow["occ"] or bld_occ
                bld_zone = _hrow["zone"]
                bld_base = _hrow["basement"]
            elif _soid:
                bld_base = "With basement" if _soid.upper().endswith("B") else "No basement"

        _default_occ = (bld_occ if bld_occ in occs
                        else "RES1" if "RES1" in occs else occs[0])

        c1, c2, c3 = st.columns(3)
        with c1:
            sel_occ = st.selectbox(
                "Occupancy", occs, index=occs.index(_default_occ),
                format_func=_occ_label, key=f"{ctx}_occ",
            )
        subO = S[S["occ"] == sel_occ]
        zones = [z for z in ["A-Zone", "V-Zone", "Coastal A/V", "Riverine"]
                 if z in set(subO["zone"].dropna())]
        with c2:
            if zones:
                _zi = (zones.index(bld_zone) if (bld_zone in zones)
                       else zones.index("A-Zone") if "A-Zone" in zones else 0)
                sel_zone = st.selectbox("Flood zone", zones, index=_zi, key=f"{ctx}_zone")
            else:
                sel_zone = None
                st.selectbox("Flood zone", ["(not in data)"], index=0, disabled=True, key=f"{ctx}_zone")
        subZ = subO if sel_zone is None else subO[subO["zone"] == sel_zone]
        bases = [b for b in ["No basement", "With basement"] if b in set(subZ["basement"].dropna())]
        with c3:
            if bases:
                _bi = bases.index(bld_base) if (bld_base in bases) else 0
                sel_base = st.selectbox("Basement", bases, index=_bi,
                                        key=f"{ctx}_base", disabled=(len(bases) == 1))
            else:
                sel_base = None
                st.selectbox("Basement", ["(not in data)"], index=0, disabled=True, key=f"{ctx}_base")

        def _filtered(df):
            sub = df[df["occ"] == sel_occ]
            if sel_zone is not None:
                sub = sub[sub["zone"] == sel_zone]
            if sel_base is not None:
                sub = sub[(sub["basement"] == sel_base) | (sub["basement"].isna())]
            return sub

        def _plot(df, title, hl_fnid):
            sub = _filtered(df)
            if sub.empty:
                st.info(f"No {title.lower()} curve for this selection.")
                return
            has_story = sub["stories"].notna().any() and sub["stories"].nunique() > 1
            series_col = "stories" if has_story else "desc"
            fig = go.Figure()
            for sv in sorted(sub[series_col].dropna().unique(), key=_frag_story_key):
                _s = sub[sub[series_col] == sv].sort_values("depth")
                is_hl = (hl_fnid is not None) and bool((_s["fnid"] == hl_fnid).any())
                _name = str(sv)
                if has_story is False:
                    _name = (_name[:38] + "\u2026") if len(_name) > 39 else _name
                if is_hl:
                    _name += "  \u25c0 this building"
                fig.add_trace(go.Scatter(
                    x=_s["depth"], y=_s["pct"], mode="lines+markers",
                    name=_name, line=dict(width=4 if is_hl else 2),
                ))
            fig.update_layout(
                title=title, xaxis_title="Flood depth above first floor (ft)",
                yaxis_title="Damage (% of value)", height=390,
                # Give the y tick numbers + rotated axis title their own room and
                # let automargin grow it as needed (with theme=None the old l=10
                # was too tight, so the numbers overlapped the title and clipped).
                margin=dict(l=70, r=20, t=44, b=52),
                legend=dict(orientation="h", y=-0.32, font=dict(size=10)),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#e5e7eb', gridwidth=1,
                           zeroline=False, automargin=True),
                yaxis=dict(showgrid=True, gridcolor='#e5e7eb', gridwidth=1,
                           zeroline=False, automargin=True, title_standoff=12,
                           ticksuffix="  "),
            )
            # theme=None so the explicit white background + gridlines above are
            # honored; Streamlit's default plotly theme otherwise overrides the
            # axis grid settings and the vertical gridlines disappear.
            st.plotly_chart(fig, use_container_width=True, theme=None, key=f"{ctx}_{title}")

        p1, p2 = st.columns(2)
        with p1:
            _plot(S, "Structure damage", hl_S)
        with p2:
            _plot(C, "Content damage", hl_C)

        _src = ss.get("_frag_src")
        _srctxt = (f" Source files: {_src[0]} / {_src[1]}." if _src else "")
        st.caption(
            "Curves overlay the available number-of-stories variants (or specific-occupancy "
            "variants) for the selected occupancy, flood zone, and basement condition. "
            "Depth is measured above the first-floor elevation; values are percent of structure "
            "or contents value (FEMA/Hazus FAST depth-damage functions)." + _srctxt
        )

    # ========================================================================
    # TAB: FLOOD MAPS - bathtub inundation for user-specified water levels
    # ========================================================================
    if active == V_FLOOD:
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
                f"**Present-day base levels** (ft NAVD88) - the same for both SLR scenarios; future "
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
            _map_years = sorted(set([2026] + [int(y) for y in available_years]))
            _years_sel = _c_yr.multiselect(
                "Planning horizons", options=_map_years, default=_map_years,
                format_func=lambda y: f"{int(y)} (present)" if int(y) == 2026 else str(int(y)),
            )
            _cz, _cd, _cb = st.columns([0.36, 0.32, 0.32])
            _zoom_factor = _cz.slider(
                "Zoom (tighten)", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
                help="Higher = tighter crop around the community (more zoomed in).",
            )
            _res_label = _cd.selectbox(
                "Map detail", ["Standard", "Fine", "Finer"], index=2,
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
                                # Title ABOVE the image (separate element - cannot overlap the figure).
                                _tgt.markdown(
                                    f"**{_lbl} - {int(_yr)}**<br>"
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
                        f"{_base_label} - © OpenStreetMap contributors / © CARTO."
                    )

    # ========================================================================
    # TAB: FLOODED ROADS - OSM road network classified by inundation
    # ========================================================================
    if active == V_ROADS:

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
                f"**Present-day base levels** (ft NAVD88) - the same inputs as the Flood Maps tab; future "
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
            _rmap_years = sorted(set([2026] + [int(y) for y in available_years]))
            _ryears_sel = _rc_yr.multiselect(
                "Planning horizons", options=_rmap_years, default=_rmap_years,
                format_func=lambda y: f"{int(y)} (present)" if int(y) == 2026 else str(int(y)),
                key="rd_years",
            )
            _rcz, _rcd, _rcb = st.columns([0.36, 0.32, 0.32])
            _rzoom = _rcz.slider(
                "Zoom (tighten)", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
                key="rd_zoom", help="Higher = tighter crop around the community.",
            )
            _rres_label = _rcd.selectbox(
                "Map detail", ["Standard", "Fine", "Finer"], index=2, key="rd_detail",
                help="Higher detail = sharper, larger images; a bit slower.",
            )
            _rres_m, _rtarget_px = {
                "Standard": (10.0, 700), "Fine": (5.0, 1000), "Finer": (3.0, 1300),
            }[_rres_label]
            _rbase_label = _rcb.selectbox("Basemap", ["OSM (color)", "Light", "Dark"], index=0, key="rd_base")
            # Reachability + entrance-gateway toggles on a single line. The
            # entrance toggle only applies when reachability is on, so its
            # column stays empty when reachability is off.
            _rk1, _rk2 = st.columns(2)
            with _rk1:
                _raccess_on = st.checkbox(
                    "Define \u201creachable\u201d from the largest connected network",
                    value=True, key="rd_access",
                    help="On (default): a dry road is flagged 'inaccessible' when flooding severs every dry "
                         "route from it to the largest connected road cluster, which is treated as the "
                         "mainland. Off: skip the reachability analysis entirely and color roads only as "
                         "flooded vs dry (no 'inaccessible' category).",
                )
            _rsource = "largest"
            _rentry_on = False
            _rentry_manual = ""
            with _rk2:
                if _raccess_on:
                    _rentry_on = st.checkbox(
                        "Keep the main entrance open as a guaranteed gateway", value=True, key="rd_entry",
                        help="The site's main access road off the main road often floods first and would otherwise make "
                             "the whole area read as inaccessible. With this on, that entrance is treated as a guaranteed "
                             "gateway: it still shows as flooded, but it no longer cuts the area off - interior roads are "
                             "still judged on their own flooding. The entrance is auto-detected from the building cluster "
                             "and shown below the maps so you can verify it.",
                    )
            if _raccess_on and _rentry_on:
                _rentry_manual = st.text_input(
                    "Entrance location override - lat, lon (optional)", value="", key="rd_entry_xy",
                    placeholder="auto-detect (leave blank)",
                    help="Leave blank to auto-detect the entrance from the building cluster. If the detected "
                         "point shown below the maps isn't the right road, pin it by pasting the entrance "
                         "coordinate here in the same 'lat, lon' order shown there (e.g. 37.5554, -76.8361).",
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
                                "public Overpass server may be busy - try again in a moment). %s" % _e
                            )

                if _Zm is not None and _ext is not None and _roads is not None:
                    st.markdown(
                        '<div style="margin:0.5rem 0 0.8rem;padding:0.55rem 0.8rem;background:#f8fafc;'
                        'border:1px solid #e2e8f0;border-radius:8px;font-size:1.05rem;">'
                        '<b style="font-size:1.2rem;">Roads</b>&nbsp;&nbsp;'
                        '<span style="color:#dc1414;font-weight:800;">━</span> flooded (surface below the water level)'
                        '&nbsp;&nbsp;&nbsp;<span style="color:#8a2be2;font-weight:800;">━</span> '
                        'inaccessible (dry but cut off from the network by flooding)'
                        '&nbsp;&nbsp;&nbsp;<span style="color:#228b22;font-weight:800;">━</span> dry (accessible)'
                        '<br><b style="font-size:1.05rem;">Water</b>&nbsp;&nbsp;'
                        '<span style="background:rgb(198,219,239);">&nbsp;&nbsp;</span>'
                        '<span style="background:rgb(107,174,214);">&nbsp;&nbsp;</span>'
                        '<span style="background:rgb(33,113,181);">&nbsp;&nbsp;</span>'
                        '<span style="background:rgb(8,69,148);">&nbsp;&nbsp;</span>'
                        '&nbsp; flood depth - light → dark blue = shallow → deep'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{len(_roads)} OpenStreetMap road way(s) in view.")
                    if _base_img is None:
                        _base_img = np.full((max(2, _Zm.shape[0]), max(2, _Zm.shape[1]), 3), 245, dtype=np.uint8)

                    # Entrance gateway: a manual override (lat, lon) wins; otherwise
                    # auto-detect from the building cluster, so a flooded entrance doesn't
                    # make the whole area read as inaccessible.
                    _entry = None
                    _entry_manual = False
                    if _rentry_on:
                        _mtxt = (_rentry_manual or "").strip()
                        if _mtxt:
                            try:
                                _v = [float(x) for x in
                                      _mtxt.replace(";", " ").replace(",", " ").split()]
                                if len(_v) >= 2:
                                    _a, _b = _v[0], _v[1]
                                    if _a < 0 and _b > 0:        # given as lon, lat
                                        _elon, _elat = _a, _b
                                    else:                         # given as lat, lon (as shown)
                                        _elat, _elon = _a, _b
                                    # treat the typed point as a "near here" hint and
                                    # snap it to the actual entrance gateway, so it need
                                    # not land exactly on a road node.
                                    try:
                                        _snap = fdem.detect_main_entrance(_roads, _ext, (_elon, _elat))
                                    except Exception:
                                        _snap = None
                                    _entry = _snap if _snap else [(_elon, _elat)]
                                    _entry_manual = True
                            except Exception:
                                _entry = None
                        if _entry is None:
                            try:
                                _site_lon = float(np.nanmean(_lonA)); _site_lat = float(np.nanmean(_latA))
                                _entry = fdem.detect_main_entrance(_roads, _ext, (_site_lon, _site_lat))
                            except Exception:
                                _entry = None
                        if _entry:
                            _elon, _elat = _entry[0]
                            _how = "pinned manually" if _entry_manual else "auto-detected from the building cluster"
                            st.caption(
                                f"Main entrance treated as a guaranteed gateway near "
                                f"**{_elat:.5f}, {_elon:.5f}** ({_how}). "
                                f"It still shows flooded, but no longer cuts the area off. If this isn't the right "
                                f"road, set the override above or untick the option."
                            )
                        else:
                            st.caption(
                                "Main-entrance exemption is on, but no single entrance could be auto-detected "
                                "from the network here, so it has no effect. You can pin one with the override above."
                            )

                    with st.spinner("Classifying roads and building maps…"):
                        for _scn in _rscn_sel:
                            st.markdown(f"##### {_rscn_pretty(_scn)}")
                            _cols = st.columns(2)
                            _k = 0
                            for _lbl, _blv in _rrows:
                                for _yr in _ryears_sel:
                                    _wl = _blv + _rslr(_scn, _yr)
                                    _segs, _cnt = fdem.classify_roads_access(
                                        _Zm, _ext, _roads, _wl, source=_rsource,
                                        entry_points=_entry, entrance_reach_m=450.0)
                                    if not _raccess_on:
                                        # Reachability analysis off: color roads as
                                        # flooded vs dry only, folding any inaccessible
                                        # segments back into dry/reachable. Flood
                                        # detection (status 2) is identical either way,
                                        # so only the inaccessible overlay is dropped.
                                        _segs = [(p0, p1, 0 if s == 1 else s)
                                                 for (p0, p1, s) in _segs]
                                        _cnt = dict(_cnt)
                                        _cnt['pct_dry'] = _cnt.get('pct_dry', 0.0) + _cnt.get('pct_inacc', 0.0)
                                        _cnt['pct_inacc'] = 0.0
                                    _depth = fdem.bathtub_depth_ft(_Zm, _wl, mask_water=True)
                                    _tgt = _cols[_k % 2]
                                    if _raccess_on:
                                        _metric_line = (
                                            f"flooded {_cnt['pct_flood']:.0f}% · "
                                            f"inaccessible {_cnt['pct_inacc']:.0f}% · "
                                            f"dry {_cnt['pct_dry']:.0f}%")
                                    else:
                                        _metric_line = (
                                            f"flooded {_cnt['pct_flood']:.0f}% · "
                                            f"dry {_cnt['pct_dry']:.0f}%")
                                    _tgt.markdown(
                                        f"**{_lbl} - {int(_yr)}**<br>"
                                        f"<span style='font-size:0.95rem;color:#374151'>"
                                        f"WL ≈ {_wl:.2f} ft NAVD88 &nbsp;•&nbsp; "
                                        f"{_metric_line}</span>",
                                        unsafe_allow_html=True,
                                    )
                                    _png = fdem.compose_road_png(_base_img, _depth, _segs, _ext)
                                    _tgt.image(_png, use_container_width=True)
                                    _k += 1

                    if _raccess_on:
                        st.caption(
                            "Road segments: red = flooded (sampled surface below the water level), "
                            "violet = dry but inaccessible (every dry route to the largest connected network is "
                            "severed by flooding), green = dry and reachable. Percentages are by segment count. "
                            "Accessibility is computed on the OpenStreetMap network graph (intersections recovered "
                            "from shared road vertices); roads that were already disconnected in the raw data are "
                            "not counted as flood-caused. Road "
                            f"elevations and flood shading from USGS 3DEP (~10 m), displayed at ~{_rres_m:.0f} m; "
                            "roads from OpenStreetMap. Open water (Z ≤ 0) is excluded from the flood mask."
                        )
                    else:
                        st.caption(
                            "Road segments: red = flooded (sampled surface below the water level), "
                            "green = dry. Reachability analysis is off, so roads are colored by flooding only "
                            "(no 'inaccessible' category). Percentages are by segment count. Road "
                            f"elevations and flood shading from USGS 3DEP (~10 m), displayed at ~{_rres_m:.0f} m; "
                            "roads from OpenStreetMap. Open water (Z ≤ 0) is excluded from the flood mask."
                        )

    # ========================================================================
    # TAB: BUILDING DEPTH - flood depth at one building vs ground / FFE / NAVD88
    # ========================================================================
    if active == V_NONRES:
        _bd_attrs = loc_entry.get('bldg_attrs') if loc_entry else None
        if _bd_attrs is None or _bd_attrs.empty:
            st.info("No building inventory (NSI) is available for this location.")
        else:
            _bd_ids = _bd_attrs['id'].astype(int).tolist()

            def _bd_fmt(bid):
                _r = _bd_attrs[_bd_attrs['id'].astype(int) == int(bid)]
                if _r.empty:
                    return str(bid)
                _r = _r.iloc[0]
                _occ = _r.get('occupancy_type', '')
                _addr = _r.get('address', '')
                _lbl = f"{int(bid)}"
                if isinstance(_occ, str) and _occ:
                    _lbl += f" - {_occ}"
                if isinstance(_addr, str) and _addr:
                    _lbl += f" - {_addr}"
                return _lbl

            _bd_default_id = 579513026
            _bd_idx = (_bd_ids.index(_bd_default_id)
                       if (selected_location == "Pamunkey" and _bd_default_id in _bd_ids) else 0)
            _bd_sel = st.selectbox(
                "Building (NSI id)", options=_bd_ids, index=_bd_idx,
                format_func=_bd_fmt, key="bd_id",
            )
            _brow = _bd_attrs[_bd_attrs['id'].astype(int) == int(_bd_sel)].iloc[0]
            _ground = _brow.get('ground_elevation')
            _ffe = _brow.get('FFE_ft')
            _fh = _brow.get('foundation_height')

            _info_bits = []
            if pd.notna(_ground):
                _info_bits.append(f"Ground elevation **{float(_ground):.2f} ft NAVD88**")
            if pd.notna(_ffe):
                _info_bits.append(f"First-floor elevation (FFE) **{float(_ffe):.2f} ft NAVD88**")
            if pd.notna(_fh):
                _info_bits.append(f"Foundation height **{float(_fh):.1f} ft**")
            if isinstance(_brow.get('occupancy_type'), str) and _brow.get('occupancy_type'):
                _info_bits.append(f"Occupancy **{_brow.get('occupancy_type')}**")
            if _info_bits:
                st.caption(" · ".join(_info_bits))

            # --- water-level scaffolding (same model as the map tabs) ----------
            _bdwl = loc_entry.get('water_levels', {}) if loc_entry else {}
            _bdscn_keys = [k for k in _bdwl.keys() if not k.endswith('_mc')]
            _bdscn_label = {
                '50th-percentile': 'Intermediate-High SLR (50th pct)',
                '90th-percentile': 'High SLR (90th pct)',
            }
            _bdscn_pretty = lambda k: _bdscn_label.get(k, k)
            _bdref = '50th-percentile' if '50th-percentile' in _bdwl else (_bdscn_keys[0] if _bdscn_keys else None)
            _bdref_df = _bdwl.get(_bdref)
            _bdbase_year = int(_bdref_df['Year'].min()) if (_bdref_df is not None and not _bdref_df.empty) else 2025

            def _bdlvl(df, year, col):
                if df is None or df.empty or col not in df.columns:
                    return None
                i = (df['Year'] - year).abs().idxmin()
                return float(df.loc[i, col])

            def _bdslr(scn_key, year):
                df = _bdwl.get(scn_key)
                a, b = _bdlvl(df, year, 'P50'), _bdlvl(df, _bdbase_year, 'P50')
                return (a - b) if (a is not None and b is not None) else 0.0

            _bdpf = {
                'annual': _bdlvl(_bdref_df, _bdbase_year, 'P50'),
                'ten':    _bdlvl(_bdref_df, _bdbase_year, 'P90'),
                'one':    _bdlvl(_bdref_df, _bdbase_year, 'P99'),
            }

            _bd_is_pam = (selected_location == "Pamunkey")
            if _bd_is_pam:
                _bddefs = [
                    ("High tide (MHHW)",         2.37,          True),
                    ("Monthly flood",            None,          False),
                    ("Annual flood",             _bdpf['annual'], False),
                    ("10% annual chance (10-yr storm)", 5.78,   True),
                    ("1% annual chance (100-yr storm)",  7.38,   True),
                ]
            else:
                _bddefs = [
                    ("High tide (MHHW)",         None,          False),
                    ("Monthly flood",            None,          False),
                    ("Annual flood",             _bdpf['annual'], _bdpf['annual'] is not None),
                    ("10% annual chance (10-yr storm)", _bdpf['ten'], _bdpf['ten'] is not None),
                    ("1% annual chance (100-yr storm)",  _bdpf['one'], _bdpf['one'] is not None),
                ]

            if _bdscn_keys:
                _bdscn = st.selectbox(
                    "SLR scenario", options=_bdscn_keys, index=0,
                    format_func=_bdscn_pretty, key="bd_scn",
                )
            else:
                _bdscn = None
                st.warning(
                    "No Monte-Carlo water-level data for this location, so future columns won't add sea-level "
                    "rise (they will equal the present-day level you enter)."
                )

            st.markdown("**Flood levels** (ft NAVD88) - tick the conditions to include and edit any value.")
            _bdrows = []
            for _lbl, _val, _on in _bddefs:
                _c0, _c1 = st.columns([0.42, 0.58])
                _inc = _c0.checkbox(_lbl, value=_on, key=f"bdf_inc_{_lbl}")
                _default = float(_val) if _val is not None else 0.0
                _lv = _c1.number_input(
                    f"{_lbl} level (ft NAVD88)", value=_default, step=0.5, format="%.2f",
                    key=f"bdf_val_{_lbl}", label_visibility="collapsed",
                )
                if _inc:
                    _bdrows.append((_lbl, float(_lv)))

            if not _bdrows:
                st.info("Tick at least one flood condition.")
            else:
                _years = sorted(set([2026] + [int(y) for y in available_years]))

                def _ftin(v):
                    if v is None or not np.isfinite(v):
                        return "-"
                    return f"{v:.2f} ft ({v * 12:.1f} in)"

                def _build_table(ref):
                    data = {}
                    for _lbl, _base_lv in _bdrows:
                        cells = []
                        for _y in _years:
                            _wl = _base_lv + (_bdslr(_bdscn, _y) if _bdscn else 0.0)
                            if ref == 'navd88':
                                _v = _wl
                            elif ref == 'ground':
                                _v = (_wl - float(_ground)) if pd.notna(_ground) else None
                            else:  # ffe
                                _v = (_wl - float(_ffe)) if pd.notna(_ffe) else None
                            cells.append(_ftin(_v))
                        data[_lbl] = cells
                    df = pd.DataFrame(data, index=[str(int(y)) for y in _years]).T
                    df.index.name = "Flood condition"
                    return df

                _scn_note = f" - {_bdscn_pretty(_bdscn)}" if _bdscn else ""
                st.caption(
                    f"Columns are evaluation years; future years add the median sea-level rise{_scn_note} "
                    f"to the present-day ({_bdbase_year}) level."
                )

                st.markdown("##### 1. Depth above ground level")
                if pd.notna(_ground):
                    st.caption(f"Water level minus the building's ground elevation ({float(_ground):.2f} ft NAVD88).")
                    st.dataframe(_build_table('ground'), use_container_width=True)
                else:
                    st.info("This building has no ground elevation in the inventory.")

                st.markdown("##### 2. Depth above first-floor elevation (FFE)")
                if pd.notna(_ffe):
                    st.caption(f"Water level minus the building's first-floor elevation ({float(_ffe):.2f} ft NAVD88).")
                    st.dataframe(_build_table('ffe'), use_container_width=True)
                else:
                    st.info("This building has no first-floor elevation in the inventory.")

                st.markdown("##### 3. Absolute water level (NAVD88)")
                st.caption("The projected still-water surface elevation itself (present-day level + sea-level rise).")
                st.dataframe(_build_table('navd88'), use_container_width=True)

    # ========================================================================
    # TAB: PER-BUILDING ANALYSIS - cross-building distributions + Plots 3/4/5
    # ========================================================================
    if active == V_DIST:
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

            # Mobile-homes-dominated area: keep only the raise-home and baseline
            # rows so the strategy distributions compare elevation vs no mitigation.
            if mobile_raise_only:
                df_b_year = _restrict_to_raise_only(df_b_year)
                st.caption(
                    "🏠 **Mobile-homes-dominated area:** distributions and effectiveness counts "
                    "consider only **raising (elevating) homes** vs the no-mitigation baseline."
                )
            
            if df_b_year.empty:
                st.warning(f"No per-building data for year {target_year}.")
            else:
                st.subheader(
                    f"Damage Distribution Across Buildings - {location_name} "
                    f"({occupancy_label}) - Year {target_year}"
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
                        would clip out - many buildings only see non-zero
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
                                # - exactly the cross-building 5/25/50/75/95
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
                # PLOTS 3, 4, 5 - Per-building damage classification
                # Ported from VVV_Visualization_for_workshop_MasticBeach.py
                # Uses per-building P90 as upper-tail proxy (matches the
                # workshop convention; P95 was tested earlier but the user
                # asked for P90 because it's less sensitive to the tail's
                # smallest realizations and reads more conservatively).
                # =============================================================
                st.subheader(f"Building Counts by Adaptation Effectiveness - Year {target_year}")
                
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
                    # fig4 below - we still always compute it so the stats
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
                    # strictly outperforms WFP Basement on P90 - i.e., elevation
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
                    if mobile_raise_only:
                        # No cheap retrofit in scope: an Elevation success is a
                        # damaged building whose P90 drops to ≤ $1k when raised.
                        if el_p90 is not None:
                            el_arr = np.where(np.isnan(el_p90), no_p90, el_p90)
                            mask_elev_works = any_damage & (el_arr <= thr)
                    elif el_p90 is not None and wb_p90 is not None:
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
                    if mobile_raise_only:
                        # Only raising homes is in scope: a single Elevation chart,
                        # using the threshold rule (raising brings P90 ≤ $1k).
                        def _v5(s):
                            nd = s['n_damaged']
                            return [100.0 * s['n_elev'] / nd if nd > 0 else 0]
                        def _c5(s): return [s['n_elev']]
                        fig5 = _make_paired_bar(
                            f"Damaged buildings where raising the home eliminates "
                            f"upper-tail damage by {target_year}",
                            _v5, _c5,
                            x_left='Elevate',
                            single_group=True,
                        )
                        col_p5, = st.columns(1)
                        with col_p5:
                            st.plotly_chart(fig5, use_container_width=True)
                    else:
                        if _use_raiseu_for_fig4:
                            def _v4(s):
                                nd = s['n_damaged']
                                return [100.0 * s['n_raiseu'] / nd if nd > 0 else 0]
                            def _c4(s): return [s['n_raiseu']]
                            fig4 = _make_paired_bar(
                                f"Damaged buildings where Raise Utilities eliminates "
                                f"upper-tail damage by {target_year}",
                                _v4, _c4,
                                x_left='Raise Utilities',
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
                                x_left='WFP Basement',
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
                            x_left='Elevate',
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
                        if mobile_raise_only:
                            # Only the raise-home column, threshold semantics.
                            row = {
                                'SLR Scenario':          s['label'],
                                'Buildings':             f"{s['n_tot']:,}",
                                'Damaged (P90 > $0)':    f"{s['n_sev_dmg']:,}  ({100*s['n_sev_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "-",
                                'Damaged (median > $0)': f"{s['n_p50_dmg']:,}  ({100*s['n_p50_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "-",
                                'Raising the home eliminates P90':
                                    f"{s['n_elev']:,}  ({100*s['n_elev']/nd:.1f}%)" if nd > 0 else "-",
                            }
                            tbl_rows.append(row)
                            continue
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
                            'Damaged (P90 > $0)':        f"{s['n_sev_dmg']:,}  ({100*s['n_sev_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "-",
                            'Damaged (median > $0)':     f"{s['n_p50_dmg']:,}  ({100*s['n_p50_dmg']/s['n_tot']:.1f}%)" if s['n_tot'] else "-",
                            elim_col_label:              f"{elim_count:,}  ({100*elim_count/nd:.1f}%)" if nd > 0 else "-",
                        }
                        if not only_above_dfe:
                            row['Elevation > WFP B (where WFP B fails)'] = (
                                f"{s['n_elev']:,}  ({100*s['n_elev']/nd:.1f}%)" if nd > 0 else "-"
                            )
                        tbl_rows.append(row)
                    if tbl_rows:
                        st.markdown("**Per-scenario summary**")
                        st.dataframe(pd.DataFrame(tbl_rows),
                                     use_container_width=True, hide_index=True)
                    
                    if mobile_raise_only:
                        st.caption(
                            "🏠 Mobile-homes-dominated area: only raising (elevating) homes is "
                            "considered. Per-building counts use the **P90** of cumulative damage as "
                            "the upper-tail proxy. The **damaged-buildings chart** shows the share of "
                            "buildings with median and with P90 damage greater than zero. The "
                            "**Elevation chart** shows, among damaged buildings, the share for which "
                            "raising the home brings P90 damage to ≤ $1k."
                        )
                    elif _use_raiseu_for_fig4:
                        st.caption(
                            "Per-building counts use the **P90** of the per-building cumulative damage as "
                            "the upper-tail proxy (matching the workshop visualization convention). "
                            "The **damaged-buildings chart** shows the share of buildings with median "
                            "damage greater than zero and the share with P90 damage greater than zero. "
                            "The **Raise Utilities chart** shows, among buildings that experience any "
                            "damage, the share for which raising utilities above BFE+2 ft brings P90 "
                            "damage to ≤ $1k. The **Elevation chart** shows, among damaged buildings "
                            "where WFP Basement is **not** sufficient on its own, the share for which "
                            "elevation strictly outperforms WFP Basement on P90 - i.e. elevation "
                            "provides meaningful additional protection beyond what basement "
                            "floodproofing achieves. This dominance rule matches the Map tab's "
                            "Adaptation Effectiveness classifier "
                            "(No Damage \u2192 Raise Utilities \u2192 Elevation \u2192 Residual)."
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
                            "elevation strictly outperforms WFP Basement on P90 - i.e. elevation "
                            "provides meaningful additional protection beyond what basement "
                            "floodproofing achieves. This dominance rule matches the Map tab's "
                            "Adaptation Effectiveness classifier "
                            "(No Damage \u2192 Raise Utilities \u2192 WFP Basement \u2192 Elevation \u2192 Residual)."
                        )
                
    
    # ========================================================================
    # TAB 2: BUILDING MAP
    # ========================================================================
    if active == V_MAP:
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
                "🗺️ Map view is unavailable for this dataset - building "
                "coordinates are missing. All other tabs remain fully functional."
            )
        elif df_buildings is not None:
            st.subheader(f"Building Risk Map - {location_name} ({occupancy_label}) - {target_year}, {scenario}")

            # Map View and Basemap on a single line, side by side.
            _mv_col, _bm_col = st.columns(2)
            with _mv_col:
                map_view = st.radio(
                    "Map View",
                    options=["Damage Heatmap", "Damage Bins", "Adaptation Effectiveness", "Flood Occurrences"],
                    horizontal=True,
                    key="map_view_selector",
                    help=(
                        "**Damage Heatmap**: continuous color by No-Mitigation P50 cumulative damage. "
                        "**Damage Bins**: discrete bins of upper-tail damage with breakpoints fixed across years. "
                        "**Adaptation Effectiveness**: classifies each building by which retrofit eliminates upper-tail damage. "
                        "**Flood Occurrences**: counts how many times each building's first floor floods (MC water level above FFE) "
                        "from 2025 through the selected horizon, colored by a chosen percentile of the 1,000-realization distribution."
                    ),
                )
            with _bm_col:
                # "Streets" = OpenStreetMap; "Aerial" overlays ESRI World
                # Imagery. Remembered globally (one key, not per-location).
                basemap_choice = st.radio(
                    "Basemap",
                    options=["Streets", "Aerial"],
                    horizontal=True,
                    key="map_basemap_choice",
                    help=(
                        "**Streets**: OpenStreetMap road network and place labels. "
                        "**Aerial**: satellite imagery (ESRI World Imagery) - useful for "
                        "spotting individual buildings, parking lots, vegetation, and "
                        "shoreline detail. Requires internet access at render time; if "
                        "the tile server is unreachable the map will fall back to a "
                        "white background."
                    ),
                )

            # Flood Occurrences view - choose which statistic of the
            # per-building MC occurrence-count distribution drives the map
            # color (mean / low / median / high). Defaults to the mean
            # (rounded up). Always defined (defaults to the mean) so
            # downstream code can reference it regardless of the active view.
            flood_pct_key = 'occ_mean'
            if map_view == "Flood Occurrences":
                _fp_label = st.radio(
                    "Flood-occurrence statistic (across the 1,000 MC water-level realizations)",
                    options=["Mean (average)", "P10 (low)", "Median (P50)", "P90 (high)"],
                    index=0, horizontal=True,
                    key="flood_occ_pct",
                    help=(
                        "For each building we count, in every MC realization, how many "
                        "times it floods (annual-max water level above its first-floor "
                        "elevation) from 2025 through the selected horizon - giving 1,000 "
                        "occurrence counts per building. This selector picks which "
                        "statistic of that distribution colors the map. The mean "
                        "(default) is the average count across the 1,000 realizations, "
                        "rounded to the nearest whole year."
                    ),
                )
                flood_pct_key = {
                    "Mean (average)": 'occ_mean',
                    "P10 (low)":      'occ_P10',
                    "Median (P50)":   'occ_P50',
                    "P90 (high)":     'occ_P90',
                }[_fp_label]

            # Map data filters + building search + point size, consolidated
            # onto a SINGLE row so all four map controls sit on one line.
            # Bound to the same committed keys the pipeline reads, so changes
            # apply to the map immediately. The building-ID search drops a
            # temporary highlight ring on the map that auto-expires after a few
            # seconds (HIGHLIGHT_TTL_SEC) so a clean screenshot can be taken;
            # "✖ Clear" dismisses it instantly. The checkbox and the Find/Clear
            # buttons get a small spacer above them so they line up with the
            # baseline of the labeled widgets on either side.
            HIGHLIGHT_TTL_SEC = 8.0
            # Pamunkey defaults to 2× point size because its small building
            # count makes the default markers read as tiny dots at the natural
            # map zoom; other locations default to 1×. Each location remembers
            # its own setting via a location-scoped slider key.
            _default_scale = 2.0 if location_name == "Pamunkey" else 1.0
            c_dfe, c_zero, c_find, c_btn, c_size = st.columns([2.6, 1.9, 2.4, 1.7, 2.4])
            with c_dfe:
                if fp_options:
                    ss.w_dfe = ss.cv_dfe
                    st.multiselect("DFE status filter", fp_options, key="w_dfe", on_change=_cb_dfe)
            with c_zero:
                st.markdown("<div style='height:2.05em'></div>", unsafe_allow_html=True)
                ss.w_showzero = ss.cv_showzero
                st.checkbox("Show buildings with $0 damage", key="w_showzero", on_change=_cb_zero)
            with c_find:
                search_id_text = st.text_input(
                    "Find building by ID",
                    value="",
                    key="map_search_bldg_id",
                    placeholder="e.g. 8466717",
                    help=(
                        "Enter a Building ID and press Enter (or click 🔍). "
                        f"A magenta ring will flash on that building for "
                        f"~{int(HIGHLIGHT_TTL_SEC)} seconds, then disappear "
                        "automatically - no extra marks left on the map for "
                        "screenshots. Click ✖ to clear immediately."
                    ),
                )
            with c_btn:
                st.markdown("<div style='height:2.05em'></div>", unsafe_allow_html=True)
                btn_a, btn_b = st.columns(2)
                with btn_a:
                    find_clicked = st.button("🔍 Find", key="map_search_find",
                                             use_container_width=True)
                with btn_b:
                    clear_clicked = st.button("✖ Clear", key="map_search_clear",
                                              use_container_width=True)
            with c_size:
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
                # stray decimal point - be forgiving.
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
                # "Hide $0-damage buildings" - pick the right damage metric
                # ----------------------------------------------------------
                # The hide filter must look at the SAME statistic the active
                # map view colors by, otherwise we hide buildings the user
                # would expect to see:
                #   * Damage Heatmap        → P50 (what the heatmap colors)
                #   * Damage Bins           → P90 (the bins are upper-tail)
                #   * Adaptation Effective. → P90 (categories are P90-based)
                # In particular, for the bins/effectiveness views, hiding by
                # P50 silently drops every building with P50 = 0 but
                # P90 > $1k - the very buildings that drive tail-risk
                # planning.
                if map_view == "Flood Occurrences":
                    # Flood view shows ALL buildings (a building that never
                    # floods is drawn green, not hidden), so the damage-based
                    # "$0 damage" hide filter doesn't apply here.
                    zero_filter_col = None
                elif map_view == "Damage Heatmap":
                    zero_filter_col = 'No mitigation_P50' if 'No mitigation_P50' in df_map.columns else None
                else:
                    # Upper-tail view - fall back to P50 only if P90 isn't loaded
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

                        # Mobile/manufactured homes (RES2) have only one realistic
                        # retrofit - raising (elevating) the whole home - so their
                        # hover lists just Elevate (+ the No-Mitigation baseline).
                        # Any non-RES2 home (RES1, RES3, RES4, …) shows every
                        # applicable option. This is decided per building from its
                        # own occupancy type, so a RES4 sitting inside a
                        # mobile-homes-dominated area still shows its full menu.
                        _row_occ = str(row.get('occupancy_type', '') or '').upper()
                        _row_is_mobile = _row_occ.startswith('RES2')

                        # Wet-floodproofing the FIRST floor only makes sense for
                        # multi-story buildings (you retreat upstairs). For a
                        # single-story building it would mean floodproofing the
                        # whole living space, so we hide that line from the hover -
                        # mirroring the building-detail cards. NaN stories => hide.
                        _row_nstory = row.get('number_of_stories')
                        _row_one_story = pd.isna(_row_nstory) or float(_row_nstory) <= 1

                        for col in action_cols_p50:
                            action_name = col.replace('_P50', '')
                            val = row.get(col, 0)

                            # RES2 (mobile/manufactured): suppress every non-Elevate
                            # action from the hover's strategy list, leaving the
                            # No-Mitigation baseline and Elevate. Non-RES2 homes fall
                            # through and show their full applicable set.
                            if _row_is_mobile and action_name not in _RAISE_ONLY_ACTIONS:
                                continue

                            # Single-story buildings: wet-floodproofing the first
                            # floor is not an applicable strategy - hide the line.
                            if action_name == 'WFP 1st' and _row_one_story:
                                continue

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

                    # Default zoom: frame the community tightly rather than the
                    # old fixed wide zoom. Fit the building extent (with a little
                    # padding) to a Web-Mercator zoom, clamped to a sensible range.
                    try:
                        _mlat = df_map['latitude'].dropna().to_numpy()
                        _mlon = df_map['longitude'].dropna().to_numpy()
                        if _mlat.size >= 2:
                            _lon_span = max((float(np.nanmax(_mlon)) - float(np.nanmin(_mlon))) * 1.3, 1e-3)
                            _lat_span = max((float(np.nanmax(_mlat)) - float(np.nanmin(_mlat))) * 1.3, 1e-3)
                            _zoom_lon = math.log2(360.0 / _lon_span)
                            _cl = max(math.cos(math.radians(float(center_lat))), 1e-6)
                            _zoom_lat = math.log2((360.0 / (_lat_span * 1.4)) * _cl)
                            _default_map_zoom = float(min(_zoom_lon, _zoom_lat))
                            # Bias the default toward a closer view of the community.
                            _default_map_zoom = max(min(_default_map_zoom + 1.5, 17.5), 13.0)
                        else:
                            _default_map_zoom = 15.5
                    except Exception:
                        _default_map_zoom = 15.5

                    # Tighter framing for the Pamunkey study area using the
                    # team-supplied corner coordinates, so the initial view
                    # focuses on the village rather than the full extent.
                    if selected_location == "Pamunkey":
                        _pam_lat = [37.5830, 37.5831, 37.5639, 37.5633]
                        _pam_lon = [-76.9852, -77.0255, -77.0257, -76.9844]
                        center_lat = sum(_pam_lat) / len(_pam_lat)
                        center_lon = sum(_pam_lon) / len(_pam_lon)
                        try:
                            _plon = max((max(_pam_lon) - min(_pam_lon)) * 1.12, 1e-3)
                            _plat = max((max(_pam_lat) - min(_pam_lat)) * 1.12, 1e-3)
                            _pz_lon = math.log2(360.0 / _plon)
                            _pcl = max(math.cos(math.radians(center_lat)), 1e-6)
                            _pz_lat = math.log2((360.0 / (_plat * 1.4)) * _pcl)
                            _default_map_zoom = float(max(min(min(_pz_lon, _pz_lat) + 0.4, 17.5), 13.5))
                        except Exception:
                            _default_map_zoom = 14.5

                    # `_point_scale` is set by the "Point size" slider at the
                    # top of the map tab (see search_col3 above). All map
                    # markers below multiply their literal sizes by this
                    # factor so the user can dial dot size up/down on the fly.
                    # Defaults to 2× for Pamunkey, 1× for other locations.

                    # =====================================================
                    # Helper: ring a small set of specific Tribal buildings
                    # beneath any colored category trace. Only the IDs in
                    # _HIGHLIGHT_RING_IDS are ringed, and only for Pamunkey -
                    # every other building (residential or not) is left
                    # unringed. (Replaces the old non-residential ring.)
                    # =====================================================
                    _HIGHLIGHT_RING_IDS = {10001, 10000013, 10002, 579513008, 10016}

                    def _add_highlight_ring(fig, df_subset, ring_size=13):
                        if location_name != "Pamunkey" or len(df_subset) == 0:
                            return
                        _ids = pd.to_numeric(df_subset['id'], errors='coerce')
                        hl = df_subset[_ids.isin(_HIGHLIGHT_RING_IDS)]
                        if len(hl) > 0:
                            fig.add_trace(go.Scattermapbox(
                                lat=hl['latitude'], lon=hl['longitude'],
                                mode='markers',
                                marker=dict(size=ring_size, color='black', opacity=1.0),
                                hoverinfo='skip',
                                showlegend=False,
                                name='_highlight_ring',
                            ))

                    def _add_highlight_ring_legend(fig):
                        """Legend-only stub explaining the black ring. Shown only
                        when Pamunkey actually has ringed Tribal buildings on screen."""
                        if location_name != "Pamunkey":
                            return
                        if not pd.to_numeric(df_map['id'], errors='coerce').isin(
                                _HIGHLIGHT_RING_IDS).any():
                            return
                        fig.add_trace(go.Scattermapbox(
                            lat=[None], lon=[None], mode='markers',
                            marker=dict(size=10 * _point_scale, color='black', opacity=1.0),
                            name='Tribal buildings (ringed)',
                            showlegend=True, hoverinfo='skip',
                        ))
                    
                    fig_map = go.Figure()
                    bin_caption_extra = ""  # for the Damage Bins view
                    flood_occ_note = ""     # for the Flood Occurrences view
                    _flood_occ_df = None    # populated by the Flood Occurrences view
                    
                    # =====================================================
                    # VIEW 1 - Damage Heatmap (existing; continuous color)
                    # =====================================================
                    if map_view == "Damage Heatmap":
                        if baseline_col:
                            df_zero = df_map[df_map[baseline_col] == 0]
                            df_nonzero = df_map[df_map[baseline_col] > 0]
                        else:
                            df_zero = pd.DataFrame()
                            df_nonzero = df_map

                        # Ring the Tribal buildings beneath the markers.
                        _add_highlight_ring(fig_map, df_map, ring_size=13 * _point_scale)

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
                        _add_highlight_ring_legend(fig_map)

                    # =====================================================
                    # VIEW 2 - Adaptation Effectiveness (4 categories)
                    # Ported from generate_action_animation.m
                    # Uses upper-tail (P95) cumulative damage as a proxy
                    # for the MATLAB script's P90.
                    # =====================================================
                    elif map_view == "Adaptation Effectiveness":
                        # Required columns - P90 is the upper-tail proxy
                        # (matches the Distributions tab and the workshop
                        # convention).
                        #
                        # Cheap-retrofit (yellow) bucket: the set of low-cost
                        # retrofits that can, on their own, bring a building's
                        # upper-tail (P90) damage to ~zero. This used to be a
                        # single per-location column (WFP Basement in basement
                        # inventories; Raise Utilities at Pamunkey's RES2/pier
                        # inventory, where WFP B is dropped at the data layer).
                        # We now consider BOTH WFP Basement AND Raise Utilities
                        # wherever their columns are present: a building lands in
                        # the yellow bucket if EITHER cheap retrofit eliminates
                        # its P90 damage. This restores Raise-Utilities
                        # effectiveness on the map in areas that are NOT mobile-
                        # home dominated (e.g. Mastic Beach), where it was
                        # previously ignored in favour of WFP Basement. At
                        # Pamunkey, WFP B is absent (dropped upstream), so only
                        # Raise Utilities contributes there - behaviour unchanged.
                        col_nomit = 'No mitigation_P90'
                        col_elev  = 'Elevate_P90'
                        # Raise Utilities and WFP Basement are now SEPARATE
                        # groups on this map (previously merged into one yellow
                        # "cheap retrofit" bucket). Either column may be absent
                        # for a given inventory (e.g. WFP Basement is dropped
                        # upstream at Pamunkey's RES2/pier inventory), in which
                        # case that group is simply empty and omitted.
                        col_raiseu = 'Raise Utilities_P90'
                        col_wfpb   = 'WFP B_P90'
                        _has_raiseu = col_raiseu in df_map.columns
                        _has_wfpb   = col_wfpb   in df_map.columns
                        _cheap_present = [(c, lbl) for c, lbl in (
                            (col_raiseu, 'Raise Utilities'),
                            (col_wfpb,   'WFP Basement'),
                        ) if c in df_map.columns]
                        _cheap_available = bool(_cheap_present)

                        # No mitigation + Elevate are hard-required; the cheap
                        # retrofit (yellow bucket) is optional. When neither cheap
                        # column is present - e.g. an inventory that is entirely
                        # RES2 manufactured housing, where Raise Utilities was
                        # dropped at the data layer - the yellow bucket is simply
                        # skipped rather than erroring the whole view.
                        _core_cols = (col_nomit, col_elev)
                        missing = [c for c in _core_cols if c not in df_map.columns]
                        if missing:
                            st.warning(
                                "This view needs P90 columns for No mitigation and "
                                f"Elevate. Missing: {', '.join(missing)}"
                            )
                        else:
                            thr = ZERO_THRESH_DISPLAY  # treat damages below $1k as zero
                            
                            # Preserve NaN so missing retrofit values don't silently
                            # count as effective (fillna(0) would do that).
                            no_mit_raw = df_map[col_nomit].values.astype(float)
                            elev_raw   = df_map[col_elev].values.astype(float)

                            no_mit = np.where(np.isnan(no_mit_raw), 0.0, no_mit_raw)
                            elev   = np.where(np.isnan(elev_raw), no_mit, elev_raw)

                            # Per-retrofit "brings P90 to <= thr" masks, computed
                            # SEPARATELY for Raise Utilities and WFP Basement so
                            # each gets its own group. NaN in a retrofit column
                            # ==> "retrofit not applied" ==> baseline (so a
                            # missing value can't push a building into a group).
                            # An absent column ==> an all-False mask.
                            def _brings_to_thr(colname, present):
                                if not present:
                                    return np.zeros(len(df_map), dtype=bool)
                                _raw = df_map[colname].values.astype(float)
                                _val = np.where(np.isnan(_raw), no_mit, _raw)
                                return _val <= thr
                            raiseu_to_thr = _brings_to_thr(col_raiseu, _has_raiseu)
                            wfpb_to_thr   = _brings_to_thr(col_wfpb,   _has_wfpb)
                            
                            # --- MAP classifier (threshold rule) ---
                            # Five buckets ordered by the cheapest adaptation
                            # that brings upper-tail damage below the $1k
                            # "no damage" threshold. Reading the colors top
                            # to bottom gives a decision pyramid: do nothing →
                            # cheap retrofit → expensive retrofit (elevation) →
                            # nothing works.
                            #
                            #   1 = No Damage        baseline P90 <= thr
                            #   2 = Raise Utilities   baseline > thr AND Raise
                            #                         Utilities brings P90 <= thr
                            #   3 = WFP Basement      baseline > thr, Raise Utilities
                            #                         doesn't reach thr, but WFP
                            #                         Basement brings P90 <= thr
                            #   4 = Elevation         baseline > thr, neither cheap
                            #                         retrofit reaches thr, but
                            #                         elevation brings P90 <= thr
                            #   5 = Residual          Damaged, still Under-DFE AND
                            #                         below BFE, and even ELEVATION
                            #                         cannot bring P90 to zero
                            #                         (<= thr). Defined purely on
                            #                         elevation: homes that raising
                            #                         still can't bring to zero at
                            #                         the P90 upper-tail level.
                            #                         Homes at or above BFE are
                            #                         excluded (already reasonably
                            #                         protected) even inside the
                            #                         BFE-to-DFE freeboard band. The
                            #                         strongest adaptation in scope
                            #                         leaves residual damage, so the
                            #                         conversation has to move to
                            #                         buyout / relocation / managed
                            #                         retreat. Drawn in red.
                            #   6 = Out of scope  Above-DFE buildings that fall
                            #                     through (their retrofit options
                            #                     are different from those in the
                            #                     three-bucket pyramid above -
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
                            elev_to_thr = elev <= thr

                            # Under-DFE membership mask (Above-DFE buildings
                            # are NEVER classified as Residual - they fall
                            # through to cat=6 / omitted instead).
                            if 'DFE_Status' in df_map.columns:
                                _dfe_lower = (df_map['DFE_Status']
                                              .fillna('').astype(str)
                                              .str.strip().str.lower())
                                is_under_dfe = _dfe_lower.str.contains('under').values
                            else:
                                is_under_dfe = np.zeros(len(df_map), dtype=bool)

                            # Above-BFE exclusion: a home whose first-floor
                            # elevation (FFE) is at or above the Base Flood
                            # Elevation is already reasonably protected, so it
                            # must NOT be colored Residual Damage - even when it
                            # sits in the BFE-to-DFE freeboard band (still tagged
                            # "Under DFE"). Only homes we can affirmatively place
                            # BELOW BFE stay eligible for Residual; a missing
                            # FFE or BFE leaves a building eligible (we don't have
                            # grounds to exclude it). NaN FFE compares False here.
                            _bfe_val = (loc_entry or {}).get('bfe_ft')
                            if 'FFE_ft' in df_map.columns and _bfe_val is not None and pd.notna(_bfe_val):
                                _ffe_vals = pd.to_numeric(df_map['FFE_ft'], errors='coerce').values.astype(float)
                                is_above_bfe = _ffe_vals >= float(_bfe_val)
                            else:
                                is_above_bfe = np.zeros(len(df_map), dtype=bool)

                            # Priority classification. Default = 6 (out of
                            # scope / omitted) so any building not affirmatively
                            # placed in 1-5 drops off the map quietly. Cheapest
                            # working measure first (Raise Utilities before WFP
                            # Basement, per the workshop cost ordering).
                            cat = np.full(len(df_map), 6, dtype=int)
                            cat[no_mit <= thr] = 1
                            cat[(no_mit > thr) & raiseu_to_thr] = 2
                            cat[(no_mit > thr) & ~raiseu_to_thr & wfpb_to_thr] = 3
                            cat[(no_mit > thr) & ~raiseu_to_thr & ~wfpb_to_thr & elev_to_thr] = 4
                            cat[(no_mit > thr) & ~raiseu_to_thr & ~wfpb_to_thr & ~elev_to_thr
                                & is_under_dfe & ~is_above_bfe] = 5

                            df_map['_cat_action'] = cat

                            # Legend counts - each building appears in exactly
                            # one bucket, so these add up to (buildings shown).
                            cat_legend_counts = {
                                1: int((cat == 1).sum()),
                                2: int((cat == 2).sum()),
                                3: int((cat == 3).sum()),
                                4: int((cat == 4).sum()),
                                5: int((cat == 5).sum()),
                            }

                            # Palette walks the decision pyramid green -> lime ->
                            # yellow -> orange -> red. cat=6 (Above-DFE
                            # fall-through) is intentionally absent from cat_specs
                            # and therefore not plotted.
                            cat_specs = [
                                (1, 'No Damage',        '#22c55e'),  # green
                                (2, 'Raise Utilities',  '#a3e635'),  # lime
                                (3, 'WFP Basement',     '#facc15'),  # yellow
                                (4, 'Elevation',        '#f97316'),  # orange
                                (5, 'Residual Damage',  '#dc2626'),  # red
                            ]
                            # Drop a cheap-retrofit group entirely when its column
                            # isn't present for this inventory (its bucket is
                            # necessarily empty).
                            if not _has_raiseu:
                                cat_specs = [cs for cs in cat_specs if cs[0] != 2]
                            if not _has_wfpb:
                                cat_specs = [cs for cs in cat_specs if cs[0] != 3]
                            
                            for ci, label, color in cat_specs:
                                df_c = df_map[df_map['_cat_action'] == ci]
                                # Legend label uses the INDEPENDENT count
                                # (e.g. Elevation includes buildings where WFP-B
                                # also works); the colored markers on the map
                                # still follow priority order, so each building
                                # appears in only one color.
                                legend_count = cat_legend_counts.get(ci, len(df_c))
                                if len(df_c) == 0:
                                    # No markers in this priority bucket - still
                                    # show a legend stub with the independent count.
                                    fig_map.add_trace(go.Scattermapbox(
                                        lat=[None], lon=[None],
                                        mode='markers',
                                        marker=dict(size=8 * _point_scale, color=color, opacity=0.92),
                                        name=f"{label} ({legend_count})",
                                        showlegend=True, hoverinfo='skip',
                                    ))
                                    continue
                                _add_highlight_ring(fig_map, df_c, ring_size=13 * _point_scale)
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=df_c['latitude'], lon=df_c['longitude'],
                                    mode='markers',
                                    marker=dict(size=8 * _point_scale, color=color, opacity=0.92),
                                    hovertemplate='%{customdata[0]}<extra></extra>',
                                    customdata=list(df_c['hover_data']),
                                    name=f"{label} ({legend_count})",
                                ))
                            _add_highlight_ring_legend(fig_map)

                    # =====================================================
                    # VIEW 3 - Damage Bins (5 categories with dynamic breaks)
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
                                _add_highlight_ring(fig_map, df_no_dmg, ring_size=13 * _point_scale)
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
                                    # in this SLR scenario) - fall back to the
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
                                    _add_highlight_ring(fig_map, df_c, ring_size=13 * _point_scale)
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
                            
                            _add_highlight_ring_legend(fig_map)
                    
                    # =====================================================
                    # VIEW 4 - Flood Occurrences (MC exceedance of FFE)
                    # For each building, count per MC realization how many
                    # years (2025 → horizon) the annual-maximum water level
                    # exceeds its first-floor elevation, then color by a
                    # chosen percentile (P10 / median / P90) of that
                    # 1,000-wide occurrence-count distribution.
                    # =====================================================
                    elif map_view == "Flood Occurrences":
                        _wl_map = loc_entry.get('water_levels', {}) if loc_entry else {}
                        _mc_df = _wl_map.get(f'{scenario}_mc')
                        _attrs_fo = loc_entry.get('bldg_attrs') if loc_entry else None
                        if _mc_df is None or 'Year' not in getattr(_mc_df, 'columns', []):
                            _scn_lbl = ('High-End SLR (P90)' if scenario == '90th-percentile'
                                        else 'Median SLR (P50)')
                            st.warning(
                                f"Monte-Carlo water levels for the {_scn_lbl} scenario aren't "
                                f"available for {location_name}, so the flood-occurrences map "
                                f"can't be built. Switch the SLR scenario (above the map) to one "
                                f"that has an MC ensemble for this location."
                            )
                        elif _attrs_fo is None or 'FFE_ft' not in _attrs_fo.columns:
                            st.warning(
                                "First-floor elevations (FFE) aren't available for this "
                                "location, so flood occurrences can't be evaluated."
                            )
                        else:
                            # FFE per building computed over the FULL location
                            # inventory (not the occupancy/DFE-filtered subset),
                            # so the cached result is keyed correctly by
                            # (location, scenario, horizon) alone.
                            _ffe_by_id = {
                                int(i): (float(f) if pd.notna(f) else None)
                                for i, f in zip(_attrs_fo['id'], _attrs_fo['FFE_ft'])
                            }
                            _occ = compute_flood_occurrences(
                                _mc_df, _ffe_by_id,
                                (location_name, scenario, int(target_year)),
                            )
                            if _occ is None or _occ.empty:
                                st.info("No buildings with a first-floor elevation to evaluate.")
                            else:
                                n_years_win = int(_occ['n_years'].iloc[0])
                                occ_lut = _occ.set_index('id')

                                # Attach occurrence stats to the shown buildings.
                                dM = df_map.copy()
                                for _c in ('occ_mean', 'occ_P10', 'occ_P50', 'occ_P90'):
                                    dM[_c] = dM['id'].map(occ_lut[_c])
                                dM['occ_show'] = dM['id'].map(occ_lut[flood_pct_key])
                                if flood_pct_key == 'occ_mean':
                                    # The mean across realizations is fractional;
                                    # round to the nearest whole year (round half
                                    # up) so the bins/labels stay whole-year
                                    # integers like the percentiles do. A mean that
                                    # rounds to 0 (e.g. 0.3) reads as "never floods".
                                    dM['occ_show'] = np.floor(dM['occ_show'] + 0.5)
                                # Buildings without an FFE drop out (NaN) rather
                                # than being miscolored.
                                dM = dM[dM['occ_show'].notna()].copy()

                                _pct_word = {'occ_mean': 'mean, rounded',
                                             'occ_P10': '10th-pct',
                                             'occ_P50': 'median',
                                             'occ_P90': '90th-pct'}[flood_pct_key]

                                # Flood-specific hover (building id rides along in
                                # customdata[1] so map clicks still resolve).
                                _hover = []
                                for _r in dM.itertuples(index=False):
                                    _addr = getattr(_r, 'address', None)
                                    if isinstance(_addr, str) and _addr.strip():
                                        _h = (f"<b>{_addr}</b> "
                                              f"<span style='color:#94a3b8'>#{int(_r.id)}</span><br>")
                                    else:
                                        _h = f"<b>Building #{int(_r.id)}</b><br>"
                                    _ffe_v = getattr(_r, 'FFE_ft', None)
                                    if pd.notna(_ffe_v):
                                        _h += f"FFE {float(_ffe_v):.2f} ft NAVD88<br>"
                                    _h += (f"Floods <b>{_r.occ_show:.0f}</b> of {n_years_win} times "
                                           f"by {int(target_year)} ({_pct_word} MC)<br>")
                                    _h += (f"<span style='color:#64748b'>MC spread - "
                                           f"mean {_r.occ_mean:.1f} · "
                                           f"P10 {_r.occ_P10:.0f} · P50 {_r.occ_P50:.0f} · "
                                           f"P90 {_r.occ_P90:.0f} times</span>")
                                    _hover.append(_h)
                                dM['_fhd'] = [[t, int(i)] for t, i in zip(_hover, dM['id'])]

                                _never = dM[dM['occ_show'] <= 0]
                                _flood = dM[dM['occ_show'] > 0].copy()

                                # Green "never floods" layer first (drawn under).
                                if len(_never) > 0:
                                    _add_highlight_ring(fig_map, _never, ring_size=13 * _point_scale)
                                    fig_map.add_trace(go.Scattermapbox(
                                        lat=_never['latitude'], lon=_never['longitude'],
                                        mode='markers',
                                        marker=dict(size=8 * _point_scale, color='#22c55e', opacity=0.85),
                                        hovertemplate='%{customdata[0]}<extra></extra>',
                                        customdata=list(_never['_fhd']),
                                        name=f"Never floods (0 times)  ({len(_never)})",
                                    ))

                                if len(_flood) == 0:
                                    flood_occ_note = (
                                        f"No building's first floor floods in any year through "
                                        f"{int(target_year)} at the {_pct_word} MC level under this "
                                        f"SLR scenario."
                                    )
                                else:
                                    # Horizon-robust bins: split the 1..n_years
                                    # window at 10/25/50/75 % of years, expressed
                                    # back in absolute year counts. Bins recompute
                                    # with the horizon (window length) but read the
                                    # same way at every horizon.
                                    fracs = [0.10, 0.25, 0.50, 0.75]
                                    cuts = sorted({max(1, int(math.ceil(n_years_win * fr)))
                                                   for fr in fracs})
                                    edges = sorted(set([0] + cuts + [n_years_win + 1]))
                                    n_bins = len(edges) - 1
                                    # Sequential severity ramp: warm yellow (rare)
                                    # → deep red (floods almost every year). Warm
                                    # hues keep adjacent bins distinguishable
                                    # (an all-blue ramp blurs together) while the
                                    # green "never floods" markers stay clearly
                                    # separate from the flooded buildings.
                                    flood_palette = ['#fed976', '#feb24c', '#fd8d3c',
                                                     '#f03b20', '#bd0026']
                                    if n_bins <= len(flood_palette):
                                        bin_colors = flood_palette[:n_bins]
                                    else:
                                        bin_colors = (flood_palette +
                                                      [flood_palette[-1]] * (n_bins - len(flood_palette)))

                                    def _occ_bin_label(ci):
                                        # bin ci covers counts in [a, b]; bin 0's
                                        # zero is already split out as "never".
                                        a = max(edges[ci], 1) if ci == 0 else edges[ci]
                                        b = edges[ci + 1] - 1
                                        if b >= n_years_win:
                                            b = n_years_win
                                        if a >= b:
                                            return f"{a} time" if a == 1 else f"{a} times"
                                        return f"{a}\u2013{b} times"

                                    vals = _flood['occ_show'].to_numpy()
                                    _bidx = np.digitize(vals, edges[1:-1], right=False)
                                    _flood['_bin'] = _bidx
                                    for ci in range(n_bins):
                                        dfc = _flood[_flood['_bin'] == ci]
                                        if len(dfc) == 0:
                                            continue
                                        _add_highlight_ring(fig_map, dfc, ring_size=13 * _point_scale)
                                        fig_map.add_trace(go.Scattermapbox(
                                            lat=dfc['latitude'], lon=dfc['longitude'],
                                            mode='markers',
                                            marker=dict(size=8 * _point_scale,
                                                        color=bin_colors[ci], opacity=0.92),
                                            hovertemplate='%{customdata[0]}<extra></extra>',
                                            customdata=list(dfc['_fhd']),
                                            name=f"{_occ_bin_label(ci)}  ({len(dfc)})",
                                        ))
                                    flood_occ_note = (
                                        f"Each building is colored by the **{_pct_word}** number of times "
                                        f"its first floor floods between 2025 and {int(target_year)} "
                                        f"(a {n_years_win}-year window), taken across the 1,000 MC "
                                        f"water-level realizations. \u201cFlooded\u201d means the simulated "
                                        f"annual-maximum water level exceeds the building's first-floor "
                                        f"elevation. Bins split the window at 10 / 25 / 50 / 75 % of its length; "
                                        f"buildings that never flood under the selected statistic are green."
                                    )

                                _add_highlight_ring_legend(fig_map)
                                _flood_occ_df = dM   # for the metrics/caption below

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
                    # st_autorefresh - without it, the ring would persist
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
                                + (f" - {_addr}" if pd.notna(_addr) and str(_addr).strip() else "")
                            )
                            # Outer halo - large translucent magenta circle
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
                            # st_autorefresh package isn't installed -
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
                                # Optional dependency - silently skip the
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
                            zoom=_default_map_zoom
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
                    # df_map['hover_data'] was built) - we extract it below.
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
                            # cd may be [text, id]  OR  [[text, id]] (nested) - unwrap as needed
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
                        # User clicked an empty area of the map - clear the
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
                                "- open the **Details** tab to see its full profile."
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
                        if mobile_raise_only:
                            st.caption(
                                "🏠 **Mobile-homes-dominated area:** only **raising (elevating) homes** "
                                "is considered. Each building is colored by whether raising it eliminates "
                                "its upper-tail (P90) cumulative damage under the selected year and SLR "
                                "scenario: **No Damage** (baseline P90 \u2264 $1k - no intervention needed) \u2192 "
                                "**Elevation** (raising the home brings P90 \u2264 $1k) \u2192 **Residual Damage** "
                                "(Under-DFE homes **below BFE** where even raising the home leaves P90 above $1k - "
                                "the conversation has to move beyond retrofits, to buyout, relocation, or "
                                "community-scale interventions). Homes at or above BFE are never colored "
                                "Residual. Above-DFE buildings not eliminated by "
                                "raising are not plotted. Non-residential buildings are marked with a "
                                "black ring. Uncheck the sidebar option to compare all retrofits."
                            )
                        else:
                            # Caption mirrors the classifier: Raise Utilities and
                            # WFP Basement are separate groups; whichever cheap
                            # column is absent for this inventory is simply omitted.
                            _grp_bits = []
                            if _has_raiseu:
                                _grp_bits.append(
                                    "**Raise Utilities** (raising at-risk utilities brings P90 \u2264 $1k)"
                                )
                            if _has_wfpb:
                                _grp_bits.append(
                                    "**WFP Basement** (basement floodproofing brings P90 \u2264 $1k)"
                                )
                            _cheap_chain = (" \u2192 " + " \u2192 ".join(_grp_bits)) if _grp_bits else ""
                            st.caption(
                                "Each building is colored by the **cheapest adaptation that "
                                "eliminates** its upper-tail (P90) cumulative damage under "
                                "the selected year and SLR scenario. Groups are checked in "
                                "priority order: "
                                "**No Damage** (baseline P90 \u2264 $1k - no intervention needed)"
                                f"{_cheap_chain} \u2192 "
                                "**Elevation** (the cheaper retrofits don't reach the threshold "
                                "but elevation does) \u2192 "
                                "**Residual Damage** (Under-DFE homes **below BFE** where even "
                                "**elevation** cannot bring P90 \u2264 $1k, i.e. raising the home still "
                                "can't bring the upper-tail damage to zero - the conversation has to "
                                "move beyond retrofits, to buyout, relocation, or larger community-scale "
                                "interventions). Homes at or above BFE are never colored Residual, even "
                                "inside the BFE-to-DFE freeboard band. "
                                "Above-DFE buildings whose damage isn't eliminated by the "
                                "retrofits shown are not "
                                "plotted on this view - their relevant adaptation options "
                                "(wet floodproofing the first floor, content-only measures, "
                                "etc.) aren't represented in the pyramid above. "
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
                    elif map_view == "Flood Occurrences":
                        if flood_occ_note:
                            st.caption(
                                flood_occ_note +
                                " Non-residential buildings are marked with a black ring. "
                                "Switch the SLR scenario or horizon (year) above to update, and use "
                                "the percentile selector to read the low / median / high MC outcome."
                            )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if map_view == "Flood Occurrences" and _flood_occ_df is not None:
                            st.metric("Buildings Evaluated", f"{len(_flood_occ_df):,}")
                        else:
                            st.metric("Buildings Shown", f"{len(df_map):,}")
                    with col2:
                        if map_view == "Flood Occurrences" and _flood_occ_df is not None and len(_flood_occ_df):
                            _nflood = int((_flood_occ_df['occ_show'] > 0).sum())
                            _pctw = {'occ_mean': 'mean', 'occ_P10': 'P10',
                                     'occ_P50': 'median', 'occ_P90': 'P90'}[flood_pct_key]
                            st.metric(
                                f"Buildings flooding \u22651 time by {int(target_year)} ({_pctw} MC)",
                                f"{_nflood:,}",
                            )
                        else:
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
                        # The live map can still use OSM - your real browser
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
                                 "Imagery satellite tiles - great for showing physical "
                                 "context (shorelines, vegetation, building footprints) "
                                 "but heavier and lower-contrast in print. Pick "
                                 "`white-bg` for a guaranteed-working export with no "
                                 "roads/labels. OpenStreetMap is intentionally excluded "
                                 "- its tile servers block headless render requests, so "
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
                                 "and zoom - useful when you've panned/zoomed in the live "
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
                                    f"✅ Map ready - {ex_w}×{ex_h} px"
                                    f"{' (×' + str(int(export_scale)) + ')' if fmt == 'png' else ''}, "
                                    f"{len(img_bytes)/1024:.0f} KB"
                                )
                                if info == "basemap_fallback":
                                    st.warning(
                                        f"⚠ The chosen basemap (`{export_basemap_value}`) "
                                        "couldn't be rendered - its tile server refused the "
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
        # Data Notes - provenance / coverage information for the bundle
        # --------------------------------------------------------------
        # Surfaced below the map so users have a one-click path to
        # check what's in the inventory, what got skipped, and which
        # hazard inputs the analysis used. The expander stays collapsed
        # by default to keep the main view clean.
        if loc_entry is not None:
            with st.expander("ℹ️ Data Notes - coverage, exclusions, and bundle metadata", expanded=False):
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
                
                # Skipped buildings table - only shown when there are any.
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
                
                # Bundle metadata - small, gray, for the technically curious.
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
    if active == V_OVERVIEW:
        if df_agg is not None:
            st.subheader(f"Community-Wide Damage Summary - {location_name} ({occupancy_label}) - {target_year}, {scenario}")
            
            # In the community summary, hide the WFP First-floor strategy for
            # Pamunkey (manufactured-/single-story-dominant, so first-floor wet
            # floodproofing isn't a meaningful community-wide option there).
            _sum_agg = (df_agg[df_agg['Action'] != 'WFP 1st']
                        if selected_location == "Pamunkey" else df_agg)
            # Mobile-homes-dominated area: compare only raising homes vs baseline.
            if mobile_raise_only:
                _sum_agg = _restrict_to_raise_only(_sum_agg)
                st.caption(
                    "🏠 **Mobile-homes-dominated area:** only **raising (elevating) homes** is "
                    "shown as an adaptation option, compared against the no-mitigation baseline. "
                    "Uncheck the sidebar option to compare all strategies."
                )

            df_current = _sum_agg[
                (_sum_agg['TargetYear'] == target_year) & 
                (_sum_agg['SLR'] == scenario)
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
            # AGGREGATED DAMAGE DISTRIBUTION - Both SLR scenarios side-by-side
            # Box edges = Q1/Q3 (interpolated from CDF); whiskers = P05/P95;
            # white center line = P50. Boxes summarize the distribution of
            # COMMUNITY-TOTAL damage across Monte Carlo realizations.
            # ================================================================
            st.subheader(f"Aggregated Damage Distribution - Year {target_year}")
            st.markdown(
                f"<p style='color:#64748b;font-size:0.95rem;margin-top:-0.5rem;'>"
                "Distribution of <b>community-total</b> cumulative damage across Monte Carlo "
                "realizations. Both SLR scenarios are shown side-by-side, regardless of the "
                "SLR Scenario chosen above.</p>",
                unsafe_allow_html=True,
            )
            
            df_year_agg = _sum_agg[_sum_agg['TargetYear'] == target_year].copy()
            
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
                
                # Median-label vertical position: adjustable via the slider
                # rendered directly under these two panels. Stored as a percent
                # of the axis height; 0 (default) sits each median label right
                # at its median line. Read BEFORE building the figures so the
                # slider (placed under the plots) drives them on the same rerun.
                _med_shift_key = "agg_med_label_shift_pct"
                ss.setdefault(_med_shift_key, 0.0)
                _med_shift = float(ss[_med_shift_key]) / 100.0

                # Cumulative-damage panel (all strategies)
                sd_cum = {slr: [_agg_stats(a, slr) for a in actions_present_cs]
                          for slr, *_ in SCENARIO_SPECS}
                fig_agg_cum = build_box_whisker_panel(
                    group_labels=[action_labels_cs[a] for a in actions_present_cs],
                    scenario_data=sd_cum,
                    panel_title="Cumulative Damage",
                    y_label="Community-Total Cumulative Damage",
                    median_label_shift=_med_shift,
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
                        median_label_shift=_med_shift,
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

                # Adjust where the median value labels sit relative to their
                # median line. 0% (default) = no shift (label on the line);
                # positive lifts labels above the line, negative pushes below.
                st.slider(
                    "Median label vertical shift (% of axis height)",
                    min_value=-5.0, max_value=10.0, step=0.5,
                    key=_med_shift_key,
                    help="Nudge the median value labels up or down relative to "
                         "the median line on the two panels above. Defaults to "
                         "0% (no shift from the median).",
                )

                st.caption(
                    "Each box summarizes the distribution of the **community-total** cumulative "
                    "damage across Monte Carlo realizations. Box edges show the 25th and 75th "
                    "percentiles, the white center line is the median (P50), and whiskers extend "
                    "to the 5th and 95th percentiles. All five percentiles are taken directly "
                    "from the aggregated Monte Carlo results stored in the workbook - no "
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
                        title="Under DFE - All Strategies")
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
                            title="Above DFE - Strategies (excl. Elevate)")
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
            
            df_timeline = _sum_agg[
                (_sum_agg['SLR'] == scenario) &
                (_sum_agg['Action'].isin(traj_action_order))
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
                    title=f"Cumulative Damage Projection - {occupancy_label} ({scenario} SLR Scenario)",
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
                                text=(f"No-Mitigation Cumulative Damage by {group_label} - "
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
    # OVERVIEW - STRUCTURE-VALUE DISTRIBUTION BY BUILDING CATEGORY
    # Inventory view (independent of the page's occupancy filter): pick a
    # building category - default RES2 (manufactured housing) - and see how
    # many buildings fall in each structure-value band. Band edges use a
    # "nice" round-number width (… 25k / 50k / 100k / 250k …) chosen from the
    # data range so the ranges read cleanly on the axis.
    # ========================================================================
    if active == V_OVERVIEW:
        _sv_attrs = loc_entry.get('bldg_attrs') if loc_entry else None
        if (_sv_attrs is not None and len(_sv_attrs) > 0
                and 'structure_value' in _sv_attrs.columns
                and 'occupancy_type' in _sv_attrs.columns):
            st.divider()
            st.subheader("Structure-Value Distribution by Building Category")
            st.markdown(
                "<p style='color:#64748b;font-size:0.95rem;margin-top:-0.5rem;'>"
                "How many buildings of a chosen category sit in each structure-value "
                "band. This view spans the full inventory and is independent of the "
                "occupancy filter above.</p>",
                unsafe_allow_html=True,
            )

            _occ_counts = (_sv_attrs['occupancy_type'].dropna().astype(str)
                           .str.upper().str.strip().value_counts())
            _occ_options = list(_occ_counts.index)
            if not _occ_options:
                st.info("No occupancy information is available for this location.")
            else:
                _def_idx = _occ_options.index('RES2') if 'RES2' in _occ_options else 0
                _sv_occ = st.selectbox(
                    "Building category",
                    _occ_options, index=_def_idx, key="ov_sv_occ",
                    format_func=lambda c: f"{c}  ·  {int(_occ_counts[c]):,} buildings",
                )

                _occ_norm = _sv_attrs['occupancy_type'].astype(str).str.upper().str.strip()
                _vals = pd.to_numeric(
                    _sv_attrs.loc[_occ_norm == _sv_occ, 'structure_value'],
                    errors='coerce').dropna()
                _vals = _vals[_vals > 0]

                if len(_vals) == 0:
                    st.info(f"No positive structure values recorded for {_sv_occ}.")
                else:
                    def _nice_width(span, target=8):
                        raw = max(span / target, 1.0)
                        mag = 10.0 ** np.floor(np.log10(raw))
                        for _m in (1, 2, 2.5, 5, 10):
                            if raw <= _m * mag:
                                return float(_m * mag)
                        return float(10 * mag)

                    def _fmt_money_k(v):
                        if v >= 1e6:
                            return f"${v/1e6:.1f}M".replace('.0M', 'M')
                        return f"${v/1e3:.0f}k"

                    def _hex_lerp(c1, c2, t):
                        a = tuple(int(c1[i:i+2], 16) for i in (0, 2, 4))
                        b = tuple(int(c2[i:i+2], 16) for i in (0, 2, 4))
                        return '#%02x%02x%02x' % tuple(
                            int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))

                    vmin = float(_vals.min()); vmax = float(_vals.max())
                    width = _nice_width((vmax - vmin) if vmax > vmin else vmax, target=8)
                    lo = float(np.floor(vmin / width) * width)
                    hi = float(np.ceil(vmax / width) * width)
                    if hi <= lo:
                        hi = lo + width
                    edges = np.arange(lo, hi + width * 0.5, width)
                    counts, _ = np.histogram(_vals, bins=edges)
                    labels = [f"{_fmt_money_k(edges[i])}–{_fmt_money_k(edges[i+1])}"
                              for i in range(len(edges) - 1)]

                    # Light→deep indigo gradient by band: lower value = lighter,
                    # higher value = deeper, so the value axis reads at a glance.
                    _nb = max(len(labels) - 1, 1)
                    bar_colors = [_hex_lerp('c7d2fe', '3730a3', i / _nb)
                                  for i in range(len(labels))]
                    _avg = float(_vals.mean())
                    _ymax = max(int(counts.max()), 1)

                    fig_sv = go.Figure()
                    fig_sv.add_trace(go.Bar(
                        x=labels, y=counts,
                        marker=dict(color=bar_colors,
                                    line=dict(color='rgba(15,23,42,0.12)', width=1)),
                        text=[str(int(c)) if c > 0 else '' for c in counts],
                        textposition='outside',
                        textfont=dict(size=12, color='#334155'),
                        hovertemplate='%{x}<br><b>%{y}</b> buildings<extra></extra>',
                        cliponaxis=False,
                    ))
                    fig_sv.update_layout(
                        title=dict(
                            text=(f"{_sv_occ}  ·  {len(_vals):,} buildings  ·  "
                                  f"average {_fmt_money_k(_avg)}"),
                            x=0.01, xanchor='left',
                            font=dict(size=15, color='#0f172a')),
                        height=430, plot_bgcolor='white', paper_bgcolor='white',
                        margin=dict(l=64, r=24, t=58, b=84),
                        bargap=0.18, showlegend=False,
                        xaxis=dict(title='Structure value', tickangle=-30,
                                   automargin=True, showgrid=False,
                                   tickfont=dict(size=11, color='#475569'),
                                   title_font=dict(size=12, color='#334155')),
                        yaxis=dict(title='Number of buildings', automargin=True,
                                   showgrid=True, gridcolor='#eef2f7', gridwidth=1,
                                   zeroline=False, rangemode='tozero',
                                   range=[0, _ymax * 1.18],
                                   title_font=dict(size=12, color='#334155')),
                    )
                    # theme=None so the white background + custom gridlines are
                    # honored (Streamlit's default plotly theme overrides them).
                    st.plotly_chart(fig_sv, use_container_width=True,
                                    theme=None, key="ov_sv_chart")
                    st.caption(
                        f"Inventory for **{location_name}** · category **{_sv_occ}** · "
                        f"n = {len(_vals):,} · average {_fmt_money_k(_avg)} · "
                        f"range {_fmt_money_k(vmin)}–{_fmt_money_k(vmax)} · "
                        f"band width {_fmt_money_k(width)}. Structure values are "
                        "replacement-cost estimates from the building inventory."
                    )

    # ========================================================================
    # TAB 3: BUILDING DETAILS
    # ========================================================================
    if active == V_RES:
        if df_buildings is not None:
            st.subheader(f"🏠 Individual Building Analysis - {location_name} ({occupancy_label})")
            
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
                if stored_id is not None:
                    default_idx = sorted_ids.index(int(stored_id))
                elif selected_location == "Pamunkey" and 579536184 in building_ids:
                    default_idx = sorted_ids.index(579536184)
                elif selected_location == "Mastic Beach" and 8386312 in building_ids:
                    default_idx = sorted_ids.index(8386312)
                else:
                    default_idx = 0
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
                
                # Building-type and SOID provenance row - small, gray,
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
                # silently skip - no fallback to percentile interpolation
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
                    
                    # Available years from the per-building damage table, plus
                    # the present-day baseline (the MC sheets' 2025 base year)
                    # prepended so the panel leads with current conditions.
                    # df_building drops the baseline, so we read it from the MC
                    # 'Year' column directly.
                    _exp_mc_years = set()
                    for _k in ('50th-percentile', '90th-percentile'):
                        _m = wl_data.get(f'{_k}_mc')
                        if _m is not None and 'Year' in _m.columns:
                            _exp_mc_years |= set(int(v) for v in _m['Year'].unique())
                    _dmg_years = [int(y) for y in df_building['TargetYear'].unique()]
                    _present_y = min(_exp_mc_years) if _exp_mc_years else None
                    target_years_local = sorted(_dmg_years)
                    if _present_y is not None and _present_y not in _dmg_years:
                        target_years_local = [_present_y] + target_years_local
                    
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
                            # Present-day baseline is shown as "2026" to match
                            # the present column used in the other tabs.
                            label = ("2026" if (_present_y is not None
                                                and int(yr) == _present_y)
                                     else str(int(yr)))
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
                            "probability for this building's first floor - a direct, "
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
                # Prepend the present-day baseline (MC sheets' 2025 base year,
                # displayed as "2026") so the table leads with current conditions.
                _present_y2 = (int(mc_for_scenario['Year'].min())
                               if (mc_for_scenario is not None
                                   and 'Year' in mc_for_scenario.columns
                                   and len(mc_for_scenario)) else None)
                _dmg_years2 = [int(y) for y in df_building['TargetYear'].unique()]
                target_years_local = sorted(_dmg_years2)
                if _present_y2 is not None and _present_y2 not in _dmg_years2:
                    target_years_local = [_present_y2] + target_years_local
                if (pd.notna(ground_elev)
                        and mc_for_scenario is not None
                        and 'Year' in mc_for_scenario.columns
                        and target_years_local):
                    st.divider()
                    scen_pretty = {
                        '50th-percentile': 'Median SLR (P50)',
                        '90th-percentile': 'High-End SLR (P90)',
                    }.get(scenario, scenario)
                    st.subheader(f"Flood depth at this building - {scen_pretty}")
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
                        # Present-day baseline row is shown as "2026" to match
                        # the present column used in the other tabs.
                        yr_int = int(yr)
                        year_display = ("2026" if (_present_y2 is not None
                                                   and yr_int == _present_y2)
                                        else str(yr_int))
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
                            "selected SLR scenario; switch SLR above "
                            "to compare. "
                            "*Note:* the three return-period proxies use "
                            "P01 / P10 / P99 of the annual-maximum distribution "
                            "as you specified - let me know if you'd prefer "
                            "P50 / P90 / P99 (the standard engineering reading "
                            "of \"yearly / 10-year / 100-year\")."
                        )
                
                # ----------------------------------------------------------
                # Retrofit "slide" cards - the numbers laid out like the
                # workshop action slides (elevation references + Benefit and
                # Remaining-recovery-cost by horizon), framed so they're easy
                # to transcribe. Cards appear only when the retrofit applies:
                # Elevate is absent above DFE, WFP Basement absent without a
                # basement, WFP First-floor only for multi-story buildings.
                # ----------------------------------------------------------
                _scn_for_cards = scenario
                _scn_lbl_cards = {
                    '50th-percentile': 'Median SLR (P50)',
                    '90th-percentile': 'High-End SLR (P90)',
                }.get(_scn_for_cards, _scn_for_cards)
                _bfe = (loc_entry or {}).get('bfe_ft')
                _dfe = (float(_bfe) + 2.0) if (_bfe is not None and pd.notna(_bfe)) else None
                _gnd = building_info.get('ground_elevation')
                _ffe_c = building_info.get('FFE_ft')
                _nstory = building_info.get('number_of_stories')

                _card_horizons = [(2040, '15-yr'), (2055, '30-yr'), (2060, '35-yr')]
                _retrofit_specs = [
                    ('Raise Utilities', 'raise mechanicals', False),
                    ('WFP B',           'wet floodproof the basement', False),
                    ('Elevate',         'raise the house', True),
                    ('WFP 1st',         'wet floodproof the first floor', False),
                ]

                def _cum(action, yr, pct):
                    _r = df_building[(df_building['Action'] == action)
                                     & (df_building['SLR'] == _scn_for_cards)
                                     & (df_building['TargetYear'] == yr)]
                    col = f'CumEAD_{pct}'
                    if _r.empty or col not in _r.columns:
                        return None
                    v = _r[col].iloc[0]
                    return float(v) if pd.notna(v) else None

                def _money_c(v):
                    return format_currency(v) if v is not None else "-"

                _cards_html = []
                for _akey, _atitle, _show_raise in _retrofit_specs:
                    if df_building[(df_building['Action'] == _akey)
                                   & (df_building['SLR'] == _scn_for_cards)].empty:
                        continue
                    # WFP 1st Floor is shown whenever the model has data for it
                    # (it is dropped upstream for RES2 / manufactured homes, so
                    # the emptiness check above already hides it there). It is
                    # no longer gated on story count, so the residential example
                    # always shows a "wet floodproof the first floor" box when
                    # the building has WFP 1st data.

                    _elev_bits = []
                    if pd.notna(_gnd):
                        _elev_bits.append(f"Ground: <b>{float(_gnd):.1f} ft</b>")
                    if pd.notna(_ffe_c):
                        if _show_raise and _dfe is not None:
                            _elev_bits.append(
                                f"First floor: <b>raised from {float(_ffe_c):.1f} ft to {_dfe:.1f} ft</b>")
                        else:
                            _elev_bits.append(f"First floor: <b>{float(_ffe_c):.1f} ft</b>")
                    if _dfe is not None:
                        _elev_bits.append(f"DFE: <b>{_dfe:.1f} ft</b>")
                    if _bfe is not None and pd.notna(_bfe):
                        _elev_bits.append(f"BFE: <b>{float(_bfe):.1f} ft</b>")

                    _ben_lines, _rem_lines = [], []
                    for _yr, _hlabel in _card_horizons:
                        b50, a50 = _cum('No mitigation', _yr, 'P50'), _cum(_akey, _yr, 'P50')
                        b05, a05 = _cum('No mitigation', _yr, 'P05'), _cum(_akey, _yr, 'P05')
                        b95, a95 = _cum('No mitigation', _yr, 'P95'), _cum(_akey, _yr, 'P95')
                        if b50 is not None and a50 is not None:
                            ben = max(b50 - a50, 0.0)
                            rng = ""
                            if None not in (b05, a05, b95, a95):
                                _e1, _e2 = max(b05 - a05, 0.0), max(b95 - a95, 0.0)
                                rng = f" ({_money_c(min(_e1, _e2))}–{_money_c(max(_e1, _e2))})"
                            _ben_lines.append(
                                f"Benefit {_hlabel} ({_yr}): <b>{_money_c(ben)}</b>{rng}")
                        if a50 is not None:
                            rrng = (f" ({_money_c(a05)}–{_money_c(a95)})"
                                    if (a05 is not None and a95 is not None) else "")
                            _rem_lines.append(
                                f"Remaining recovery cost {_hlabel} ({_yr}): <b>{_money_c(a50)}</b>{rrng}")

                    card = (
                        '<div style="border:2px solid #1f6f8b;border-radius:10px;'
                        'padding:0.85rem 1.05rem;margin:0.6rem 0;background:#f5fafc;">'
                        f'<div style="font-size:1.18rem;font-weight:800;color:#0f4c5c;'
                        f'margin-bottom:0.35rem;">Should they {_atitle}?</div>'
                        + ('<div style="color:#374151;margin-bottom:0.5rem;">'
                           + ' &nbsp;•&nbsp; '.join(_elev_bits) + '</div>' if _elev_bits else '')
                        + ''.join(f'<div style="color:#1f2937;font-size:1.02rem;">{_l}</div>'
                                  for _l in _ben_lines)
                        + ('<div style="height:0.35rem;"></div>' if _rem_lines else '')
                        + ''.join(f'<div style="color:#6b7280;">{_l}</div>' for _l in _rem_lines)
                        + '</div>'
                    )
                    _cards_html.append(card)

                if _cards_html:
                    st.divider()
                    st.subheader("Retrofit options - figures for the action slides")
                    st.caption(
                        f"Computed under **{_scn_lbl_cards}** (switch SLR above). "
                        f"Benefit = avoided cumulative recovery cost vs. taking no action; remaining = the cost "
                        f"that still occurs with the action. Median with (low–high) range. Each box is one slide."
                    )
                    for _c in _cards_html:
                        st.markdown(_c, unsafe_allow_html=True)

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
                            name=f"{style['label']} - 90% CI",
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
                            text=f"Cumulative Damage Trajectory - Building #{selected_id}",
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
                            return "-"
                        return f"{format_currency(lo)} – {format_currency(hi)}"
                    
                    def _fmt_val(v):
                        return "-" if pd.isna(v) else format_currency(v)
                    
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
                            'Median SLR - P50': _fmt_val(med),
                            'Median SLR - 90% CI': _fmt_range(med_lo, med_hi),
                            'High-End SLR - P50': _fmt_val(high),
                            'High-End SLR - 90% CI': _fmt_range(high_lo, high_hi),
                            'High-End vs Median (Δ)': _fmt_val(delta),
                            'Increase (%)': f"{pct:.1f}%" if pd.notna(pct) else "-",
                        })
                    
                    st.markdown("**Side-by-side comparison across planning horizons**")
                    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                
                st.subheader(f"Adaptation Strategy Comparison - Both SLR Scenarios ({target_year})")
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
                    # and 1st-floor wet-floodproofing aren't viable retrofits -
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
                    
                    # Pivots over (Action, SLR) - values are this building's P05/P50/P95
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
                            # monotone - even if rank-correlation between
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
                                f"Benefit (avoided damage) by Strategy and SLR Scenario - "
                                f"Building #{selected_id}, Year {target_year}"
                            ),
                            y_label="Benefit - avoided cumulative damage vs No Mitigation",
                            height=500,
                        )
                        
                        st.plotly_chart(fig_strat, use_container_width=True)
                        st.caption(
                            "Each box shows the distribution of this building's **avoided damage** "
                            "(No-Mitigation damage minus the strategy's remaining damage) at the same "
                            "percentile rank, under the selected target year and for both SLR scenarios. "
                            "The white center line is the median benefit (matches the *Benefit - Median* "
                            "column below); the whiskers reach to the 5th and 95th percentile benefits "
                            "(matching the *Benefit - 5th pctile* and *Benefit - 95th pctile* columns). "
                            "No Mitigation is omitted because it has no self-benefit."
                        )
                    
                    # ============================================================
                    # Strategy performance tables - Damage, Benefit, Remaining damage
                    # Separate Median / Min / Max columns (not single "range" cells)
                    # ============================================================
                    def _fmt_v(v):
                        return "-" if pd.isna(v) else format_currency(v)
                    def _fmt_pct(v):
                        return "-" if pd.isna(v) else f"{v:.1f}%"
                    
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
                        """Per-strategy rows for a given (year, SLR) - split columns.
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
                                'Cost estimate':                        (ADAPTATION_COST_ESTIMATES.get(action, "-")
                                                                         if action != 'No mitigation' else "-"),
                                'Remaining damage - 5th pctile':        _fmt_v(dmg_lo),
                                'Remaining damage - Median':            _fmt_v(dmg_med),
                                'Remaining damage - 95th pctile':       _fmt_v(dmg_hi),
                                'Benefit - 5th pctile':                 _fmt_v(ben_lo) if action != 'No mitigation' else "-",
                                'Benefit - Median':                     _fmt_v(ben_med) if action != 'No mitigation' else "-",
                                'Benefit - 95th pctile':                _fmt_v(ben_hi) if action != 'No mitigation' else "-",
                                'Reduction (median)':                   _fmt_pct(red_pct) if action != 'No mitigation' else "-",
                            })
                        return rows
                    
                    # ---- Selected-year tables, one per SLR scenario ----
                    st.markdown(f"**Strategy performance - Year {target_year}**")
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
                        "median always falls between them. **Cost estimate** is a rough per-measure "
                        "installed-cost range and does not vary by year or SLR scenario. "
                        + ADAPTATION_COST_SOURCE
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
                            f"Building #{selected_id} - {scenario_label} "
                            "(change the SLR Scenario above to see the other)</span>",
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
                                "given strategy performs in 2040 vs 2055 vs 2060 vs 2100 - and how its benefit "
                                "and remaining damage evolve as sea level rises."
                            )
        else:
            st.warning("No per-building data available for this location.")
    
    # ========================================================================
    # TAB 4: SCENARIO COMPARISON
    # ========================================================================
    if active == V_OVERVIEW:
        st.divider()
        st.subheader("📈 Trends - scenario comparison across horizons")
        st.markdown('<p class="tab-description">Compare cumulative damage projections between Median (50th-percentile) and High-End (90th-percentile) sea level rise scenarios across all time horizons.</p>', unsafe_allow_html=True)
        
        if df_agg is not None:
            st.subheader(f"📈 Scenario Comparison - {location_name} ({occupancy_label})")
            
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
                    title=f"No Mitigation Damage: Median vs High-End SLR Scenarios - {occupancy_label}")
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
            # Per-SLR-scenario trajectories - all mitigation actions together
            # Separate plot per SLR scenario so each strategy's trend over time
            # can be compared directly to the No-Mitigation baseline.
            # ================================================================
            st.divider()
            st.subheader("📉 Mitigation Strategies Over Time - by SLR Scenario")
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
                'No mitigation':   '#ef4444',   # red - baseline
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
                    title=dict(text=f"{slr_label} - {occupancy_label}",
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
    # NSI DATASET VIEW  (embedded field-survey tool - app2)
    # ------------------------------------------------------------------------
    # Self-contained React/Leaflet survey app, embedded via components.html.
    # It is fully isolated from the rest of ADAPT: it does not read or write
    # any of the location/occupancy/year/scenario filters above, and it talks
    # to its own backend (Google Apps Script + ArcGIS/USGS/OSM). Its controls
    # render as a top bar and its building-detail form as a right panel, so it
    # matches the layout of the other tabs. No global settings row is shown
    # for this tab (V_NSI is intentionally absent from _PAGE_SETTINGS).
    # ========================================================================
    if active == V_NSI:
        # The NSI tool is a self-contained React/Leaflet browser app, so it
        # must live in a component iframe (that is what every Streamlit
        # component is). To make it read as a NATIVE full-page view rather
        # than a small fixed "box", we force ITS iframe to fill the viewport,
        # strip the frame border, and trim the surrounding block padding.
        # The height rule is scoped to the NSI component's own keyed container
        # (`.st-key-nsi_embed`) so it does NOT catch the tiny invisible
        # copy-fix helper component near the top of the page - an earlier,
        # unscoped rule inflated that height:0 iframe to 560px, which is what
        # produced the big empty band above the title on this tab.
        # Tune the "205px" offset if you want more/less breathing room above.
        st.markdown(
            "<style>"
            "section[data-testid='stMain'] .block-container,"
            "[data-testid='stMainBlockContainer']{padding-top:1.1rem;padding-bottom:0;}"
            ".st-key-nsi_embed [data-testid='stCustomComponentV1'],"
            ".st-key-nsi_embed [data-testid='stIFrame'],"
            ".st-key-nsi_embed iframe{"
            "height:calc(100vh - 205px)!important;min-height:560px!important;"
            "width:100%!important;border:none!important;display:block;"
            "}"
            "</style>",
            unsafe_allow_html=True,
        )
        nsi_html = load_nsi_tool_html()
        if nsi_html is None:
            st.error(
                "\u26a0\ufe0f `nsi_tool.html` not found. Place it next to `app.py` "
                "in the deployment so the NSI dataset tab can load."
            )
        else:
            # Pass the GLOBAL location into the embedded tool. The survey app
            # keys its locations by slug (e.g. 'Mastic Beach' -> 'masticbeach');
            # we normalize the rail's location name the same way and inject it
            # as window.__ADAPT_LOCATION, which the tool reads as its initial
            # location. Changing the rail selector re-runs Streamlit with a new
            # injected value, so the iframe reloads on the new location. If the
            # tool doesn't know the slug, it falls back to its own default.
            _slug = _re.sub(r'[^a-z0-9]', '', str(selected_location).lower())
            _inject = '<script>window.__ADAPT_LOCATION = "%s";</script>\n' % _slug
            # Anchor on the tag PREFIX (no trailing '>') so the injection still
            # lands ahead of the app's script even though the babel tag now
            # carries attributes (type="text/babel" data-presets="react"). The
            # injected plain <script> runs before Babel transpiles the app, so
            # window.__ADAPT_LOCATION is set in time for the tool to read it.
            _anchor = '<script type="text/babel"'
            nsi_html_loc = nsi_html.replace(_anchor, _inject + _anchor, 1)
            # The height arg is a fallback/min; the scoped CSS above stretches
            # this component's iframe to fill the viewport. The keyed container
            # gives it the `.st-key-nsi_embed` hook the CSS targets, so only
            # this iframe is resized (not the page's other components).
            with _keyed_container("nsi_embed"):
                _components.html(nsi_html_loc, height=900, scrolling=False)

    # ========================================================================
    # FRAGILITY CURVES VIEW
    # ========================================================================
    if active == V_FRAG:
        render_fragility_curves(building_row=None, ctx="frag_main")

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
            © 2025 Erfan Amini. All rights reserved.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
