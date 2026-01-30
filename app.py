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
# CUSTOM CSS FOR STYLING - LARGER FONTS
# ============================================================================
st.markdown("""
<style>
    /* Main title - ADAPT (increased from 3.5rem to 4rem) */
    .main-title {
        font-size: 4rem;
        font-weight: bold;
        color: #0ea5e9;
        margin-bottom: 0rem;
        text-align: center;
    }
    /* Subtitle - spelled out name (increased from 1.5rem to 1.75rem) */
    .main-subtitle {
        font-size: 1.75rem;
        color: #334155;
        margin-bottom: 0.5rem;
        text-align: center;
        font-weight: 500;
    }
    /* Tagline (increased from 1.1rem to 1.25rem) */
    .main-tagline {
        font-size: 1.25rem;
        color: #64748b;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Tab titles - bigger font (increased from 1.2rem to 1.4rem) */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.4rem !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 0.85rem 1.75rem;
    }
    
    /* Section headers/subheaders inside tabs (increased from 1.5rem to 1.75rem) */
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
    
    /* Tab description (increased from 1rem to 1.1rem) */
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
    name = os.path.basename(filename).replace('.csv', '').replace('.CSV', '')
    
    location = "Unknown Location"
    location_patterns = [
        ('MasticBeach', 'Mastic Beach'),
        ('Mastic_Beach', 'Mastic Beach'),
        ('WestPoint', 'West Point'),
        ('West_Point', 'West Point'),
        ('Shinnecock', 'Shinnecock'),
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
            if part not in ['CSV1', 'CSV2', 'Aggregated', 'PerBuilding', 'RES', 'NONRES', 'COM', 'ALL']:
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
    """Format large numbers as currency"""
    if pd.isna(value) or value == 0:
        return "$0"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.0f}"


@st.cache_data
def load_csv_file(filepath):
    """Load CSV file from path"""
    return pd.read_csv(filepath)


@st.cache_data
def load_csv_upload(file):
    """Load CSV file from upload"""
    return pd.read_csv(file)


def load_data_from_folder(data_folder="data"):
    """Load all CSV files from the data folder"""
    data_store = {}
    
    if not os.path.exists(data_folder):
        return data_store, []
    
    available_locations = set()
    
    csv_files = glob.glob(os.path.join(data_folder, "*.csv")) + glob.glob(os.path.join(data_folder, "*.CSV"))
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        location = parse_filename(filename)
        available_locations.add(location)
        
        if location not in data_store:
            data_store[location] = {'agg': None, 'buildings': None}
        
        df = load_csv_file(filepath)
        
        if 'CSV1' in filename.upper() or 'AGGREGATED' in filename.upper():
            data_store[location]['agg'] = df
        elif 'CSV2' in filename.upper() or 'PERBUILDING' in filename.upper() or 'PER_BUILDING' in filename.upper():
            if 'Floodplain_Status' in df.columns:
                df['Floodplain_Status'] = df['Floodplain_Status'].apply(convert_floodplain_status)
            data_store[location]['buildings'] = df
    
    return data_store, sorted(list(available_locations))


def prepare_map_data(df_buildings, target_year, scenario):
    """Prepare building data for map display."""
    df_filtered = df_buildings[
        (df_buildings['TargetYear'] == target_year) &
        (df_buildings['Scenario'] == scenario)
    ].copy()
    
    if df_filtered.empty:
        return None
    
    attr_cols = [col for col in df_filtered.columns if col not in 
                 ['Action', 'CumEAD_P05', 'CumEAD_P50', 'CumEAD_P95', 'TargetYear', 'Scenario']]
    
    df_base = df_filtered[df_filtered['Action'] == 'Baseline'][attr_cols].copy()
    
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
        (df_buildings['Scenario'] == scenario)
    ].copy()
    
    if df_filtered.empty:
        return None
    
    agg_data = []
    for action in df_filtered['Action'].unique():
        df_action = df_filtered[df_filtered['Action'] == action]
        
        row = {
            'TargetYear': target_year,
            'Scenario': scenario,
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
# MAIN APPLICATION
# ============================================================================

def main():
    # ========================================================================
    # LOAD DATA FROM FOLDER (if exists)
    # ========================================================================
    data_store, available_locations = load_data_from_folder("data")
    data_loaded_from_folder = len(available_locations) > 0
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        if not data_loaded_from_folder:
            st.header("📁 Data Files")
            
            st.subheader("Aggregated Data (CSV1)")
            agg_files = st.file_uploader(
                "Upload aggregated CSV files",
                type=['csv'],
                accept_multiple_files=True,
                key="agg_uploader",
                help="e.g., CSV1_Aggregated_WestPoint_ALL.csv"
            )
            
            st.subheader("Per-Building Data (CSV2)")
            building_files = st.file_uploader(
                "Upload per-building CSV files",
                type=['csv'],
                accept_multiple_files=True,
                key="building_uploader",
                help="e.g., CSV2_PerBuilding_WestPoint_ALL.csv"
            )
            
            if agg_files:
                for file in agg_files:
                    location = parse_filename(file.name)
                    if location not in available_locations:
                        available_locations.append(location)
                    
                    if location not in data_store:
                        data_store[location] = {'agg': None, 'buildings': None}
                    
                    df = load_csv_upload(file)
                    data_store[location]['agg'] = df
            
            if building_files:
                for file in building_files:
                    location = parse_filename(file.name)
                    if location not in available_locations:
                        available_locations.append(location)
                    
                    if location not in data_store:
                        data_store[location] = {'agg': None, 'buildings': None}
                    
                    df = load_csv_upload(file)
                    if 'Floodplain_Status' in df.columns:
                        df['Floodplain_Status'] = df['Floodplain_Status'].apply(convert_floodplain_status)
                    data_store[location]['buildings'] = df
            
            available_locations = sorted(list(set(available_locations)))
            
            st.divider()
        else:
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
        
        if selected_location and selected_location in data_store:
            df_agg_raw = data_store[selected_location].get('agg')
            df_buildings_raw = data_store[selected_location].get('buildings')
        
        df_buildings = filter_by_occupancy(df_buildings_raw, selected_occupancy)
        
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
        
        available_scenarios = ['P50', 'P90']
        if df_buildings is not None and 'Scenario' in df_buildings.columns:
            available_scenarios = sorted(df_buildings['Scenario'].unique())
        elif df_agg_raw is not None and 'Scenario' in df_agg_raw.columns:
            available_scenarios = sorted(df_agg_raw['Scenario'].unique())
        
        scenario = st.selectbox(
            "🌊 SLR Scenario",
            options=available_scenarios,
            format_func=lambda x: 'Median SLR (P50)' if x == 'P50' else 'High-End SLR (P90)' if x == 'P90' else x
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
    # HEADER
    # ========================================================================
    
    st.markdown('<p class="main-title">🌊 ADAPT</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Assessment of Damage and Adaptation Planning Tool</p>', unsafe_allow_html=True)
    
    location_name = selected_location if selected_location else ""
    occupancy_label = selected_occupancy if selected_occupancy != "All" else "All Buildings"
    
    if selected_location:
        tagline = f"Building-level flood damage assessment for <b>{selected_location}</b>"
        if selected_occupancy != "All":
            tagline += f" — {selected_occupancy}"
    else:
        tagline = "Building-level flood damage assessment under climate change scenarios"
    
    st.markdown(f'<p class="main-tagline">{tagline}</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # CHECK IF DATA IS LOADED
    # ========================================================================
    
    if len(available_locations) == 0:
        st.info("👆 Please upload your data files using the sidebar to get started.")
        
        st.subheader("📋 Expected Data Format")
        
        st.markdown("""
        Upload files with **ALL** buildings (the app will filter by occupancy type):
        
        **Example file names:**
        - `CSV1_Aggregated_WestPoint_ALL.csv`
        - `CSV2_PerBuilding_WestPoint_ALL.csv`
        
        The app will automatically categorize buildings based on the `occupancy_type` column:
        - **Residential**: Types starting with `RES` (RES1, RES2, RES3, etc.)
        - **Non-Residential**: All other types (COM, IND, GOV, EDU, etc.)
        """)
        
        st.stop()
    
    if df_buildings is None or len(df_buildings) == 0:
        st.warning(f"No {selected_occupancy.lower()} buildings found in the uploaded data for {selected_location}.")
        st.stop()
    
    # ========================================================================
    # COMPUTE AGGREGATED DATA
    # ========================================================================
    
    df_agg = None
    if df_buildings is not None:
        agg_frames = []
        for yr in df_buildings['TargetYear'].unique():
            for scn in df_buildings['Scenario'].unique():
                agg_df = aggregate_filtered_data(df_buildings, yr, scn)
                if agg_df is not None:
                    agg_frames.append(agg_df)
        if agg_frames:
            df_agg = pd.concat(agg_frames, ignore_index=True)
    
    # ========================================================================
    # MAIN CONTENT - TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Building Map", 
        "📊 Community Summary",
        "🏠 Building Details",
        "📈 Comparison"
    ])
    
    # ========================================================================
    # TAB 1: BUILDING MAP
    # ========================================================================
    with tab1:
        st.markdown('<p class="tab-description">Interactive map showing building-level flood risk; hover over buildings to compare baseline damage with all adaptation strategies.</p>', unsafe_allow_html=True)
        
        if df_buildings is not None:
            st.subheader(f"Building Risk Map — {location_name} ({occupancy_label}) — {target_year}, {scenario}")
            
            df_map = prepare_map_data(df_buildings, target_year, scenario)
            
            if df_map is None or len(df_map) == 0:
                st.warning("No buildings match the current filters.")
            else:
                if dfe_filter and 'Floodplain_Status' in df_map.columns:
                    df_map = df_map[df_map['Floodplain_Status'].isin(dfe_filter)]
                
                baseline_col = 'Baseline_P50' if 'Baseline_P50' in df_map.columns else None
                
                if baseline_col and not show_zero_damage:
                    df_map = df_map[df_map[baseline_col] > 0]
                
                if len(df_map) == 0:
                    st.warning("No buildings match the current filters.")
                else:
                    if baseline_col:
                        non_zero_damages = df_map[df_map[baseline_col] > 0][baseline_col]
                        max_damage = non_zero_damages.max() if len(non_zero_damages) > 0 else 1
                    else:
                        max_damage = 1
                    
                    action_cols_p50 = [col for col in df_map.columns if col.endswith('_P50')]
                    
                    hover_texts = []
                    for idx, row in df_map.iterrows():
                        text = f"<b>Building #{row['id']}</b><br>"
                        
                        if 'occupancy_type' in row:
                            text += f"Type: {row['occupancy_type']}<br>"
                        if 'structure_value' in row and pd.notna(row['structure_value']):
                            text += f"Structure Value: {format_currency(row['structure_value'])}<br>"
                        if 'Floodplain_Status' in row:
                            text += f"DFE Status: {row['Floodplain_Status']}<br>"
                        
                        text += "<br><b>━━━ Cumulative EAD ━━━</b><br>"
                        
                        baseline_val = row.get('Baseline_P50', 0)
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
                            
                            if action_name == 'Baseline':
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
                    
                    fig_map = go.Figure()
                    
                    if baseline_col:
                        df_zero = df_map[df_map[baseline_col] == 0]
                        df_nonzero = df_map[df_map[baseline_col] > 0]
                    else:
                        df_zero = pd.DataFrame()
                        df_nonzero = df_map
                    
                    if len(df_zero) > 0:
                        fig_map.add_trace(go.Scattermapbox(
                            lat=df_zero['latitude'],
                            lon=df_zero['longitude'],
                            mode='markers',
                            marker=dict(size=8, color='#1a1a1a', opacity=0.7),
                            hovertemplate='%{customdata}<extra></extra>',
                            customdata=df_zero['hover_text'],
                            name='No Damage ($0)'
                        ))
                    
                    if len(df_nonzero) > 0 and baseline_col:
                        fig_map.add_trace(go.Scattermapbox(
                            lat=df_nonzero['latitude'],
                            lon=df_nonzero['longitude'],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=df_nonzero[baseline_col],
                                colorscale=[
                                    [0, '#22c55e'],
                                    [0.10, '#84cc16'],
                                    [0.25, '#eab308'],
                                    [0.50, '#f97316'],
                                    [1.0, '#ef4444']
                                ],
                                cmin=0,
                                cmax=max_damage,
                                colorbar=dict(
                                    title="Baseline EAD ($)",
                                    tickformat="$,.0f",
                                    len=0.7,
                                    y=0.5
                                ),
                                opacity=0.85
                            ),
                            hovertemplate='%{customdata}<extra></extra>',
                            customdata=df_nonzero['hover_text'],
                            name='At Risk'
                        ))
                    
                    center_lat = df_map['latitude'].mean()
                    center_lon = df_map['longitude'].mean()
                    
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
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Buildings Shown", f"{len(df_map):,}")
                    with col2:
                        total_baseline = df_map[baseline_col].sum() if baseline_col else 0
                        st.metric("Total Baseline EAD", format_currency(total_baseline))
                    
                    st.subheader(f"🔴 Top 10 Highest Risk Buildings (Baseline)")
                    
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
            st.warning("Please upload Per-Building Data (CSV2) to view the map.")
    
    # ========================================================================
    # TAB 2: COMMUNITY SUMMARY
    # ========================================================================
    with tab2:
        st.markdown('<p class="tab-description">Aggregated community-wide damage statistics comparing all adaptation strategies, separated by buildings Under DFE and Above DFE.</p>', unsafe_allow_html=True)
        
        if df_agg is not None:
            st.subheader(f"Community-Wide Damage Summary — {location_name} ({occupancy_label}) — {target_year}, {scenario}")
            
            df_current = df_agg[
                (df_agg['TargetYear'] == target_year) & 
                (df_agg['Scenario'] == scenario)
            ]
            
            col1, col2, col3, col4 = st.columns(4)
            
            baseline_row = df_current[df_current['Action'] == 'Baseline']
            
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
                    st.metric(label="Baseline EAD (P50)", value=format_currency(baseline_p50),
                              help=f"Range: {format_currency(baseline_p05)} - {format_currency(baseline_p95)}")
                with col3:
                    st.metric(label="Under DFE (Baseline)", value=format_currency(infp_baseline),
                              help="Buildings with FFE below Design Flood Elevation (BFE+2)")
                with col4:
                    st.metric(label="Above DFE (Baseline)", value=format_currency(outfp_baseline),
                              help="Buildings with FFE above Design Flood Elevation (BFE+2)")
            
            st.divider()
            
            st.subheader("Adaptation Strategy Comparison by DFE Status")
            
            col_under, col_above = st.columns(2)
            
            with col_under:
                st.markdown("### 🔴 Under DFE (Below BFE+2)")
                
                if 'InFP_CumEAD_P50' in df_current.columns:
                    under_dfe_data = []
                    baseline_infp = df_current[df_current['Action'] == 'Baseline']['InFP_CumEAD_P50'].values
                    baseline_infp = baseline_infp[0] if len(baseline_infp) > 0 else 0
                    
                    for _, row in df_current.iterrows():
                        action = row['Action']
                        val = row['InFP_CumEAD_P50']
                        savings = baseline_infp - val
                        pct = (savings / baseline_infp * 100) if baseline_infp > 0 else 0
                        under_dfe_data.append({
                            'Action': action,
                            'EAD ($)': val,
                            'Savings': savings,
                            'Reduction (%)': pct
                        })
                    
                    df_under = pd.DataFrame(under_dfe_data)
                    
                    fig_under = px.bar(df_under, x='Action', y='EAD ($)', color='Action',
                        color_discrete_map={'Baseline': '#ef4444', 'Raise Utilities': '#f97316',
                            'WFP B': '#eab308', 'Elevate': '#22c55e', 'WFP 1st': '#3b82f6'},
                        title="Under DFE — All Strategies")
                    fig_under.update_layout(showlegend=False, height=300, yaxis_tickformat="$,.0f")
                    st.plotly_chart(fig_under, use_container_width=True)
                    
                    df_under_display = df_under.copy()
                    df_under_display['EAD ($)'] = df_under_display['EAD ($)'].apply(format_currency)
                    df_under_display['Savings'] = df_under_display['Savings'].apply(format_currency)
                    df_under_display['Reduction (%)'] = df_under_display['Reduction (%)'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(df_under_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No Under DFE data available.")
            
            with col_above:
                st.markdown("### 🟢 Above DFE (Above BFE+2)")
                
                if 'OutFP_CumEAD_P50' in df_current.columns:
                    above_dfe_data = []
                    baseline_outfp = df_current[df_current['Action'] == 'Baseline']['OutFP_CumEAD_P50'].values
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
                            'EAD ($)': val,
                            'Savings': savings,
                            'Reduction (%)': pct
                        })
                    
                    df_above = pd.DataFrame(above_dfe_data)
                    
                    if not df_above.empty:
                        fig_above = px.bar(df_above, x='Action', y='EAD ($)', color='Action',
                            color_discrete_map={'Baseline': '#ef4444', 'Raise Utilities': '#f97316',
                                'WFP B': '#eab308', 'WFP 1st': '#3b82f6'},
                            title="Above DFE — Strategies (excl. Elevate)")
                        fig_above.update_layout(showlegend=False, height=300, yaxis_tickformat="$,.0f")
                        st.plotly_chart(fig_above, use_container_width=True)
                        
                        df_above_display = df_above.copy()
                        df_above_display['EAD ($)'] = df_above_display['EAD ($)'].apply(format_currency)
                        df_above_display['Savings'] = df_above_display['Savings'].apply(format_currency)
                        df_above_display['Reduction (%)'] = df_above_display['Reduction (%)'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(df_above_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No Above DFE data available.")
            
            st.divider()
            
            st.subheader("Damage Trajectory Over Time")
            
            df_timeline = df_agg[
                (df_agg['Scenario'] == scenario) & 
                (df_agg['Action'].isin(['Baseline', 'Raise Utilities', 'WFP B', 'Elevate']))
            ]
            
            if not df_timeline.empty:
                fig_line = px.line(df_timeline, x='TargetYear', y='Total_CumEAD_P50', color='Action',
                    markers=True, color_discrete_map={'Baseline': '#ef4444', 'Raise Utilities': '#f97316',
                        'WFP B': '#eab308', 'Elevate': '#22c55e'},
                    title=f"Cumulative EAD Projection — {occupancy_label} ({scenario} SLR Scenario)")
                fig_line.update_layout(yaxis_title="Cumulative EAD ($)", xaxis_title="Year", height=400)
                fig_line.update_yaxes(tickformat="$,.0f")
                st.plotly_chart(fig_line, use_container_width=True)
            
            if 'InFP_CumEAD_P50' in df_current.columns and 'OutFP_CumEAD_P50' in df_current.columns:
                st.subheader("Damage Distribution: Under DFE vs Above DFE")
                
                baseline_data = df_current[df_current['Action'] == 'Baseline']
                if not baseline_data.empty:
                    baseline_data = baseline_data.iloc[0]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Under DFE', 'Above DFE'],
                        values=[baseline_data['InFP_CumEAD_P50'], baseline_data['OutFP_CumEAD_P50']],
                        marker_colors=['#ef4444', '#3b82f6'],
                        hole=0.4
                    )])
                    fig_pie.update_layout(title=f"Baseline Damage Distribution — {occupancy_label} ({target_year})", height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Please upload data to view community summary.")
    
    # ========================================================================
    # TAB 3: BUILDING DETAILS
    # ========================================================================
    with tab3:
        st.markdown('<p class="tab-description">Select an individual building to view detailed damage projections across time horizons and compare adaptation options.</p>', unsafe_allow_html=True)
        
        if df_buildings is not None:
            st.subheader(f"🏠 Individual Building Analysis — {location_name} ({occupancy_label})")
            
            building_ids = df_buildings['id'].unique()
            
            selected_id = st.selectbox("Select Building ID", options=sorted(building_ids),
                format_func=lambda x: f"Building #{x}")
            
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
                
                st.subheader("Damage Trajectory")
                
                df_building_baseline = df_building[
                    (df_building['Action'] == 'Baseline') & (df_building['Scenario'] == scenario)
                ].sort_values('TargetYear')
                
                if not df_building_baseline.empty:
                    fig_building = go.Figure()
                    
                    fig_building.add_trace(go.Scatter(
                        x=df_building_baseline['TargetYear'], y=df_building_baseline['CumEAD_P50'],
                        mode='lines+markers', name='Median (P50)',
                        line=dict(color='#f97316', width=3), marker=dict(size=10)))
                    
                    fig_building.add_trace(go.Scatter(
                        x=list(df_building_baseline['TargetYear']) + list(df_building_baseline['TargetYear'])[::-1],
                        y=list(df_building_baseline['CumEAD_P95']) + list(df_building_baseline['CumEAD_P05'])[::-1],
                        fill='toself', fillcolor='rgba(249, 115, 22, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'), name='90% CI (P05-P95)'))
                    
                    fig_building.update_layout(
                        title=f"Cumulative EAD for Building #{selected_id} ({scenario} Scenario)",
                        xaxis_title="Year", yaxis_title="Cumulative EAD ($)", height=400, yaxis_tickformat="$,.0f")
                    
                    st.plotly_chart(fig_building, use_container_width=True)
                
                st.subheader("Adaptation Strategy Comparison")
                
                df_building_current = df_building[
                    (df_building['TargetYear'] == target_year) & (df_building['Scenario'] == scenario)
                ]
                
                if is_above_dfe:
                    df_building_current = df_building_current[df_building_current['Action'] != 'Elevate']
                
                if not df_building_current.empty:
                    color_map = {'Baseline': '#ef4444', 'Raise Utilities': '#f97316',
                        'WFP B': '#eab308', 'WFP 1st': '#3b82f6'}
                    if not is_above_dfe:
                        color_map['Elevate'] = '#22c55e'
                    
                    fig_actions = px.bar(df_building_current, x='Action', y='CumEAD_P50', color='Action',
                        color_discrete_map=color_map,
                        title=f"EAD by Adaptation Strategy ({target_year}, {scenario})")
                    fig_actions.update_layout(showlegend=False, height=350, yaxis_tickformat="$,.0f")
                    st.plotly_chart(fig_actions, use_container_width=True)
                    
                    baseline_val = df_building_current[df_building_current['Action'] == 'Baseline']['CumEAD_P50'].values
                    
                    if len(baseline_val) > 0:
                        baseline_val = baseline_val[0]
                        savings_data = []
                        for _, row in df_building_current.iterrows():
                            savings = baseline_val - row['CumEAD_P50']
                            pct = (savings / baseline_val * 100) if baseline_val > 0 else 0
                            savings_data.append({
                                'Action': row['Action'],
                                'EAD (P50)': format_currency(row['CumEAD_P50']),
                                'Savings': format_currency(savings),
                                'Reduction': f"{pct:.1f}%"
                            })
                        st.dataframe(pd.DataFrame(savings_data), use_container_width=True, hide_index=True)
        else:
            st.warning("Please upload Per-Building Data (CSV2) to view building details.")
    
    # ========================================================================
    # TAB 4: SCENARIO COMPARISON
    # ========================================================================
    with tab4:
        st.markdown('<p class="tab-description">Compare cumulative damage projections between Median (P50) and High-End (P90) sea level rise scenarios across all time horizons.</p>', unsafe_allow_html=True)
        
        if df_agg is not None:
            st.subheader(f"📈 Scenario Comparison — {location_name} ({occupancy_label})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Median SLR (P50)")
                df_p50 = df_agg[(df_agg['Scenario'] == 'P50') & (df_agg['Action'] == 'Baseline')].sort_values('TargetYear')
                for _, row in df_p50.iterrows():
                    st.metric(label=f"Year {int(row['TargetYear'])}", value=format_currency(row['Total_CumEAD_P50']))
            
            with col2:
                st.markdown("### High-End SLR (P90)")
                df_p90 = df_agg[(df_agg['Scenario'] == 'P90') & (df_agg['Action'] == 'Baseline')].sort_values('TargetYear')
                for _, row in df_p90.iterrows():
                    p50_val = df_p50[df_p50['TargetYear'] == row['TargetYear']]['Total_CumEAD_P50'].values
                    delta = row['Total_CumEAD_P50'] - p50_val[0] if len(p50_val) > 0 else 0
                    st.metric(label=f"Year {int(row['TargetYear'])}", value=format_currency(row['Total_CumEAD_P50']),
                        delta=f"+{format_currency(delta)} vs P50", delta_color="inverse")
            
            st.divider()
            
            df_comparison = df_agg[df_agg['Action'] == 'Baseline'].copy()
            df_comparison['Label'] = df_comparison['Scenario'].map({'P50': 'Median SLR', 'P90': 'High-End SLR'})
            
            if not df_comparison.empty:
                fig_comp = px.line(df_comparison, x='TargetYear', y='Total_CumEAD_P50', color='Label',
                    markers=True, color_discrete_map={'Median SLR': '#3b82f6', 'High-End SLR': '#ef4444'},
                    title=f"Baseline Damage: P50 vs P90 SLR Scenarios — {occupancy_label}")
                fig_comp.update_layout(height=450, yaxis_tickformat="$,.0f", yaxis_title="Cumulative EAD ($)", xaxis_title="Year")
                st.plotly_chart(fig_comp, use_container_width=True)
            
            st.subheader("📋 Full Data Table")
            
            df_display = df_agg.copy()
            for col in ['Total_CumEAD_P05', 'Total_CumEAD_P50', 'Total_CumEAD_P95', 'InFP_CumEAD_P50', 'OutFP_CumEAD_P50']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}")
            
            rename_cols = {'InFP_CumEAD_P50': 'Under_DFE_P50', 'OutFP_CumEAD_P50': 'Above_DFE_P50'}
            df_display = df_display.rename(columns={k: v for k, v in rename_cols.items() if k in df_display.columns})
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.warning("Please upload data to view comparisons.")

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
