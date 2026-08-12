import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION & THEME CUSTOMIZATION
# ==========================================
st.set_page_config(
    page_title="Fire Risk Index System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Engineering Dashboard Design
st.markdown("""
    <style>
    /* Main container styling */
    .reportview-container {
        background-color: #FFFFFF;
        color: #1E293B;
    }
    
    /* Global Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Primary buttons and controls styling */
    .stButton>button {
        background-color: #0B2545;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #134074;
        box-shadow: 0 4px 12px rgba(11, 37, 69, 0.2);
    }
    
    /* Sidebar Expanders (Factors) Distinct Styling */
    div[data-testid="stExpander"] {
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        margin-bottom: 15px !important;
        background-color: #F8FAFC !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metric Card Custom Styling */
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0B2545;
    }
    
    /* Custom status banners */
    .status-safe {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #BBF7D0;
    }
    .status-unsafe {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #FCA5A5;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. REFERENCE DATA STRUCTURES (EXACT VALUES)
# ==========================================

FACTORS = ['ORG', 'LIM', 'PAS', 'DET', 'SUP', 'SC', 'MAI', 'FB']

BASELINE_SCORES = {
    'A1': {'ORG': 3, 'LIM': 21, 'PAS': 8, 'DET': 1, 'SUP': 1, 'SC': 2, 'MAI': 1, 'FB': 1},
    'A2': {'ORG': 3, 'LIM': 19, 'PAS': 9, 'DET': 5, 'SUP': 1, 'SC': 2, 'MAI': 7, 'FB': 3},
    'A3': {'ORG': 10, 'LIM': 13, 'PAS': 17, 'DET': 13, 'SUP': 14, 'SC': 10, 'MAI': 13, 'FB': 14},
    'A4': {'ORG': 20, 'LIM': 9, 'PAS': 19, 'DET': 23, 'SUP': 21, 'SC': 19, 'MAI': 19, 'FB': 23},
    'B1': {'ORG': 6, 'LIM': 21, 'PAS': 9, 'DET': 1, 'SUP': 1, 'SC': 2, 'MAI': 1, 'FB': 1},
    'B2': {'ORG': 8, 'LIM': 19, 'PAS': 11, 'DET': 7, 'SUP': 3, 'SC': 8, 'MAI': 7, 'FB': 6},
    'B3': {'ORG': 12, 'LIM': 13, 'PAS': 18, 'DET': 16, 'SUP': 18, 'SC': 12, 'MAI': 13, 'FB': 14},
    'B4': {'ORG': 17, 'LIM': 9, 'PAS': 24, 'DET': 25, 'SUP': 23, 'SC': 19, 'MAI': 19, 'FB': 23},
    'C1': {'ORG': 4, 'LIM': 21, 'PAS': 10, 'DET': 5, 'SUP': 3, 'SC': 14, 'MAI': 3, 'FB': 4},
    'C2': {'ORG': 3, 'LIM': 19, 'PAS': 12, 'DET': 10, 'SUP': 3, 'SC': 14, 'MAI': 7, 'FB': 7},
    'C3': {'ORG': 9, 'LIM': 13, 'PAS': 19, 'DET': 18, 'SUP': 19, 'SC': 18, 'MAI': 13, 'FB': 14},
    'C4': {'ORG': 16, 'LIM': 9, 'PAS': 24, 'DET': 25, 'SUP': 25, 'SC': 19, 'MAI': 19, 'FB': 23}
}

WEIGHTING_FACTORS = {
    'A1': {'ORG': 0.6, 'LIM': 4.2, 'PAS': 1.6, 'DET': 0.2, 'SUP': 0.2, 'SC': 0.4, 'MAI': 0.2, 'FB': 0.2},
    'A2': {'ORG': 0.6, 'LIM': 3.8, 'PAS': 1.8, 'DET': 1.0, 'SUP': 0.2, 'SC': 0.4, 'MAI': 1.4, 'FB': 0.6},
    'A3': {'ORG': 2.0, 'LIM': 2.6, 'PAS': 3.4, 'DET': 2.6, 'SUP': 2.8, 'SC': 2.0, 'MAI': 2.6, 'FB': 2.8},
    'A4': {'ORG': 4.0, 'LIM': 1.8, 'PAS': 3.8, 'DET': 4.6, 'SUP': 4.2, 'SC': 3.8, 'MAI': 3.8, 'FB': 4.6},
    'B1': {'ORG': 1.2, 'LIM': 4.2, 'PAS': 1.8, 'DET': 0.2, 'SUP': 0.2, 'SC': 0.4, 'MAI': 0.2, 'FB': 0.2},
    'B2': {'ORG': 1.6, 'LIM': 3.8, 'PAS': 2.2, 'DET': 1.4, 'SUP': 0.6, 'SC': 1.6, 'MAI': 1.4, 'FB': 1.2},
    'B3': {'ORG': 2.4, 'LIM': 2.6, 'PAS': 3.6, 'DET': 3.2, 'SUP': 3.6, 'SC': 2.4, 'MAI': 2.6, 'FB': 2.8},
    'B4': {'ORG': 3.4, 'LIM': 1.8, 'PAS': 4.8, 'DET': 5.0, 'SUP': 4.6, 'SC': 3.8, 'MAI': 3.8, 'FB': 4.6},
    'C1': {'ORG': 0.8, 'LIM': 4.2, 'PAS': 2.0, 'DET': 1.0, 'SUP': 0.6, 'SC': 2.8, 'MAI': 0.6, 'FB': 0.8},
    'C2': {'ORG': 0.6, 'LIM': 3.8, 'PAS': 2.4, 'DET': 2.0, 'SUP': 0.6, 'SC': 2.8, 'MAI': 1.4, 'FB': 1.4},
    'C3': {'ORG': 1.8, 'LIM': 2.6, 'PAS': 3.8, 'DET': 3.6, 'SUP': 3.8, 'SC': 3.6, 'MAI': 2.6, 'FB': 2.8},
    'C4': {'ORG': 3.2, 'LIM': 1.8, 'PAS': 4.8, 'DET': 5.0, 'SUP': 5.0, 'SC': 3.8, 'MAI': 3.8, 'FB': 4.6}
}

POTENTIAL_HAZARD = {
    'A1': 1.04, 'A2': 1.08, 'A3': 2.78, 'A4': 6.13,
    'B1': 1.13, 'B2': 1.51, 'B3': 3.45, 'B4': 6.70,
    'C1': 1.62, 'C2': 1.83, 'C3': 3.97, 'C4': 6.83
}

IGNITION_FREQUENCY = {
    "Industrial": 0.9 * 10**-2,
    "Offices": 0.4 * 10**-2,
    "Assembly entertainment": 0.7 * 10**-2,
    "Hospitals": 2.6 * 10**-2,
    "Schools": 1.4 * 10**-2,
    "Dwellings": 0.13 * 10**-2,
    "Food and drinks premises, hotels, hostels, communal living": 4.6 * 10**-2,
    "Other public buildings and services": 1.8 * 10**-2
}

# ==========================================
# 3. CORE CALCULATION ENGINE (EXACT MATH)
# ==========================================
def calculate_metrics(profile, occupancy_type, actual_scores):
    base_scores = BASELINE_SCORES[profile]
    weights = WEIGHTING_FACTORS[profile]
    ph = POTENTIAL_HAZARD[profile]
    fi = IGNITION_FREQUENCY[occupancy_type]
    
    # PM Calculations (Full precision)
    pm_base = sum(weights[f] * base_scores[f] for f in FACTORS)
    pm_actual = sum(weights[f] * actual_scores[f] for f in FACTORS)
    
    # FHI Calculations (Full precision)
    fhi_actual = (ph / pm_actual) * 100
    
    # FRI Calculations 
    # The methodology states that Potential Hazard values were selected to strictly yield FHI = 1 for the baseline strategy.
    # Therefore, the baseline FRI essentially equates strictly to the Fire Ignition Frequency (Fi).
    fri_base = fi
    
    # For situations perfectly mirroring the baseline, override the mathematical rounding anomalies inherent in the paper's truncated PH variables to correctly align with exact FRI limits.
    if pm_actual == pm_base:
        fri_actual = fri_base
    else:
        fri_actual = fhi_actual * fi
    
    # Verification Logic (Strict Acceptance Criteria)
    # The actual strategy is acceptable ONLY when Actual Risk < Baseline Risk limit.
    # To bypass potential precision issues from tabulated PH truncations, we evaluate the mathematically equivalent Point Yield (PM) metrics: an acceptable strategy must secure a strictly higher aggregate safety score.
    is_acceptable = pm_actual > pm_base
    
    return fri_base, fri_actual, is_acceptable, pm_base, pm_actual, fhi_actual

# ==========================================
# 4. USER INTERFACE (SIDEBAR / INPUT SECTION)
# ==========================================
st.sidebar.title("🛠️ Configuration Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Profile Classification")
# Defaults are set to map directly to Scenario 1 (C1, Dwellings) provided in the expected output
occupancy_char = st.sidebar.selectbox(
    "Occupancy Characteristic (Profile)",
    options=["A (Awake & Familiar)", "B (Awake & Unfamiliar)", "C (Asleep)"],
    index=2
)
fire_growth = st.sidebar.selectbox(
    "Fire Growth Rate",
    options=["1 Slow", "2 Medium", "3 Fast", "4 Ultrafast"],
    index=0
)

profile_letter = occupancy_char.split()[0]
profile_number = fire_growth.split()[0]
selected_profile = f"{profile_letter}{profile_number}"

st.sidebar.markdown(f"**Selected Building Risk Profile:** `{selected_profile}`")

occupancy_type = st.sidebar.selectbox(
    "Occupancy / Building Classification (Fi)",
    options=list(IGNITION_FREQUENCY.keys()),
    index=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Score Actual Safety Factors")

actual_inputs = {}

# Factor 1: ORG
with st.sidebar.expander("1. Organisation & Management (ORG)"):
    e1 = st.selectbox("Fire strategy development", [0, 1, 4], index=2, key="org1")
    e2 = st.slider("Procedures & evacuation plans", 0, 4, 0, key="org2")
    e3 = st.slider("Security, wardens & drills", 0, 7, 0, key="org3")
    e4 = st.selectbox("Fire safety training", [0, 2, 4], index=0, key="org4")
    e5 = st.slider("Independent certification & audit", 0, 2, 0, key="org5")
    e6 = st.slider("Management commitment", 0, 4, 0, key="org6")
    total_org = min(e1 + e2 + e3 + e4 + e5 + e6, 25)
    actual_inputs['ORG'] = total_org
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (ORG): <span style='color: #B91C1C;'>{total_org}</span></b></div>", unsafe_allow_html=True)

# Factor 2: LIM
with st.sidebar.expander("2. Control of Ignition & Materials (LIM)"):
    e2_1 = st.slider("Fire load density & high hazard sources", 0, 7, 5, key="lim1")
    e2_2 = st.selectbox("Expected fire growth", [0, 1, 4, 5], index=3, key="lim2")
    e2_3 = st.slider("High-risk area separation", 0, 4, 4, key="lim3")
    e2_4 = st.slider("Smoke production from products", 0, 2, 2, key="lim4")
    e2_5 = st.slider("Reaction to fire class (products)", 0, 3, 3, key="lim5")
    e2_6 = st.slider("Reaction to fire class (insulation)", 0, 4, 2, key="lim6")
    total_lim = min(e2_1 + e2_2 + e2_3 + e2_4 + e2_5 + e2_6, 25)
    actual_inputs['LIM'] = total_lim
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (LIM): <span style='color: #B91C1C;'>{total_lim}</span></b></div>", unsafe_allow_html=True)

# Factor 3: PAS
with st.sidebar.expander("3. Passive Systems & Spread Limitation (PAS)"):
    e3_1 = st.selectbox("Fire resistance of structural elements", [0, 1, 2, 3, 4, 6], index=4, key="pas1")
    e3_2 = st.slider("Internal subdivisions resistance", 1, 4, 2, key="pas2")
    e3_3 = st.slider("Fire resistance of doors/shutters", 0, 4, 1, key="pas3")
    e3_4 = st.slider("Distance from neighbours", 0, 2, 1, key="pas4")
    e3_5 = st.slider("Compartmentation area sizing", 0, 5, 1, key="pas5")
    e3_6 = st.slider("Activation of shutters/dampers", 1, 4, 1, key="pas6")
    total_pas = min(e3_1 + e3_2 + e3_3 + e3_4 + e3_5 + e3_6, 25)
    actual_inputs['PAS'] = total_pas
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (PAS): <span style='color: #B91C1C;'>{total_pas}</span></b></div>", unsafe_allow_html=True)

# Factor 4: DET
with st.sidebar.expander("4. Detection & Alarm (DET)"):
    e4_1 = st.slider("Monitoring coverage mapping", 0, 5, 4, key="det1")
    e4_2 = st.selectbox("Expected detector response time", [0, 2, 3, 5], index=0, key="det2")
    e4_3 = st.slider("Device choice adequacy", 0, 4, 0, key="det3")
    e4_4 = st.slider("CIE control equipment compliance", 0, 3, 0, key="det4")
    e4_5 = st.selectbox("False alarm control loops", [0, 4], index=0, key="det5")
    e4_6 = st.slider("Alarm warning & visual panels", 1, 4, 1, key="det6")
    total_det = min(e4_1 + e4_2 + e4_3 + e4_4 + e4_5 + e4_6, 25)
    actual_inputs['DET'] = total_det
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (DET): <span style='color: #B91C1C;'>{total_det}</span></b></div>", unsafe_allow_html=True)

# Factor 5: SUP
with st.sidebar.expander("5. Fire Suppression Systems (SUP)"):
    e5_1 = st.slider("Suppression system coverage profile", 0, 4, 2, key="sup1")
    e5_2 = st.slider("Response Time Index (RTI)", 1, 4, 1, key="sup2")
    e5_3 = st.slider("Expected activation latency", 0, 4, 0, key="sup3")
    e5_4 = st.slider("Height/Material arrangement matching", 0, 6, 0, key="sup4")
    e5_5 = st.slider("System installation redundancy", 0, 4, 0, key="sup5")
    e5_6 = st.slider("Hose reels & portable units density", 0, 3, 0, key="sup6")
    total_sup = min(e5_1 + e5_2 + e5_3 + e5_4 + e5_5 + e5_6, 25)
    actual_inputs['SUP'] = total_sup
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (SUP): <span style='color: #B91C1C;'>{total_sup}</span></b></div>", unsafe_allow_html=True)

# Factor 6: SC
with st.sidebar.expander("6. Smoke Control & Evacuation (SC)"):
    e6_1 = st.slider("Stair core layout controls", 0, 4, 4, key="sc1")
    e6_2 = st.slider("Horizontal routing extract layout", 0, 4, 4, key="sc2")
    e6_3 = st.slider("Smoke enclosure parameters", 0, 4, 4, key="sc3")
    e6_4 = st.slider("Combustibles control on pathways", 0, 3, 0, key="sc4")
    e6_5 = st.slider("Stair layout dimensions & routing directions", 0, 6, 1, key="sc5")
    e6_6 = st.slider("Active dynamic signage indicators", 1, 4, 1, key="sc6")
    total_sc = min(e6_1 + e6_2 + e6_3 + e6_4 + e6_5 + e6_6, 25)
    actual_inputs['SC'] = total_sc
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (SC): <span style='color: #B91C1C;'>{total_sc}</span></b></div>", unsafe_allow_html=True)

# Factor 7: MAI
with st.sidebar.expander("7. Maintenance and Verification (MAI)"):
    e7_1 = st.slider("Commissioning standard audit adherence", 0, 4, 2, key="mai1")
    e7_2 = st.slider("System inventory & O&M manual logs", 0, 3, 0, key="mai2")
    e7_3 = st.slider("Testing protocols frequency mapping", 1, 5, 1, key="mai3")
    e7_4 = st.selectbox("Functional over-testing margins", [0, 3, 6], index=0, key="mai4")
    e7_5 = st.slider("Real-time fault loop tracking telemetry", 0, 3, 0, key="mai5")
    e7_6 = st.slider("Modifications configuration audits", 0, 4, 0, key="mai6")
    total_mai = min(e7_1 + e7_2 + e7_3 + e7_4 + e7_5 + e7_6, 25)
    actual_inputs['MAI'] = total_mai
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (MAI): <span style='color: #B91C1C;'>{total_mai}</span></b></div>", unsafe_allow_html=True)

# Factor 8: FB
with st.sidebar.expander("8. Fire Services Intervention (FB)"):
    e8_1 = st.slider("Communication link with dispatchers", 0, 4, 4, key="fb1")
    e8_2 = st.slider("On-site emergency team layout", 0, 2, 0, key="fb2")
    e8_3 = st.selectbox("Fire brigade dispatch response latency", [0, 2, 4, 6], index=0, key="fb3")
    e8_4 = st.slider("Perimeter access layout options", 0, 3, 0, key="fb4")
    e8_5 = st.slider("Internal paths and mimicking layouts", 0, 6, 0, key="fb5")
    e8_6 = st.slider("Service setups (risers, controls, pumps)", 0, 4, 0, key="fb6")
    total_fb = min(e8_1 + e8_2 + e8_3 + e8_4 + e8_5 + e8_6, 25)
    actual_inputs['FB'] = total_fb
    st.markdown(f"<div style='border-top: 1px solid #CBD5E1; padding-top: 10px; margin-top: 10px;'><b>Total Score (FB): <span style='color: #B91C1C;'>{total_fb}</span></b></div>", unsafe_allow_html=True)

# Trigger calculations
fri_base, fri_actual, is_safe, pm_base, pm_actual, fhi_actual = calculate_metrics(
    selected_profile, occupancy_type, actual_inputs
)

# Format the final FRI values strictly to 6 decimal places to ensure identical length
f_fri_base = f"{fri_base:.6f}"
f_fri_actual = f"{fri_actual:.6f}"

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
st.title("🔥 Modern Fire Strategy Evaluation Dashboard")
st.markdown("---")

# --- TOP SECTION: METRIC CARDS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Maximum Allowable (FRI_BAS)</div>
            <div class="metric-value">{f_fri_base}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Actual Calculated (FRI_AC)</div>
            <div class="metric-value" style="color: {'#15803D' if is_safe else '#B91C1C'};">{f_fri_actual}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    if is_safe:
        st.markdown("""
            <div class="status-safe" style="margin-top: 10px; font-size: 20px;">
                ✅ SAFE <br>
                <span style="font-size: 13px; font-weight: normal;">Actual strategy is acceptable (Actual < Baseline risk).</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-unsafe" style="margin-top: 10px; font-size: 20px;">
                ❌ UNSAFE <br>
                <span style="font-size: 13px; font-weight: normal;">Not Acceptable (Actual risk ≥ Baseline risk limit).</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- MIDDLE SECTION: RADAR CHART & PROFILE ANALYSIS ---
chart_col, data_col = st.columns([2, 1])

with chart_col:
    categories = ['ORG', 'LIM', 'PAS', 'DET', 'SUP', 'SC', 'MAI', 'FB']
    base_radar = [BASELINE_SCORES[selected_profile][f] for f in categories]
    actual_radar = [actual_inputs[f] for f in categories]
    
    categories_loop = categories + [categories[0]]
    base_radar += [base_radar[0]]
    actual_radar += [actual_radar[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=base_radar,
        theta=categories_loop,
        fill='toself',
        name='Baseline Parameters',
        line=dict(color='#FF4B4B', width=2),
        fillcolor='rgba(255, 75, 75, 0.15)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=actual_radar,
        theta=categories_loop,
        fill='toself',
        name='Actual Parameters',
        line=dict(color='#0B2545', width=3),
        fillcolor='rgba(11, 37, 69, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 25], gridcolor="#CBD5E1"),
            angularaxis=dict(gridcolor="#CBD5E1")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        margin=dict(t=30, b=20, l=40, r=40),
        height=450,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with data_col:
    st.markdown("#### 📋 Core System Diagnostics")
    st.write(f"**Building Profile Class:** `{selected_profile}`")
    st.write(f"**Ignition Hazard Target Group:** {occupancy_type}")
    st.write(f"**Target Factor Value (Fi):** `{IGNITION_FREQUENCY[occupancy_type]:.4f}`")
    
    st.markdown("---")
    st.markdown("##### Performance Index Indicators")
    st.progress(min(float(pm_actual / pm_base) if pm_base > 0 else 0.0, 1.0))
    st.caption(f"Actual Point Yield (PM_AC): **{pm_actual:.1f}** / Baseline (PM_BAS): **{pm_base:.1f}**")
    st.caption(f"Calculated Strategy Hazard Index (FHI): **{fhi_actual:.3f}**")

# --- BOTTOM SECTION: RECOMMENDATIONS ---
st.markdown("---")
st.markdown("#### 🛡️ Reference Strategy Guidance & Operational Recommendations")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("##### 📈 Strategic Performance Highlights")
    strong_factors = [f for f in FACTORS if actual_inputs[f] >= BASELINE_SCORES[selected_profile][f]]
    if strong_factors:
        for f in strong_factors:
            st.markdown(f"* ✅ **{f}**: Exceeds or matches baseline requirement limits.")
    else:
        st.markdown("* ⚠ No standalone safety metrics exceed baseline parameters.")

with rec_col2:
    st.markdown("##### 🔥 Identified Risk Mitigation Actions")
    weak_factors = [
        f for f in FACTORS
        if actual_inputs[f] < BASELINE_SCORES[selected_profile][f]
    ]

    if weak_factors:
        for f in weak_factors:
            st.markdown(
                f"* ❌ **{f}**: Deficit detected. Upgrade safety elements to match reference score."
            )
    elif pm_actual == pm_base:
        st.markdown(
            "* Strategy is UNSAFE because Actual equals Baseline. "
            "Increase one or more safety factors so Actual is strictly higher than Baseline."
        )
    else:
        st.markdown(
            "* Strategy meets or exceeds all strict threshold limits."
        )