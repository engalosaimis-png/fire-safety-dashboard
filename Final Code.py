"""Unified fire-safety decision-support dashboard."""
from __future__ import annotations
import ast
import html
import types
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# التعديل هنا: قراءة الملفات من نفس مجلد المشروع مباشرة لتناسب الاستضافة
FRI_SOURCE = Path("FRI.py")
QRA_SOURCE = Path("risk matrix.py")
DISCOUNT_RATE = 0.035
P_FOD: dict[str, float] = {
    "No alarm system": 1.00,
    "Battery-powered alarms": 0.23,
    "Hardwired (mains) alarms": 0.08,
}
DAMAGE_FACTORS: dict[int, float] = {1: 0.01, 2: 0.10, 3: 0.25, 4: 0.80}

ZONE_COLOR_OVERRIDE: dict[str, str] = {
    "High": "#dc2626",        # أحمر
    "Moderate": "#f97316",    # برتقالي
    "Low": "#eab308",         # أصفر
    "Negligible": "#16a34a",  # أخضر
}

def _is_definition(node: ast.AST) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        def upper_case_target(target: ast.expr) -> bool:
            if isinstance(target, ast.Name):
                return target.id.isupper()
            if isinstance(target, (ast.Tuple, ast.List)):
                return all(upper_case_target(item) for item in target.elts)
            return False
        return all(upper_case_target(target) for target in targets)
    return False

@st.cache_resource(show_spinner=False)
def load_core_module(source_path: str, module_name: str) -> types.ModuleType:
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required source module was not found: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [node for node in tree.body if _is_definition(node)]
    module_tree = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module_tree)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(module_tree, str(path), "exec"), module.__dict__)
    return module

def calculate_financial_appraisal(
    lam: float, probabilities: dict[int, float], building_value: float,
    project_life: int, installation_cost: float, annual_maintenance_cost: float,
    current_alarm: str, proposed_alarm: str
) -> dict[str, float]:
    expected_loss = sum(float(probabilities.get(rank, 0.0)) * factor for rank, factor in DAMAGE_FACTORS.items()) * building_value * 1.10
    current_annual_risk = lam * P_FOD[current_alarm] * expected_loss
    proposed_annual_risk = lam * P_FOD[proposed_alarm] * expected_loss
    annual_benefit = current_annual_risk - proposed_annual_risk
    present_value_net_benefits = sum((annual_benefit - annual_maintenance_cost) / ((1 + DISCOUNT_RATE) ** year) for year in range(1, project_life + 1))
    npv = present_value_net_benefits - installation_cost
    return {
        "expected_loss": expected_loss, "current_annual_risk": current_annual_risk,
        "proposed_annual_risk": proposed_annual_risk, "annual_benefit": annual_benefit,
        "present_value_net_benefits": present_value_net_benefits, "npv": npv
    }

def format_currency(value: float) -> str:
    return f"£{value:,.2f}"

def build_radar_chart(fri: types.ModuleType, profile: str, actual_scores: dict[str, int]) -> go.Figure:
    factors = list(fri.FACTORS)
    baseline = [fri.BASELINE_SCORES[profile][factor] for factor in factors]
    actual = [actual_scores[factor] for factor in factors]
    closed_factors = factors + [factors[0]]
    figure = go.Figure()
    figure.add_trace(go.Scatterpolar(r=baseline + [baseline[0]], theta=closed_factors, fill="toself", name="Baseline compliance", line={"color": "#dc2626", "width": 2}, fillcolor="rgba(220, 38, 38, 0.12)"))
    figure.add_trace(go.Scatterpolar(r=actual + [actual[0]], theta=closed_factors, fill="toself", name="Current assessment", line={"color": "#0f3d5e", "width": 3}, fillcolor="rgba(15, 61, 94, 0.28)"))
    figure.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 25], "gridcolor": "#cbd5e1"}, "angularaxis": {"gridcolor": "#cbd5e1"}}, legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "center", "x": 0.5}, margin={"t": 45, "b": 20, "l": 35, "r": 35}, height=430, template="plotly_white")
    return figure

def select_default(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0

def synchronise_alarm_configuration(qra_alarm_type: str) -> tuple[str, int, int]:
    alarm = str(qra_alarm_type).strip().casefold()
    if "battery" in alarm: return "Battery-powered alarms", 1, 7
    if "mains" in alarm or "hardwired" in alarm: return "Hardwired (mains) alarms", 8, 25
    return "No alarm system", 0, 0

def render_metric_card(label: str, value: str, tone: str = "navy", accent_color: str | None = None) -> None:
    colors = {"navy": "#0b2545", "green": "#166534", "red": "#b91c1c", "amber": "#92400e"}
    color = accent_color or colors[tone]
    safe_label, safe_value = html.escape(label), html.escape(value)
    st.markdown(f"<div class='metric-card {tone}' style='border-top-color:{color};'><div class='metric-label'>{safe_label}</div><div class='metric-value' style='color:{color}'>{safe_value}</div></div>", unsafe_allow_html=True)

def main() -> None:
    st.set_page_config(page_title="Unified Fire Safety Decision Support", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .block-container {max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem;}
        .metric-card {background:#f8fafc; border:1px solid #e2e8f0; border-top:4px solid #0b2545; border-radius:12px; padding:18px; min-height:104px; box-shadow:0 2px 6px rgba(15,23,42,.06)}
        .metric-card.green {border-top-color:#16a34a}.metric-card.red {border-top-color:#dc2626}
        .metric-card.amber {border-top-color:#d97706}.metric-label {font-size:.76rem; letter-spacing:.06em; color:#64748b; font-weight:700; text-transform:uppercase}.metric-value {font-size:1.38rem; font-weight:750; margin-top:.5rem}
        .verdict {border-radius:12px; padding:18px 20px; font-size:1rem; font-weight:650; margin-top:22px;}
        .verdict-ok {background:#dcfce7; border:1px solid #86efac; color:#166534}
        .verdict-no {background:#fee2e2; border:1px solid #fecaca; color:#b91c1c}
        .verdict-no span.verdict-note {display:block; margin-top:8px}
        </style>
        """, unsafe_allow_html=True)

    st.title("🔥 Unified Fire Safety Decision-Support Dashboard")
    st.caption("Deterministic BS 9999 FRI benchmark · actuarial frequency (λ) · Random Forest severity prediction · ALARP financial appraisal")

    try:
        fri = load_core_module(str(FRI_SOURCE), "fri_original_core")
        qra = load_core_module(str(QRA_SOURCE), "qra_original_core")
        if hasattr(qra, "ZONE_COLORS"):
            qra.ZONE_COLORS.update(ZONE_COLOR_OVERRIDE)
    except Exception as error:
        st.error(f"The supplied calculation engine could not be loaded: {error}")
        return

    full_fire_df, train_fire_df, test_fire_df, exposure_df, errors = qra.try_load_data()
    if errors:
        st.error("The QRA data could not be loaded, so the predictive and financial views are unavailable.")
        for item in errors: st.caption(f"• {item}")
        return

    assert full_fire_df is not None and train_fire_df is not None
    assert test_fire_df is not None and exposure_df is not None

    try:
        detected_year = qra.find_year_column(full_fire_df)
        if detected_year is None: raise ValueError("No usable incident year/date column was found in the full fire data.")
        frequency_fire_df = qra.filter_frequency_period(qra.attach_year_column(full_fire_df, detected_year))
        clean_train = qra.prepare_model_features(train_fire_df)
        options: dict[str, list[str]] = {column: sorted(clean_train[column].astype(str).unique().tolist()) for column in qra.INSPECTOR_INPUT_COLUMNS}
    except Exception as error:
        st.error(f"The QRA data preparation failed: {error}")
        return

    st.sidebar.header("Inspection configuration")
    st.sidebar.caption("QRA alarm selection automatically controls FRI DET and the financial current alarm.")

    with st.sidebar.expander("1. Predictive QRA assessment", expanded=True):
        st.caption("Options are derived only from the supplied training data.")
        qra_inputs = {
            "OCCUPANCY_TYPE": st.selectbox("Occupancy type", options["OCCUPANCY_TYPE"]),
            "DWELLING_TYPE": st.selectbox("Dwelling type", options["DWELLING_TYPE"]),
            "BUILDING_SPECIAL_CONSTRUCTION": st.selectbox("Building construction", options["BUILDING_SPECIAL_CONSTRUCTION"]),
            "ALARM_SYSTEM_TYPE": st.selectbox("Installed alarm system", options["ALARM_SYSTEM_TYPE"]),
            "FRS_TERRITORY": st.selectbox("FRS territory", options["FRS_TERRITORY"]),
            "DAY_NIGHT": st.selectbox("Time of day", options["DAY_NIGHT"]),
            "OCCUPIED_NORMAL": st.selectbox("Normally occupied", options["OCCUPIED_NORMAL"]),
            "WEEKDAY_WEEKEND": st.selectbox("Weekday / weekend", options["WEEKDAY_WEEKEND"]),
        }

    current_alarm, det_min, det_max = synchronise_alarm_configuration(qra_inputs["ALARM_SYSTEM_TYPE"])

    with st.sidebar.expander("2. Deterministic FRI assessment", expanded=True):
        occupancy_characteristic = st.selectbox("Occupancy characteristic", ["A (Awake & Familiar)", "B (Awake & Unfamiliar)", "C (Asleep)"], index=2)
        fire_growth = st.selectbox("Fire growth rate", ["1 Slow", "2 Medium", "3 Fast", "4 Ultrafast"], index=0)
        fri_occupancy = st.selectbox("FRI building classification (Fi)", list(fri.IGNITION_FREQUENCY.keys()), index=select_default(list(fri.IGNITION_FREQUENCY), "Dwellings"))
        profile = f"{occupancy_characteristic.split()[0]}{fire_growth.split()[0]}"
        st.caption(f"Building risk profile: **{profile}**")
        st.caption("Score each factor from 0 (no provision) to 25 (maximum provision).")
        actual_scores: dict[str, int] = {}
        for factor in fri.FACTORS:
            if factor != "DET":
                actual_scores[factor] = st.slider(factor, min_value=0, max_value=25, value=int(fri.BASELINE_SCORES[profile][factor]), key=f"fri_score_{factor}")
                continue
            if det_min == det_max:
                actual_scores["DET"] = 0
                st.number_input("DET (Detection & Alarm)", min_value=0, max_value=25, value=0, step=1, disabled=True, key="fri_det_locked")
                st.caption("Locked at 0: QRA records no installed alarm, so DET cannot receive a detection/alarm score.")
                continue
            det_key = "fri_score_DET"
            default_det = min(max(int(fri.BASELINE_SCORES[profile]["DET"]), det_min), det_max)
            current_det = int(st.session_state.get(det_key, default_det))
            st.session_state[det_key] = min(max(current_det, det_min), det_max)
            actual_scores["DET"] = st.slider("DET (Detection & Alarm)", min_value=det_min, max_value=det_max, key=det_key)
            st.caption(f"QRA alarm: {qra_inputs['ALARM_SYSTEM_TYPE']} — DET constrained to {det_min}–{det_max}.")

    with st.sidebar.expander("3. Economic configuration", expanded=True):
        building_value = st.number_input("Building value (GBP £)", min_value=0.0, value=500_000.0, step=25_000.0, format="%.2f")
        project_life = st.slider("Project life (years)", min_value=10, max_value=30, value=20)
        installation_cost = st.number_input("Installation cost (GBP £)", min_value=0.0, value=5_000.0, step=500.0, format="%.2f")
        annual_maintenance_cost = st.number_input("Annual maintenance cost (GBP £)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
        st.text_input("Current alarm arrangement (from QRA)", value=current_alarm, disabled=True)
        proposed_alarm_options = [alarm for alarm in P_FOD if alarm != "No alarm system"]
        proposed_alarm = st.selectbox("Proposed alarm arrangement", proposed_alarm_options, index=select_default(proposed_alarm_options, "Hardwired (mains) alarms"))

    try:
        fri_base, fri_actual, is_safe, pm_base, pm_actual, fhi_actual = fri.calculate_metrics(profile, fri_occupancy, actual_scores)
        with st.spinner("Running the supplied Random Forest severity model..."):
            model, feature_columns, active_alarm_reference, class_priors, metrics = qra.train_severity_model(train_fire_df, test_fire_df)
        lam, total_fires, total_exposure = qra.compute_frequency(frequency_fire_df, exposure_df, qra_inputs["OCCUPANCY_TYPE"])
        frequency_bin = qra.classify_frequency_bin(lam)
        predicted_rank, proba_by_rank, safety_override = qra.predict_severity(model, feature_columns, active_alarm_reference, class_priors, qra_inputs)
        severity_bin = qra.RANK_TO_LABEL[predicted_rank]
        risk_zone = qra.RISK_ZONE_MAP[(frequency_bin, severity_bin)]
        appraisal = calculate_financial_appraisal(lam, proba_by_rank, building_value, project_life, installation_cost, annual_maintenance_cost, current_alarm, proposed_alarm)
    except Exception as error:
        st.error(f"Assessment could not be completed: {error}")
        return

    tab_decision, tab_details = st.tabs(["Decision Support", "Technical Details"])

    with tab_decision:
        st.subheader("Current compliance and risk status")
        card_1, card_2, card_3, card_4 = st.columns(4)
        with card_1: render_metric_card("FRI baseline limit", f"{fri_base:.6f}")
        with card_2: render_metric_card("Current FRI", f"{fri_actual:.6f}", "green" if is_safe else "red")
        with card_3: render_metric_card("Compliance status", "SAFE" if is_safe else "UNSAFE", "green" if is_safe else "red")
        with card_4: render_metric_card("QRA risk zone", risk_zone, accent_color=ZONE_COLOR_OVERRIDE.get(risk_zone, "#0b2545"))
        st.caption(f"FRI point yield: current {pm_actual:.1f} / baseline {pm_base:.1f}; FHI {fhi_actual:.3f}. The supplied FRI engine accepts only a strictly higher point yield.")
        if not is_safe: st.error("FRI remains UNSAFE: the economic ALARP result does not change any FRI factor score or turn this assessment into SAFE. Address the FRI factor deficits separately.")

        st.divider()
        radar_column, matrix_column = st.columns(2, gap="large")
        with radar_column:
            st.markdown("#### Deterministic FRI: current versus baseline")
            st.plotly_chart(build_radar_chart(fri, profile, actual_scores), width="stretch")
        with matrix_column:
            st.markdown("#### Predictive QRA: frequency versus severity")
            matrix_figure = qra.plot_matrix(frequency_bin, severity_bin, lam, predicted_rank)
            st.pyplot(matrix_figure, width="stretch")
            plt.close(matrix_figure)
            st.caption(f"λ = {lam:.7f} ({frequency_bin}); predicted severity: {severity_bin}. Actuarial basis: {total_fires:,} fires / {total_exposure:,.0f} dwelling-years.")
            if safety_override: st.info("The original QRA engineering no-alarm safety constraint was applied to this prediction.")

        st.divider()
        st.subheader("Financial report and ALARP verdict")
        finance_1, finance_2, finance_3, finance_4 = st.columns(4)
        with finance_1: render_metric_card("Expected loss per event", format_currency(appraisal["expected_loss"]))
        with finance_2: render_metric_card("Annual risk avoided", format_currency(appraisal["annual_benefit"]), "green" if appraisal["annual_benefit"] > 0 else "amber")
        with finance_3: render_metric_card("PV net benefit", format_currency(appraisal["present_value_net_benefits"]))
        with finance_4: render_metric_card("Net present value", format_currency(appraisal["npv"]), "green" if appraisal["npv"] > 0 else "red")

        if appraisal["npv"] > 0:
            st.markdown("<div class='verdict verdict-ok'>✅ Upgrade Recommended: The investment is economically viable and supports property protection goals under ALARP.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='verdict verdict-no'>❌ Upgrade Not Justified: The financial benefits do not exceed the investment cost.<br><span class='verdict-note' style='font-size:.86rem;font-weight:500;'>This financial result does not alter FRI factor scores or the FRI SAFE/UNSAFE status.</span></div>", unsafe_allow_html=True)

    financial_breakdown = pd.DataFrame([
        ("Building value", format_currency(building_value)), ("Project life", f"{project_life} years"),
        ("Installation cost", format_currency(installation_cost)), ("Annual maintenance cost", format_currency(annual_maintenance_cost)),
        ("Current P_fod", f"{P_FOD[current_alarm]:.2f} ({current_alarm})"), ("Proposed P_fod", f"{P_FOD[proposed_alarm]:.2f} ({proposed_alarm})"),
        ("Annual risk — current", format_currency(appraisal["current_annual_risk"])), ("Annual risk — proposed", format_currency(appraisal["proposed_annual_risk"])),
        ("Annual benefit / risk avoided", format_currency(appraisal["annual_benefit"])), ("PV net benefit", format_currency(appraisal["present_value_net_benefits"])),
        ("Discount rate", "3.5%"),
    ], columns=["Financial input / calculation", "Value"])

    strong_factors = [factor for factor in fri.FACTORS if actual_scores[factor] >= fri.BASELINE_SCORES[profile][factor]]
    weak_factors = [factor for factor in fri.FACTORS if actual_scores[factor] < fri.BASELINE_SCORES[profile][factor]]
    probability_fig = qra.plot_severity_probabilities(proba_by_rank, predicted_rank)
    probability_uri = qra.fig_to_data_uri(probability_fig)

    with tab_details:
        st.subheader("1. Severity Probabilities")
        st.pyplot(probability_fig, width="stretch")
        plt.close(probability_fig)
        st.divider()
        st.subheader("2. Reference Strategy")
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            st.markdown("##### Compliant / Strong Factors")
            if strong_factors:
                for factor in strong_factors: st.success(f"✓ **{factor}**: Score {actual_scores[factor]} (Baseline: {fri.BASELINE_SCORES[profile][factor]})")
            else: st.info("No factors currently meet or exceed baseline limits.")
        with ref_col2:
            st.markdown("##### Deficient / Weak Factors")
            if weak_factors:
                for factor in weak_factors: st.error(f"✗ **{factor}**: Score {actual_scores[factor]} (Baseline: {fri.BASELINE_SCORES[profile][factor]})")
            else: st.success("All factors meet or exceed baseline limits.")
        st.divider()
        st.subheader("3. Financial Inputs & Calculations")
        st.dataframe(financial_breakdown, width="stretch", hide_index=True)

if __name__ == "__main__":
    main()