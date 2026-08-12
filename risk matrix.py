"""UK Residential Fire Safety Risk Assessment Matrix"""
from __future__ import annotations
import base64
import io
import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# التعديل الأهم: جعل المسارات ديناميكية لتعمل على السيرفر (Streamlit Cloud)
DATA_DIR = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
FULL_FIRE_CSV_PATH = os.path.join(DATA_DIR, "Full_Processed_Fire_Data.csv")
TRAIN_FIRE_CSV_PATH = os.path.join(DATA_DIR, "Training_Set_Fire_Data.csv")
TEST_FIRE_CSV_PATH = os.path.join(DATA_DIR, "Testing_Set_Fire_Data.csv")
EXPOSURE_XLSX_PATH = os.path.join(DATA_DIR, "تقسيم المساكن.xlsx")

YEAR_START, YEAR_END = 2017, 2023
RANDOM_STATE = 42
FREQUENCY_FIRE_END_YEAR_START = 2018
FREQUENCY_FIRE_END_YEAR_END = 2023
PRIOR_CALIBRATION_EXPONENT = 0.50

INSPECTOR_INPUT_COLUMNS = [
    "OCCUPANCY_TYPE", "DWELLING_TYPE", "BUILDING_SPECIAL_CONSTRUCTION",
    "ALARM_SYSTEM_TYPE", "FRS_TERRITORY", "DAY_NIGHT", "OCCUPIED_NORMAL", "WEEKDAY_WEEKEND",
]
TARGET_COLUMN = "SPREAD_OF_FIRE"

RANK_TO_LABEL = {1: "Negligible", 2: "Marginal", 3: "Critical", 4: "Catastrophic"}
SPREAD_TEXT_TO_RANK = {"item": 1, "confined to item": 1, "room": 2, "confined to room": 2, "floor": 3, "confined to floor": 3, "building": 4, "confined to building": 4, "beyond building": 4, "beyond": 4}
UNKNOWN_CATEGORY_LABELS = {"", "not known", "unknown", "not-known", "not_known", "n/a", "na", "null", "nan"}
CUSTOM_NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]

def _file_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

def _normalise_column_name(column: Any) -> str:
    return str(column).replace("\ufeff", "").replace("\xa0", " ").strip()

def _normalise_category(series: pd.Series) -> pd.Series:
    clean = series.astype("string").str.strip()
    unknown_mask = clean.str.casefold().isin(UNKNOWN_CATEGORY_LABELS)
    return clean.mask(unknown_mask, "Unknown").fillna("Unknown")

def redistribute_unknown_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns: return df
    result = df.copy()
    values = result[column].astype("string").str.strip()
    unknown_mask = values.str.casefold().isin(UNKNOWN_CATEGORY_LABELS)
    unknown_count = int(unknown_mask.sum())
    if unknown_count == 0: return result
    known_values = values.loc[~unknown_mask]
    if known_values.empty: return result
    proportions = known_values.value_counts(normalize=True)
    exact_counts = proportions * unknown_count
    allocation = np.floor(exact_counts).astype(int)
    remainder = unknown_count - int(allocation.sum())
    fractional = (exact_counts - allocation).sort_values(ascending=False, kind="stable")
    for category in fractional.index[:remainder]: allocation.loc[category] += 1
    replacement_values: list[str] = []
    for category, count in allocation.items(): replacement_values.extend([str(category)] * int(count))
    result.loc[result.index[unknown_mask], column] = replacement_values
    return result

def parse_spread_rank(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_numeric(text, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        extracted = text[missing].str.extract(r"(\d+)", expand=False)
        parsed.loc[missing] = pd.to_numeric(extracted, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        lowered = text[missing].str.casefold()
        def _match_keyword(value: str) -> float:
            for phrase, rank in SPREAD_TEXT_TO_RANK.items():
                if phrase in value: return float(rank)
            return np.nan
        parsed.loc[missing] = lowered.map(_match_keyword)
    parsed = parsed.where(parsed.isin([1, 2, 3, 4, 5]))
    return parsed.replace({5: 4}).astype("Float64")

def _validate_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing: raise KeyError(f"{name} is missing required column(s): {missing}.\nAvailable columns: {list(df.columns)}")

@st.cache_data(show_spinner=False)
def load_fire_data(path: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    frame = pd.read_csv(path, keep_default_na=False, na_values=CUSTOM_NA_VALUES)
    frame.columns = [_normalise_column_name(column) for column in frame.columns]
    frame = redistribute_unknown_category(frame, "OCCUPIED_NORMAL")
    return frame

@st.cache_data(show_spinner=False)
def load_exposure_data(path: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    frame = pd.read_excel(path)
    frame.columns = [_normalise_column_name(column) for column in frame.columns]
    _validate_required_columns(frame, ["Category", "Date"], "Exposure workbook")
    frame["Category"] = _normalise_category(frame["Category"])
    frame["Date"] = normalise_year_series(frame["Date"])
    return frame

def normalise_year_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series): return pd.to_numeric(series, errors="coerce").round().astype("Int64")
    text = series.astype("string").str.strip()
    from_financial_year = text.str.extract(r"((?:19|20)\d{2})", expand=False)
    numeric_year = pd.to_numeric(from_financial_year, errors="coerce")
    if numeric_year.notna().any(): return numeric_year.astype("Int64")
    dates = pd.to_datetime(text, errors="coerce")
    return dates.dt.year.astype("Int64")

def normalise_incident_period_year(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series): return pd.to_numeric(series, errors="coerce").round().astype("Int64")
    text = series.astype("string").str.strip()
    financial = text.str.extract(r"^(?P<start>(?:19|20)\d{2})\s*[/  \-  ]\s*(?P<end>(?:(?:19|20)?\d{2}))$")
    start = pd.to_numeric(financial["start"], errors="coerce")
    end_text = financial["end"]
    end = pd.to_numeric(end_text, errors="coerce")
    short_end = end_text.str.len().eq(2)
    end.loc[short_end] = ((start.loc[short_end] // 100) * 100 + end.loc[short_end])
    end.loc[end < start] = end.loc[end < start] + 100
    result = end.astype("Int64")
    unresolved = result.isna()
    if unresolved.any(): result.loc[unresolved] = normalise_year_series(text.loc[unresolved])
    return result

def find_year_column(frame: pd.DataFrame) -> str | None:
    priority = ["FINANCIAL_YEAR", "YEAR", "Year", "year", "INCIDENT_YEAR"]
    for candidate in priority:
        if candidate in frame.columns: return candidate
    for column in frame.columns:
        upper = column.upper()
        if "YEAR" in upper or "DATE" in upper:
            if normalise_year_series(frame[column]).notna().any(): return column
    return None

def year_like_columns(frame: pd.DataFrame) -> list[str]:
    candidates = []
    for column in frame.columns:
        if "YEAR" in column.upper() or "DATE" in column.upper():
            if normalise_year_series(frame[column]).notna().any(): candidates.append(column)
    return candidates

def attach_year_column(frame: pd.DataFrame, year_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["YEAR"] = normalise_incident_period_year(result[year_column])
    return result

def filter_frequency_period(frame: pd.DataFrame) -> pd.DataFrame:
    if "YEAR" not in frame.columns: raise ValueError("A year column is required for frequency analysis.")
    filtered = frame.loc[frame["YEAR"].between(FREQUENCY_FIRE_END_YEAR_START, FREQUENCY_FIRE_END_YEAR_END)].copy()
    if filtered.empty: raise ValueError(f"No fire incidents fall within {YEAR_START}-{YEAR_END} after year parsing.")
    return filtered

def try_load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    required_paths = {
        "Full fire incident CSV": FULL_FIRE_CSV_PATH,
        "Training fire incident CSV": TRAIN_FIRE_CSV_PATH,
        "Testing fire incident CSV": TEST_FIRE_CSV_PATH,
        "Exposure workbook": EXPOSURE_XLSX_PATH,
    }
    missing_files = [f"{label}: {path}" for label, path in required_paths.items() if not os.path.isfile(path)]
    if missing_files: return None, None, None, None, ["The following required project file(s) were not found:", *missing_files]
    try:
        full_fire = load_fire_data(FULL_FIRE_CSV_PATH, _file_mtime(FULL_FIRE_CSV_PATH))
        train_fire = load_fire_data(TRAIN_FIRE_CSV_PATH, _file_mtime(TRAIN_FIRE_CSV_PATH))
        test_fire = load_fire_data(TEST_FIRE_CSV_PATH, _file_mtime(TEST_FIRE_CSV_PATH))
        exposure = load_exposure_data(EXPOSURE_XLSX_PATH, _file_mtime(EXPOSURE_XLSX_PATH))
        return full_fire, train_fire, test_fire, exposure, []
    except Exception as error: return None, None, None, None, [str(error)]

def exposure_column(frame: pd.DataFrame) -> str:
    candidates = [column for column in frame.columns if "allocated dwellings" in column.casefold()]
    if not candidates: raise KeyError("The exposure workbook has no 'Allocated Dwellings per Category' column.")
    return candidates[0]

def exposure_to_dwelling_years(values: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty: return numeric
    if "million" in column_name.casefold() and non_null.abs().median() < 10_000: return numeric * 1_000_000
    return numeric

def _compute_frequency_raw(frequency_fire_df: pd.DataFrame, exposure_df: pd.DataFrame, occupancy_type: str) -> tuple[float, int, float]:
    fire_subset = frequency_fire_df.loc[frequency_fire_df["OCCUPANCY_TYPE"].astype("string").str.casefold() == occupancy_type.strip().casefold()]
    fire_subset = fire_subset.loc[fire_subset["YEAR"].between(FREQUENCY_FIRE_END_YEAR_START, FREQUENCY_FIRE_END_YEAR_END)]
    dcol = exposure_column(exposure_df)
    exposure_subset = exposure_df.loc[(exposure_df["Category"].astype("string").str.casefold() == occupancy_type.strip().casefold()) & exposure_df["Date"].between(YEAR_START, YEAR_END)].copy()
    dwelling_years = exposure_to_dwelling_years(exposure_subset[dcol], dcol)
    total_exposure = float(dwelling_years.sum(skipna=True))
    total_fires = int(len(fire_subset))
    if not np.isfinite(total_exposure) or total_exposure <= 0: return 0.0, total_fires, 0.0
    return total_fires / total_exposure, total_fires, total_exposure

def compute_frequency(frequency_fire_df: pd.DataFrame, exposure_df: pd.DataFrame, occupancy_type: str) -> tuple[float, int, float]:
    return _compute_frequency_raw(frequency_fire_df, exposure_df, occupancy_type)

def classify_frequency_bin(lam: float) -> str:
    if lam > 0.01: return "Anticipated (A)"
    if 0.0001 < lam <= 0.01: return "Unlikely (U)"
    if 0.000001 < lam <= 0.0001: return "Extremely Unlikely (EU)"
    return "Beyond Extremely Unlikely (BEU)"

def prepare_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in INSPECTOR_INPUT_COLUMNS: result[column] = _normalise_category(frame[column])
    return result

def prepare_labelled_dataset(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    features = prepare_model_features(frame)
    labels = parse_spread_rank(frame[TARGET_COLUMN])
    valid = labels.notna()
    features = features.loc[valid].copy()
    labels = labels.loc[valid].astype(int).copy()
    return features, labels

def encode_features(features: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(features, columns=INSPECTOR_INPUT_COLUMNS, prefix=INSPECTOR_INPUT_COLUMNS, dtype=np.uint8)
    if feature_columns is None: return encoded, list(encoded.columns)
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    return aligned, feature_columns

def is_absent_alarm(value: Any) -> bool:
    text = str(value).strip().casefold()
    if not text or text in UNKNOWN_CATEGORY_LABELS: return False
    return any(token in text for token in ("no alarm", "alarm absent", "no alarms"))

def _probability_row_to_rank(probability_row: np.ndarray) -> int:
    highest = float(np.max(probability_row))
    return int(np.flatnonzero(np.isclose(probability_row, highest))[-1] + 1)

def calibrate_to_training_priors(probabilities: np.ndarray, class_priors: dict[int, float]) -> np.ndarray:
    weights = np.array([class_priors.get(rank, 0.0) ** PRIOR_CALIBRATION_EXPONENT for rank in range(1, 5)], dtype=float)
    adjusted = np.asarray(probabilities, dtype=float) * weights
    totals = adjusted.sum(axis=-1, keepdims=True)
    return np.divide(adjusted, totals, out=np.full_like(adjusted, 0.25), where=totals > 0)

def expected_severity(probability_row: np.ndarray) -> float:
    return float(np.dot(np.asarray(probability_row, dtype=float), np.arange(1, 5)))

def apply_alarm_safety_uplift(no_alarm_probabilities: np.ndarray, active_alarm_probabilities: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(no_alarm_probabilities, dtype=float).copy()
    target_mean = min(3.75, max(expected_severity(adjusted), expected_severity(active_alarm_probabilities) + 0.30))
    for source_rank in (0, 1, 2):
        remaining_uplift = target_mean - expected_severity(adjusted)
        if remaining_uplift <= 0: break
        severity_gain = 3 - source_rank
        transfer = min(adjusted[source_rank], remaining_uplift / severity_gain)
        adjusted[source_rank] -= transfer
        adjusted[3] += transfer
    high_rank = 2 if adjusted[2] >= adjusted[3] else 3
    low_rank = int(np.argmax(adjusted[:2]))
    if adjusted[high_rank] <= adjusted[low_rank]:
        transfer = min(adjusted[low_rank], (adjusted[low_rank] - adjusted[high_rank] + 0.002) / 2)
        adjusted[low_rank] -= transfer
        adjusted[high_rank] += transfer
    return adjusted / adjusted.sum()

def _make_active_alarm_counterfactual(features: pd.DataFrame, active_alarm_reference: str) -> pd.DataFrame:
    counterfactual = features.copy()
    counterfactual["ALARM_SYSTEM_TYPE"] = active_alarm_reference
    return counterfactual

def constrained_probabilities(model: RandomForestClassifier, feature_columns: list[str], raw_features: pd.DataFrame, active_alarm_reference: str, class_priors: dict[int, float]) -> tuple[np.ndarray, int]:
    encoded, _ = encode_features(raw_features, feature_columns)
    raw_probabilities = model.predict_proba(encoded)
    probabilities = np.zeros((len(raw_features), 4), dtype=float)
    for class_index, rank in enumerate(model.classes_):
        if int(rank) in RANK_TO_LABEL: probabilities[:, int(rank) - 1] = raw_probabilities[:, class_index]
    probabilities = calibrate_to_training_priors(probabilities, class_priors)
    absent_alarm = raw_features["ALARM_SYSTEM_TYPE"].map(is_absent_alarm).to_numpy(dtype=bool)
    override_count = 0
    if absent_alarm.any():
        active_features = _make_active_alarm_counterfactual(raw_features.loc[absent_alarm], active_alarm_reference)
        active_encoded, _ = encode_features(active_features, feature_columns)
        active_raw = model.predict_proba(active_encoded)
        active_probabilities = np.zeros((len(active_features), 4), dtype=float)
        for class_index, rank in enumerate(model.classes_):
            if int(rank) in RANK_TO_LABEL: active_probabilities[:, int(rank) - 1] = active_raw[:, class_index]
        active_probabilities = calibrate_to_training_priors(active_probabilities, class_priors)
        for output_row, active_row in zip(np.flatnonzero(absent_alarm), active_probabilities):
            probabilities[output_row] = apply_alarm_safety_uplift(probabilities[output_row], active_row)
            override_count += 1
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(probabilities, row_sums, out=np.full_like(probabilities, 0.25), where=row_sums > 0)
    return probabilities, override_count

def choose_active_alarm_reference(features: pd.DataFrame) -> str:
    values = features["ALARM_SYSTEM_TYPE"].dropna().astype(str).tolist()
    for preferred in ("Mains Powered", "Battery Powered"):
        if preferred in values: return preferred
    return "Mains Powered"

def grouped_feature_importance(model: RandomForestClassifier, feature_columns: list[str]) -> pd.Series:
    raw = pd.Series(model.feature_importances_, index=feature_columns)
    grouped = {}
    for source_column in INSPECTOR_INPUT_COLUMNS:
        prefix = f"{source_column}_"
        grouped[source_column] = float(raw.loc[raw.index.str.startswith(prefix)].sum())
    result = pd.Series(grouped).sort_values(ascending=False)
    return result / result.sum() if result.sum() > 0 else result

@st.cache_resource(show_spinner=True)
def train_severity_model(train_fire_df: pd.DataFrame, test_fire_df: pd.DataFrame) -> tuple[RandomForestClassifier, list[str], str, dict[int, float], dict[str, Any]]:
    train_features, y_train = prepare_labelled_dataset(train_fire_df, "Training fire incident CSV")
    test_features, y_test = prepare_labelled_dataset(test_fire_df, "Testing fire incident CSV")
    x_train, feature_columns = encode_features(train_features)
    model = RandomForestClassifier(n_estimators=100, criterion="log_loss", max_features="sqrt", min_samples_leaf=8, max_samples=0.85, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(x_train, y_train)
    active_alarm_reference = choose_active_alarm_reference(train_features)
    class_priors = {rank: float((y_train == rank).mean()) for rank in range(1, 5)}
    final_probabilities, safety_override_count = constrained_probabilities(model, feature_columns, test_features, active_alarm_reference, class_priors)
    predicted = np.array([_probability_row_to_rank(p) for p in final_probabilities], dtype=int)
    metrics: dict[str, Any] = {"accuracy": accuracy_score(y_test, predicted), "safety_override_count": safety_override_count}
    return model, feature_columns, active_alarm_reference, class_priors, metrics

def predict_severity(model: RandomForestClassifier, feature_columns: list[str], active_alarm_reference: str, class_priors: dict[int, float], inputs: dict[str, str]) -> tuple[int, dict[int, float], bool]:
    input_frame = pd.DataFrame([inputs])
    clean_features = prepare_model_features(input_frame)
    probabilities, override_count = constrained_probabilities(model, feature_columns, clean_features, active_alarm_reference, class_priors)
    final_row = probabilities[0]
    rank = _probability_row_to_rank(final_row)
    probability_by_rank = {index + 1: float(value) for index, value in enumerate(final_row)}
    return rank, probability_by_rank, bool(override_count)

RISK_ZONE_MAP = {
    ("Anticipated (A)", "Catastrophic"): "High", ("Anticipated (A)", "Critical"): "High",
    ("Anticipated (A)", "Marginal"): "Moderate", ("Anticipated (A)", "Negligible"): "Negligible",
    ("Unlikely (U)", "Catastrophic"): "High", ("Unlikely (U)", "Critical"): "Moderate",
    ("Unlikely (U)", "Marginal"): "Low", ("Unlikely (U)", "Negligible"): "Negligible",
    ("Extremely Unlikely (EU)", "Catastrophic"): "Moderate", ("Extremely Unlikely (EU)", "Critical"): "Low",
    ("Extremely Unlikely (EU)", "Marginal"): "Low", ("Extremely Unlikely (EU)", "Negligible"): "Negligible",
    ("Beyond Extremely Unlikely (BEU)", "Catastrophic"): "Negligible", ("Beyond Extremely Unlikely (BEU)", "Critical"): "Negligible",
    ("Beyond Extremely Unlikely (BEU)", "Marginal"): "Negligible", ("Beyond Extremely Unlikely (BEU)", "Negligible"): "Negligible",
}
ZONE_COLORS = {"Negligible": "#27ae60", "Low": "#f1c40f", "Moderate": "#e67e22", "High": "#e74c3c"}
FREQ_ORDER = ["Beyond Extremely Unlikely (BEU)", "Extremely Unlikely (EU)", "Unlikely (U)", "Anticipated (A)"]
SEV_ORDER = ["Negligible", "Marginal", "Critical", "Catastrophic"]

def fig_to_data_uri(fig: plt.Figure, dpi: int = 150) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def plot_matrix(freq_bin: str, sev_bin: str, lam: float, predicted_rank: int) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(4.4, 4.4))
    for severity_index, severity in enumerate(SEV_ORDER):
        for frequency_index, frequency in enumerate(FREQ_ORDER):
            zone = RISK_ZONE_MAP[(frequency, severity)]
            axis.add_patch(plt.Rectangle((frequency_index, severity_index), 1, 1, facecolor=ZONE_COLORS[zone], edgecolor="white", linewidth=2.2, alpha=0.95))
            axis.text(frequency_index + 0.5, severity_index + 0.5, zone, ha="center", va="center", fontsize=7.2, color="white", fontweight="medium")
    frequency_index = FREQ_ORDER.index(freq_bin)
    severity_index = SEV_ORDER.index(sev_bin)
    axis.add_patch(plt.Rectangle((frequency_index, severity_index), 1, 1, facecolor="none", edgecolor="black", linewidth=3.2, zorder=6))
    axis.set_xticks([index + 0.5 for index in range(4)])
    axis.set_xticklabels([label.split(" (")[1].rstrip(")") for label in FREQ_ORDER], fontsize=7.6, fontweight="medium")
    axis.set_yticks([index + 0.5 for index in range(4)])
    axis.set_yticklabels(SEV_ORDER, fontsize=7.4, fontweight="medium")
    axis.set_xlim(0, 4)
    axis.set_ylim(0, 4)
    axis.set_title(f"{sev_bin}   |   λ = {lam:.6f}", fontsize=9, fontweight="medium", pad=8)
    for spine in axis.spines.values(): spine.set_visible(False)
    axis.tick_params(length=0)
    fig.tight_layout()
    return fig

def plot_severity_probabilities(proba_by_rank: dict[int, float], predicted_rank: int) -> plt.Figure:
    severity_colors = [ZONE_COLORS["Negligible"], ZONE_COLORS["Low"], ZONE_COLORS["Moderate"], ZONE_COLORS["High"]]
    ranks = sorted(proba_by_rank)
    values = [proba_by_rank[rank] * 100 for rank in ranks]
    labels = [RANK_TO_LABEL[rank] for rank in ranks]
    colors = [severity_colors[rank - 1] for rank in ranks]
    fig, axis = plt.subplots(figsize=(3.0, 2.6))
    bars = axis.barh(labels, values, color=colors, edgecolor="white", height=0.55, alpha=0.9)
    for bar, value, rank in zip(bars, values, ranks):
        axis.text(min(value + 2, 96), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=7.5)
        if rank == predicted_rank: bar.set_edgecolor("black"); bar.set_linewidth(1.4)
    axis.set_xlim(0, 100)
    axis.set_title("Severity probability", fontsize=7.8)
    for spine in ("top", "right"): axis.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    st.write("Please run the main dashboard file.")
