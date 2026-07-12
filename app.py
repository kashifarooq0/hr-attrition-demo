"""
HR Attrition Risk + Survival Curve — Live Demo (Streamlit)
-------------------------------------------------------------
Mirrors the modeling pipeline from the notebook:
  - Cox Proportional Hazards model (lifelines) -> survival curve over tenure
  - XGBoost classifier + sigmoid calibration (sklearn) -> calibrated attrition risk %

On startup, the app trains both models on the IBM HR Analytics dataset
(https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
Place the CSV file "WA_Fn-UseC_-HR-Employee-Attrition.csv" in the same folder
as this script before running / deploying.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud (free):
    1. Push app.py, requirements.txt, and the CSV to a public GitHub repo
    2. Go to https://share.streamlit.io -> "New app"
    3. Point it at the repo / app.py -> Deploy
    4. You get a permanent public URL, e.g. https://yourname-hr-attrition.streamlit.app
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

from lifelines import CoxPHFitter
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

import dice_ml
from dice_ml import Dice

st.set_page_config(page_title="HR Attrition Risk & Survival Demo", page_icon="🔍", layout="wide")

CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


# --------------------------------------------------------------------------
# 1. Load data + train models (cached so this only runs once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on startup (Cox PH + XGBoost)...")
def load_and_train():
    if not os.path.exists(CSV_NAME):
        st.error(
            f"Could not find '{CSV_NAME}' in the app directory.\n\n"
            "Download it from Kaggle (pavansubhasht/ibm-hr-analytics-attrition-dataset) "
            "and place it next to app.py before deploying."
        )
        st.stop()

    df = pd.read_csv(CSV_NAME)

    # ---- Survival dataset + Cox PH model ----
    df_surv = df.copy()
    df_surv["event"] = (df_surv["Attrition"] == "Yes").astype(int)
    df_surv["duration"] = df_surv["YearsAtCompany"].replace(0, 0.5)

    leakage_cols = [
        "TotalWorkingYears", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager", "JobLevel",
    ]
    drop_cols = [
        "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours",
        "Attrition", "YearsAtCompany",
    ] + leakage_cols

    features_df = df_surv.drop(columns=drop_cols)
    features_df["OverTime"] = (features_df["OverTime"] == "Yes").astype(int)

    categorical_cols = features_df.select_dtypes(include="object").columns.tolist()
    features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
    features_df.columns = [
        c.replace(" ", "_").replace("-", "_").replace("&", "and") for c in features_df.columns
    ]

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(features_df, duration_col="duration", event_col="event")
    survival_feature_cols = [c for c in features_df.columns if c not in ("duration", "event")]

    # ---- Classifier dataset + XGBoost + calibration ----
    clf_df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
    clf_df["OverTime"] = (clf_df["OverTime"] == "Yes").astype(int)
    clf_df["Attrition"] = (clf_df["Attrition"] == "Yes").astype(int)

    categorical_cols_clf = clf_df.select_dtypes(include="object").columns.tolist()
    clf_encoded = pd.get_dummies(clf_df, columns=categorical_cols_clf, drop_first=True)
    clf_encoded.columns = [
        c.replace(" ", "_").replace("-", "_").replace("&", "and") for c in clf_encoded.columns
    ]

    X = clf_encoded.drop(columns=["Attrition"])
    y = clf_encoded["Attrition"]
    clf_feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # scikit-learn >=1.6 removed cv="prefit"; it now requires wrapping the
    # already-fitted estimator in FrozenEstimator instead. Support both.
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(xgb_model), method="sigmoid")
    except ImportError:
        calibrated_model = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_test, y_test)

    survival_defaults = features_df[survival_feature_cols].median(numeric_only=True)
    clf_defaults = X[clf_feature_cols].median(numeric_only=True)

    meta = {
        "departments": sorted(df["Department"].unique().tolist()),
        "job_roles": sorted(df["JobRole"].unique().tolist()),
        "marital_statuses": sorted(df["MaritalStatus"].unique().tolist()),
        "business_travel": sorted(df["BusinessTravel"].unique().tolist()),
        "genders": sorted(df["Gender"].unique().tolist()),
    }

    # ---- DiCE counterfactuals: "what's the minimal change that lowers risk?" ----
    # Same setup as the notebook: restrict changes to a small set of realistic,
    # HR-actionable levers, and search for the smallest edit that flips the
    # predicted class from "will leave" to "will stay".
    actionable_features = [
        "OverTime", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
        "TrainingTimesLastYear", "DistanceFromHome", "JobInvolvement",
    ]
    actionable_features = [f for f in actionable_features if f in X_train.columns]

    continuous_features = [col for col in X_train.columns if X_train[col].nunique() > 15]

    train_dataset = X_train.copy()
    train_dataset["Attrition"] = y_train.values

    dice_data = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=continuous_features,
        outcome_name="Attrition",
    )
    # DiCE's random-search method returns candidate rows as object dtype,
    # which XGBoost rejects. Wrap the model so every prediction call casts
    # back to numeric first.
    class _NumericCastWrapper:
        def __init__(self, model, cols):
            self.model = model
            self.cols = cols

        def predict(self, X):
            Xc = pd.DataFrame(X, columns=self.cols).astype(float)
            return self.model.predict(Xc)

        def predict_proba(self, X):
            Xc = pd.DataFrame(X, columns=self.cols).astype(float)
            return self.model.predict_proba(Xc)

    wrapped_model_for_dice = _NumericCastWrapper(calibrated_model, clf_feature_cols)
    dice_model = dice_ml.Model(model=wrapped_model_for_dice, backend="sklearn")
    # method="random" is fast and dependency-light -- good fit for a live web demo
    dice_exp = Dice(dice_data, dice_model, method="random")

    return (
        cph, survival_feature_cols, survival_defaults,
        calibrated_model, clf_feature_cols, clf_defaults, meta,
        dice_exp, actionable_features, X_train.dtypes,
    )


(
    cph, SURVIVAL_FEATURE_COLS, SURVIVAL_DEFAULTS,
    calibrated_model, CLF_FEATURE_COLS, CLF_DEFAULTS, meta,
    dice_exp, ACTIONABLE_FEATURES, CLF_TRAIN_DTYPES,
) = load_and_train()


# --------------------------------------------------------------------------
# 2. Helpers to turn form inputs into model-ready rows
# --------------------------------------------------------------------------
def _set_onehot(row, defaults, prefix, value):
    for col in defaults.index:
        if col.startswith(prefix + "_"):
            row[col] = 0
    target_col = f"{prefix}_{value}".replace(" ", "_").replace("-", "_").replace("&", "and")
    if target_col in row.index:
        row[target_col] = 1
    return row


def build_feature_row(defaults, cols, inputs):
    row = pd.Series(defaults, index=cols).fillna(0)

    row["Age"] = inputs["age"]
    row["MonthlyIncome"] = inputs["monthly_income"]
    row["DistanceFromHome"] = inputs["distance_from_home"]
    row["OverTime"] = 1 if inputs["overtime"] == "Yes" else 0
    row["JobSatisfaction"] = inputs["job_satisfaction"]
    row["EnvironmentSatisfaction"] = inputs["env_satisfaction"]
    row["WorkLifeBalance"] = inputs["work_life_balance"]
    row["JobInvolvement"] = inputs["job_involvement"]
    row["NumCompaniesWorked"] = inputs["num_companies_worked"]
    row["StockOptionLevel"] = inputs["stock_option_level"]
    row["TrainingTimesLastYear"] = inputs["training_times"]

    row = _set_onehot(row, defaults, "Department", inputs["department"])
    row = _set_onehot(row, defaults, "JobRole", inputs["job_role"])
    row = _set_onehot(row, defaults, "MaritalStatus", inputs["marital_status"])
    row = _set_onehot(row, defaults, "BusinessTravel", inputs["business_travel"])
    row = _set_onehot(row, defaults, "Gender", inputs["gender"])

    return pd.DataFrame([row])[cols]


def median_tenure_from_curve(surv_curve):
    """Returns the year at which predicted survival probability first drops
    below 50%, or None if it never does within the modeled horizon."""
    below_half = surv_curve[surv_curve.iloc[:, 0] < 0.5]
    if len(below_half) > 0:
        return float(below_half.index[0])
    return None


# --------------------------------------------------------------------------
# 3. UI
# --------------------------------------------------------------------------
st.title("🔍 HR Attrition Risk & Survival Curve — Live Demo")
st.markdown(
    """
    Enter an employee profile below. The app runs two models trained on the
    [IBM HR Analytics dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset):

    - **XGBoost classifier (calibrated)** → predicted probability this employee leaves
    - **Cox Proportional Hazards model** → a full survival curve showing *when* risk builds over tenure
    - **DiCE counterfactual search** → the smallest realistic change (overtime, satisfaction, income, etc.)
      that would lower this employee's risk **and** push out their expected time-to-leave

    Same modeling pipeline as the accompanying notebook — try changing Overtime,
    Monthly Income, or Job Satisfaction and watch the risk score move.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee profile")
    age = st.slider("Age", 18, 60, 35)
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
    distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 5)
    overtime = st.radio("Works Overtime", ["Yes", "No"], index=1, horizontal=True)
    department = st.selectbox("Department", meta["departments"])
    job_role = st.selectbox("Job Role", meta["job_roles"])
    gender = st.selectbox("Gender", meta["genders"])
    marital_status = st.selectbox("Marital Status", meta["marital_statuses"])
    business_travel = st.selectbox("Business Travel", meta["business_travel"])

with col2:
    st.subheader("Satisfaction & engagement (1 = low, 4 = high)")
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    num_companies_worked = st.slider("Number of Companies Worked", 0, 9, 2)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 0)
    training_times = st.slider("Training Times Last Year", 0, 6, 2)

st.divider()

if st.button("🔮 Predict risk & survival curve", type="primary"):
    inputs = dict(
        age=age, monthly_income=monthly_income, distance_from_home=distance_from_home,
        overtime=overtime, job_satisfaction=job_satisfaction, env_satisfaction=env_satisfaction,
        work_life_balance=work_life_balance, job_involvement=job_involvement,
        num_companies_worked=num_companies_worked, stock_option_level=stock_option_level,
        training_times=training_times, department=department, job_role=job_role,
        marital_status=marital_status, business_travel=business_travel, gender=gender,
    )

    # --- Calibrated attrition risk (XGBoost) ---
    clf_row = build_feature_row(CLF_DEFAULTS, CLF_FEATURE_COLS, inputs)
    risk = calibrated_model.predict_proba(clf_row)[0, 1]

    if risk < 0.15:
        risk_label = "🟢 Low risk"
    elif risk < 0.40:
        risk_label = "🟡 Medium risk"
    else:
        risk_label = "🔴 High risk"

    st.subheader(f"{risk_label} — {risk:.1%} predicted probability of leaving")

    # --- Survival curve (Cox PH) ---
    surv_row = build_feature_row(SURVIVAL_DEFAULTS, SURVIVAL_FEATURE_COLS, inputs)
    surv_curve = cph.predict_survival_function(surv_row)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.step(surv_curve.index, surv_curve.iloc[:, 0], where="post", color="#4C72B0", linewidth=2)
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Probability of still being employed")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Predicted survival curve for this employee profile")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)

    median_tenure = median_tenure_from_curve(surv_curve)
    if median_tenure is not None:
        st.markdown(f"Median expected tenure (50% survival point): **~{median_tenure:.1f} years**")
    else:
        st.markdown("Predicted survival probability stays above 50% for the entire modeled horizon.")

    # --- Minimal change recommendation (DiCE counterfactuals) ---
    st.divider()
    st.subheader("🎯 What's the smallest change that would lower this risk — and push out when they'd leave?")

    with st.spinner("Searching for minimal, realistic changes..."):
        try:
            # DiCE validates the query row's dtypes against the training data's
            # dtypes (mostly int64 for Likert scales / one-hot columns). Our
            # row is built as a pandas Series, which forces float64 across the
            # board -- cast back to the training dtypes before calling DiCE.
            clf_row_for_dice = clf_row.copy()
            for col in clf_row_for_dice.columns:
                if col in CLF_TRAIN_DTYPES.index:
                    clf_row_for_dice[col] = clf_row_for_dice[col].astype(CLF_TRAIN_DTYPES[col])

            income = float(clf_row_for_dice["MonthlyIncome"].values[0])
            distance = float(clf_row_for_dice["DistanceFromHome"].values[0])
            permitted_range = {}
            if "MonthlyIncome" in ACTIONABLE_FEATURES:
                permitted_range["MonthlyIncome"] = [income, income * 1.2]
            if "DistanceFromHome" in ACTIONABLE_FEATURES:
                permitted_range["DistanceFromHome"] = [0, distance]

            cf_result = dice_exp.generate_counterfactuals(
                clf_row_for_dice,
                total_CFs=3,
                desired_class="opposite",
                features_to_vary=ACTIONABLE_FEATURES,
                permitted_range=permitted_range if permitted_range else None,
            )
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
        except Exception:
            cf_df = None

    if cf_df is not None and len(cf_df) > 0:
        original_row = clf_row.iloc[0]
        any_scenario_shown = False

        for i, (_, cf_row_result) in enumerate(cf_df.iterrows(), start=1):
            changes = []
            for col in ACTIONABLE_FEATURES:
                old_val = original_row[col]
                new_val = cf_row_result[col]
                if old_val != new_val:
                    if col == "OverTime":
                        old_disp = "Yes" if old_val == 1 else "No"
                        new_disp = "Yes" if new_val == 1 else "No"
                        changes.append(f"**Overtime**: {old_disp} → {new_disp}")
                    else:
                        changes.append(f"**{col}**: {old_val:g} → {new_val:g}")

            if not changes:
                continue

            any_scenario_shown = True

            new_row = clf_row.copy()
            for col in ACTIONABLE_FEATURES:
                new_row[col] = cf_row_result[col]
            new_risk = calibrated_model.predict_proba(new_row)[0, 1]

            # Apply the same edits to the survival-model row so we can show
            # how the expected time-to-leave shifts, not just the risk %.
            new_surv_row = surv_row.copy()
            for col in ACTIONABLE_FEATURES:
                if col in new_surv_row.columns:
                    new_surv_row[col] = cf_row_result[col]
            new_surv_curve = cph.predict_survival_function(new_surv_row)
            new_median_tenure = median_tenure_from_curve(new_surv_curve)

            if median_tenure is not None and new_median_tenure is not None:
                tenure_text = (
                    f"expected tenure shifts from **~{median_tenure:.1f} yrs** "
                    f"to **~{new_median_tenure:.1f} yrs** "
                    f"({'+' if new_median_tenure >= median_tenure else ''}{new_median_tenure - median_tenure:.1f} yrs)"
                )
            elif new_median_tenure is not None:
                tenure_text = f"expected tenure becomes **~{new_median_tenure:.1f} yrs** (previously beyond the modeled horizon)"
            elif median_tenure is not None:
                tenure_text = f"expected tenure moves beyond the modeled horizon (previously **~{median_tenure:.1f} yrs**)"
            else:
                tenure_text = "expected tenure stays beyond the modeled horizon in both cases"

            st.markdown(
                f"**Scenario {i}:** {', '.join(changes)}\n\n"
                f"→ Predicted risk changes from **{risk:.1%}** to **{new_risk:.1%}**, and {tenure_text}."
            )

        if not any_scenario_shown:
            st.info("No change within the actionable feature set materially lowers this employee's risk.")
    else:
        st.info(
            "No realistic, bounded combination of the actionable features "
            "(Overtime, Monthly Income, Job Satisfaction, Work-Life Balance, Training, "
            "Distance From Home, Job Involvement) was found to flip this employee to low risk. "
            "Their risk may be driven mainly by factors outside typical HR policy levers "
            "(e.g. age, tenure, marital status)."
        )

st.divider()
st.caption(
    "Predicted probabilities are calibrated against held-out test data (sigmoid calibration), "
    "so a \"40% risk\" prediction roughly matches real-world attrition rates for similar profiles. "
    "This is a demo built on a public dataset — not a production HR tool."
)
