import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 2.3rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px;}
    .sub-header {font-size: 1.1rem; color: #4B5563; margin-bottom: 25px;}
    .stAlert {border-radius: 8px;}
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Caching & Resource Loading
# ==========================================
@st.cache_resource
def load_model():
    """Loads the pre-trained machine learning model."""
    model_path = "Loan_Predictor.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_dataset():
    """Loads historical dataset and cleans string/categorical columns."""
    data_paths = ["loan_approval_data.csv", "loan_dataset.csv"]
    data_path = next((path for path in data_paths if os.path.exists(path)), None)

    if data_path:
        df = pd.read_csv(data_path)

        # Clean column names and string whitespace
        df.columns = df.columns.str.strip()
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # Map target column to standardized 'Status' and 'Loan_Approved'
        target_col = next(
            (
                c
                for c in ["Loan_Approved", "loan_status", "Loan_Status", "Status"]
                if c in df.columns
            ),
            None,
        )

        if target_col:
            approved_vals = [
                1,
                1.0,
                True,
                "1",
                "Y",
                "Yes",
                "Approved",
                "approved",
            ]
            df["Status"] = df[target_col].apply(
                lambda x: "Approved" if x in approved_vals else "Rejected"
            )
            df["Loan_Approved"] = (df["Status"] == "Approved").astype(int)

        return df

    # Fallback mock dataset if CSV file is absent
    np.random.seed(42)
    n = 500
    mock_df = pd.DataFrame(
        {
            "Applicant_ID": [float(i + 1) for i in range(n)],
            "Applicant_Income": np.random.normal(10500, 4000, n).round(2),
            "Coapplicant_Income": np.random.normal(5000, 2000, n).round(2),
            "Employment_Status": np.random.choice(
                ["Salaried", "Self-employed", "Contract", "Unemployed"], n
            ),
            "Age": np.random.randint(21, 60, n).astype(float),
            "Marital_Status": np.random.choice(["Married", "Single"], n),
            "Dependents": np.random.choice([0.0, 1.0, 2.0, 3.0], n),
            "Credit_Score": np.random.normal(675, 50, n).round(0),
            "Existing_Loans": np.random.choice([0.0, 1.0, 2.0, 3.0, 4.0], n),
            "DTI_Ratio": np.random.uniform(0.1, 0.6, n).round(2),
            "Savings": np.random.normal(10000, 4000, n).round(2),
            "Collateral_Value": np.random.normal(25000, 10000, n).round(2),
            "Loan_Amount": np.random.normal(20000, 8000, n).round(2),
            "Loan_Term": np.random.choice(
                [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0], n
            ),
            "Loan_Purpose": np.random.choice(
                ["Personal", "Car", "Business", "Home", "Education"], n
            ),
            "Property_Area": np.random.choice(
                ["Urban", "Semiurban", "Rural"], n
            ),
            "Education_Level": np.random.choice(
                ["Not Graduate", "Graduate"], n
            ),
            "Gender": np.random.choice(["Female", "Male"], n),
            "Employer_Category": np.random.choice(
                ["Private", "Government", "Unemployed", "MNC", "Business"], n
            ),
            "Loan_Approved": np.random.choice([1, 0], n, p=[0.3, 0.7]),
        }
    )
    mock_df["Status"] = mock_df["Loan_Approved"].apply(
        lambda x: "Approved" if x == 1 else "Rejected"
    )
    return mock_df


model = load_model()
df = load_dataset()


def get_categories(col_name):
    """Extracts unique non-null categories present in the dataset."""
    if col_name in df.columns:
        vals = [
            str(v).strip()
            for v in df[col_name].dropna().unique()
            if str(v).lower() != "nan"
        ]
        return sorted(list(set(vals)))
    return []


def get_numeric_default(col_name, fallback_val):
    """Computes median numeric value from the dataset as a reliable default."""
    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
        val = df[col_name].dropna().median()
        if not pd.isna(val):
            return float(val)
    return float(fallback_val)


# ==========================================
# Application Header
# ==========================================
st.markdown(
    '<h1 class="main-header">🏦 Smart Loan Decision Engine</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h3 class="sub-header">AI-driven loan approval predictions and risk insights.</h3>',
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "⚠️ `Loan_Predictor.pkl` not found in the root directory. Running in demonstration mode using default rule heuristics."
    )

# ==========================================
# Sidebar UI: Inputs directly from dataset categories
# ==========================================
st.sidebar.header("📝 Applicant Profile")
st.sidebar.write("Fill in applicant details:")

with st.sidebar.form("applicant_form"):
    applicant_income = st.number_input(
        "Applicant Income ($)",
        min_value=0.0,
        value=get_numeric_default("Applicant_Income", 10000.0),
        step=500.0,
    )
    employment_status = st.selectbox(
        "Employment Status", options=get_categories("Employment_Status")
    )
    marital_status = st.selectbox(
        "Marital Status", options=get_categories("Marital_Status")
    )
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=int(get_numeric_default("Credit_Score", 675)),
        step=1,
    )
    dti_ratio = st.slider(
        "Debt-to-Income (DTI) Ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(get_numeric_default("DTI_Ratio", 0.35)),
        step=0.01,
    )
    loan_purpose = st.selectbox(
        "Loan Purpose", options=get_categories("Loan_Purpose")
    )
    property_area = st.selectbox(
        "Property Area", options=get_categories("Property_Area")
    )
    education_level = st.selectbox(
        "Education Level", options=get_categories("Education_Level")
    )
    gender = st.selectbox("Gender", options=get_categories("Gender"))
    employer_category = st.selectbox(
        "Employer Category", options=get_categories("Employer_Category")
    )

    submit_btn = st.form_submit_button(
        "Predict Approval Status", use_container_width=True
    )

# ==========================================
# Prediction & Evaluation Logic
# ==========================================
if submit_btn:
    # 1. Capture user inputs
    user_inputs = {
        "Applicant_Income": float(applicant_income),
        "Employment_Status": str(employment_status),
        "Marital_Status": str(marital_status),
        "Credit_Score": float(credit_score),
        "DTI_Ratio": float(dti_ratio),
        "Loan_Purpose": str(loan_purpose),
        "Property_Area": str(property_area),
        "Education_Level": str(education_level),
        "Gender": str(gender),
        "Employer_Category": str(employer_category),
    }

    # 2. Build baseline dataframe populated with dataset defaults for missing features
    target_cols = ["Loan_Approved", "Status"]
    dataset_features = [col for col in df.columns if col not in target_cols]

    full_input_dict = {}
    for col in dataset_features:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                val = df[col].dropna().median()
                full_input_dict[col] = float(val) if not pd.isna(val) else 0.0
            else:
                mode_vals = df[col].dropna().mode()
                full_input_dict[col] = (
                    str(mode_vals[0]) if len(mode_vals) > 0 else ""
                )
        else:
            full_input_dict[col] = 0.0

    # Override defaults with user form choices
    full_input_dict.update(user_inputs)
    input_data = pd.DataFrame([full_input_dict])

    # 3. Model Prediction with Automatic Encoding Handling
    with st.spinner("Processing evaluation..."):
        if model is not None:
            try:
                # Direct prediction if model handles categorical objects via internal Pipeline
                pred = model.predict(input_data)[0]
            except (TypeError, ValueError):
                # Fallback for models trained on raw/one-hot encoded numeric matrices
                input_data_encoded = pd.get_dummies(input_data)

                if hasattr(model, "feature_names_in_"):
                    expected_features = list(model.feature_names_in_)
                elif hasattr(model, "steps") and hasattr(
                    model.steps[0][1], "feature_names_in_"
                ):
                    expected_features = list(
                        model.steps[0][1].feature_names_in_
                    )
                else:
                    expected_features = input_data_encoded.columns.tolist()

                # Reindex features to align exactly with model expectations
                input_data_encoded = input_data_encoded.reindex(
                    columns=expected_features, fill_value=0
                )
                input_data_encoded = input_data_encoded.astype(float)

                pred = model.predict(input_data_encoded)[0]

            is_approved = bool(
                pred in [1, 1.0, True, "Y", "Yes", "Approved", "approved"]
            )
        else:
            # Rule heuristic fallback if pkl model is absent
            is_approved = (
                (credit_score >= 650)
                and (dti_ratio <= 0.40)
                and (employment_status != "Unemployed")
            )

    # 4. Display Decision Results
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Decision Result")
        if is_approved:
            st.success("✅ **APPROVED**")
            st.write("This applicant meets standard risk requirements.")
        else:
            st.error("❌ **REJECTED**")
            st.write("This applicant exceeds default risk thresholds.")

    # 5. Actionable Feedback Section
    with res_col2:
        st.subheader("💡 Profile Analysis & Suggestions")
        if not is_approved:
            st.write("**Recommended adjustments for re-application:**")
            if employment_status == "Unemployed":
                st.info(
                    "- **Employment:** Unemployed profiles face high rejection rates. Verified employment stability is essential."
                )
            if credit_score < 680:
                st.info(
                    f"- **Credit Score:** Current score ({credit_score}) is below standard targets. Target 680+ to improve approval odds."
                )
            if dti_ratio > 0.36:
                st.info(
                    f"- **DTI Ratio:** Current debt ratio ({dti_ratio*100:.1f}%) is high. Pay down outstanding balances to reduce DTI below 36%."
                )
        else:
            st.write("**Strong Profile Characteristics:**")
            if (
                property_area in ["Urban", "Semiurban", "Semi-Urban"]
                and dti_ratio < 0.35
                and applicant_income > 10000
            ):
                st.success(
                    "- High Acceptance Cohort: Urban applicants with solid earnings and low debt maintain top acceptance rates."
                )
            if credit_score >= 720:
                st.success(
                    "- Strong credit history significantly reduces risk score."
                )
            st.write("No corrective steps required.")

    st.markdown("---")

    # ==========================================
    # Interactive Visualizations
    # ==========================================
    st.subheader("📊 Portfolio Context & Market Insights")
    st.write(
        "Contextualizing current applicant against historical dataset parameters."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1 = px.box(
            df,
            x="Status",
            y="Applicant_Income",
            color="Status",
            title="Income Distribution by Loan Status",
            category_orders={"Status": ["Approved", "Rejected"]},
            color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
        )
        fig1.add_hline(
            y=applicant_income,
            line_dash="dash",
            line_color="black",
            annotation_text="Applicant Income",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        fig2 = px.histogram(
            df,
            x="Credit_Score",
            color="Status",
            barmode="overlay",
            title="Credit Score Distribution",
            nbins=30,
            opacity=0.7,
            color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
        )
        fig2.add_vline(
            x=credit_score,
            line_dash="dash",
            line_color="black",
            annotation_text="Applicant Score",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with chart_col1:
        fig3 = px.violin(
            df,
            x="Status",
            y="DTI_Ratio",
            color="Status",
            box=True,
            title="Debt-to-Income (DTI) Ratio Profile",
            color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
        )
        fig3.add_hline(
            y=dti_ratio,
            line_dash="dash",
            line_color="black",
            annotation_text="Applicant DTI",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col2:
        if "Loan_Amount" in df.columns and "Loan_Purpose" in df.columns:
            avg_df = (
                df.groupby(["Loan_Purpose", "Status"])["Loan_Amount"]
                .mean()
                .reset_index()
            )
            fig4 = px.bar(
                avg_df,
                x="Loan_Purpose",
                y="Loan_Amount",
                color="Status",
                barmode="group",
                title="Average Loan Amount by Purpose",
                color_discrete_map={
                    "Approved": "#22c55e",
                    "Rejected": "#ef4444",
                },
            )
            st.plotly_chart(fig4, use_container_width=True)

else:
    st.info(
        "👈 **Get Started:** Configure applicant parameters in the left sidebar and click **'Predict Approval Status'** to execute prediction."
    )
    st.write("### 📌 Portfolio Benchmark Rules")
    st.markdown(
        """
    * **Employment Priority:** Unemployed applicants show significantly higher risk profiles across historical benchmarks.
    * **Optimal Profile:** Urban applicants with stable income and DTI ratios under 35% demonstrate the highest likelihood of approval.
    """
    )