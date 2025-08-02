import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import joblib
import os
import base64

# ---------- CONFIG & STYLING ----------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.markdown("""
<style>
/* Global centering and background for most content */
.stApp {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 2rem;
  background: linear-gradient(135deg, #0f1e35 0%, #1f3358 60%, #006d77 100%);
  min-height: 100vh;
}
.main-container {
  display: flex;
  flex-direction: column;
  align-items: center; /* center everything inside except overridden parts */
  width: 90%;
  max-width: 1000px;
  margin: 0 auto;
  color: white;
}
.card {
  background: rgba(255,255,255,0.07);
  border-radius: 15px;
  padding: 2rem;
  margin: 1rem 0;
  box-shadow: 0 15px 40px rgba(0,0,0,0.35);
  width: 100%;
  text-align: center;
}
.stButton>button {
  background-color: #ffffff;
  color: #1f3358;
  font-weight: bold;
  border-radius: 8px;
  padding: 0.5em 1em;
  box-shadow: 0 6px 20px rgba(0,0,0,0.2);
  margin: 0 auto;
  display: block;
}

/* Make the Tableau embed container centered explicitly */
.tableau-container-wrapper {
  display: flex;
  justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Container wrapper for centered content
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ---------- TITLE ----------
st.header("Loan Default Risk Prediction")
st.markdown("An Intelligent ML System to assess your **Loan Default Risk** using Financial and Credit Parameters.")

# ---------- SVG helpers for Visualization Insights (unaligned intentionally) ----------
def svg_to_data_uri(svg_str):
    return "data:image/svg+xml;base64," + base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")

interaction_svg = """<svg width="120" height="80" viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
  <circle cx="30" cy="40" r="8" fill="#4caf50"/>
  <circle cx="60" cy="25" r="8" fill="#2196f3"/>
  <circle cx="90" cy="55" r="8" fill="#f9a825"/>
  <line x1="30" y1="40" x2="60" y2="25" stroke="#ffffff" stroke-width="2"/>
  <line x1="60" y1="25" x2="90" y2="55" stroke="#ffffff" stroke-width="2"/>
  <line x1="30" y1="40" x2="90" y2="55" stroke="#ffffff" stroke-width="1"/>
  <text x="5" y="75" font-size="8" fill="white">Feature Interaction</text>
</svg>"""

matrix_svg2 = """<svg width="120" height="80" viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="25" height="25" fill="none" stroke="white" stroke-width="1"/>
  <rect x="40" y="10" width="25" height="25" fill="none" stroke="white" stroke-width="1"/>
  <rect x="70" y="10" width="25" height="25" fill="none" stroke="white" stroke-width="1"/>
  <rect x="10" y="40" width="25" height="25" fill="none" stroke="white" stroke-width="1"/>
  <rect x="40" y="40" width="25" height="25" fill="none" stroke="white" stroke-width="1"/>
  <rect x="70" y="40" width="25" height="25" fill="none" stroke="white" stroke-width="1"/>
  <text x="10" y="77" font-size="8" fill="white">Pairwise Relationships</text>
</svg>"""

loan_income_svg = """<svg width="120" height="80" viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">
  <rect x="15" y="45" width="15" height="25" fill="#4caf50"/>
  <rect x="45" y="35" width="15" height="35" fill="#2196f3"/>
  <rect x="75" y="25" width="15" height="45" fill="#f9a825"/>
  <text x="5" y="75" font-size="8" fill="white">Loan Amount vs Income</text>
</svg>"""

# ---------- LOAD DATA ----------
sample_data_path = "C:/Users/RITUL/OneDrive/Desktop/loan-default-dashboard/data/loan_data_cleaned.csv"
df = None
if os.path.exists(sample_data_path):
    df = pd.read_csv(sample_data_path)
else:
    st.warning("⚠️ Please make sure `loan_data_cleaned.csv` exists inside `data/`.")

# ---------- Visualization Insights (NOT forced centered) ----------
st.markdown("### Visualization Insights")
carousel_items = [
    {
        "title": "Loan Amount vs Income",
        "image_data": svg_to_data_uri(loan_income_svg),
        "desc": "Borrowers with higher incomes tend to request larger loans; default risk modulates with interest rate and grade."
    },
    {
        "title": "Feature Interaction",
        "image_data": svg_to_data_uri(interaction_svg),
        "desc": "Shows how key numerical features co-relate; strong interactions can signal compounded risk factors."
    },
    {
        "title": "Pairwise Relationships",
        "image_data": svg_to_data_uri(matrix_svg2),
        "desc": "Highlights strongest pairwise correlations among numerical features to surface multicollinearity or risk clusters."
    },
]

titles_top = [it["title"] for it in carousel_items]
choice_top = st.radio("Select Insights", titles_top, horizontal=True, key="insight_top_radio")
selected_top = next(it for it in carousel_items if it["title"] == choice_top)

cols = st.columns([1, 2])
with cols[0]:
    st.image(selected_top["image_data"], use_container_width=True)
with cols[1]:
    st.markdown(
        f"""
<ul>
    <li style='font-size:20px;'><b>{selected_top['title']}</b></li>
</ul>
""",
        unsafe_allow_html=True
    )
    st.markdown(f"*{selected_top['desc']}*")

    if df is not None:
        if choice_top == "Feature Interaction":
            num_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment']
            available = [c for c in num_cols if c in df.columns]
            if len(available) >= 2:
                corr = df[available].dropna().corr().abs()
                pairs = (
                    corr.where(~np.eye(len(corr), dtype=bool))
                        .stack()
                        .sort_values(ascending=False)
                        .drop_duplicates()
                )
                top3 = pairs.head(3)
                st.markdown(
                    """
<ul>
    <li style='font-size:20px;'><b>Top Feature Interaction Strengths (Abs Correlation)</b></li>
</ul>
""",
                    unsafe_allow_html=True
                )
                top3_df = pd.DataFrame([
                    {"**Feature A**": a, "**Feature B**": b, "**Abs Correlation**": round(val, 2)}
                    for (a, b), val in top3.items()
                ])
                st.table(top3_df)
            else:
                st.info("Need at least two numerical features for interaction insight.")
        elif choice_top == "Pairwise Relationships":
            num_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment']
            available = [c for c in num_cols if c in df.columns]
            if len(available) >= 2:
                corr = df[available].dropna().corr()
                abs_corr = corr.abs()
                pairs = (
                    abs_corr.where(~np.eye(len(abs_corr), dtype=bool))
                            .stack()
                            .sort_values(ascending=False)
                            .drop_duplicates()
                )
                top_pairs = pairs.head(5)
                st.markdown(
                    """
<ul>
    <li style='font-size:20px;'><b>Strongest Pairwise Correlations</b></li>
</ul>
""",
                    unsafe_allow_html=True
                )
                insights = []
                for (a, b), v in top_pairs.items():
                    sign = "**Positive**" if corr.loc[a, b] >= 0 else "**Negative**"
                    insights.append({
                        "**Feature A**": a,
                        "**Feature B**": b,
                        "**Correlation**": f"{corr.loc[a,b]:.2f}",
                        "**Type**": sign
                    })
                st.table(pd.DataFrame(insights))
            else:
                st.info("Need at least two numerical features for pairwise insight.")
        elif choice_top == "Loan Amount vs Income":
            if {"loan_status", "loan_amnt", "annual_inc"}.issubset(df.columns):
                avg_loan_by_status = df.groupby("loan_status")["loan_amnt"].mean().rename("Avg Loan Amount")
                st.markdown(
                    """
<ul>
    <li style='font-size:20px;'><b>Average Loan Amount by Status</b></li>
</ul>
""",
                    unsafe_allow_html=True
                )
                st.dataframe(avg_loan_by_status.to_frame().style.format("{:.0f}"))
            else:
                st.info("Required columns missing for this insight.")

# ---------- LOAD MODEL & COLUMNS ----------
from download_model import ensure_all_models, load_model

# Ensure model artifacts are present (downloads if missing)
ensure_all_models()

model = None
feature_columns = None
try:
    model = load_model(os.path.join("models", "gbm_pipeline.pkl"))
    # assume model_columns.pkl was saved with joblib.dump; load directly
    feature_columns = joblib.load(os.path.join("models", "model_columns.pkl"))
except FileNotFoundError as e:
    st.error(f"🔴 Required file not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model or columns: {e}")
    st.stop()

# Normalize if wrapped in a dict
if isinstance(feature_columns, dict) and "columns" in feature_columns:
    feature_columns = feature_columns["columns"]

# ---------- INPUT FORM ----------
st.markdown("## Applicant Financial Details")
col1, col2 = st.columns(2, gap="large")
with col1:
    loan_amnt = st.number_input("Loan Amount (₹)", min_value=1000, max_value=1000000, value=10000, key="input_loan_amnt")
    annual_inc = st.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, value=50000, key="input_annual_inc")
    mort_acc = st.number_input("Number of Mortgage Accounts", min_value=0, max_value=50, value=2, key="input_mort_acc")
    issue_d_month = st.selectbox("Issue Month", list(range(1, 13)), index=0, key="input_issue_month")
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, 40.0, key="input_revol_util")
with col2:
    issue_d_year = st.number_input("Issue Year", min_value=1990, max_value=2025, value=2015, key="input_issue_year")
    term = st.selectbox("Loan Term (months)", [36, 60], key="input_term")
    grade = st.selectbox("Loan Grade", list("ABCDEFG"), key="input_grade")
    zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=22690, key="input_zip")
    int_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 13.0, key="input_int_rate")

input_df = pd.DataFrame([{
    "revol_util": revol_util,
    "issue_d_month": issue_d_month,
    "loan_amnt": loan_amnt,
    "mort_acc": mort_acc,
    "annual_inc": annual_inc,
    "int_rate": int_rate,
    "issue_d_year": issue_d_year,
    "term": term,
    "grade": grade,
    "zip": zip_code
}])

if feature_columns is not None:
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ---------- PREDICTION ----------
if st.button("Predict Loan Default", key="predict_button"):
    input_df = input_df.fillna(0)
    try:
        prediction = model.predict(input_df)[0]
        # choose colors
        if prediction:  # default
            label = "High Risk: Default"
            square = "<span style='display:inline-block;width:16px;height:16px;background:#64b5f6;border-radius:3px;margin-right:6px;'></span>"
        else:  # no default
            label = "Low Risk: No Default"
            square = "<span style='display:inline-block;width:16px;height:16px;background:#0d47a1;border-radius:3px;margin-right:6px;'></span>"

        st.markdown(f"### Prediction: {square}{label}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
            st.metric("Default Probability", f"{proba:.2%}")
            fig = px.pie(
                values=[proba, 1 - proba],
                names=["Default", "No Default"],
                hole=0.5,
                title="Prediction Probability",
                color_discrete_sequence=["#64b5f6", "#0d47a1"]
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- INTERACTIVE VISUALS ----------
st.markdown("---")
st.header("Interactive Loan Data Visualizations")

if df is not None:
    with st.expander("📈 Income vs Loan Amount by Status"):
        if {"annual_inc", "loan_amnt", "loan_status"}.issubset(df.columns):
            fig1 = px.scatter(df, x="annual_inc", y="loan_amnt", color="loan_status",
                              title="Income vs Loan Amount by Loan Status")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Required columns missing for this visualization.")

    with st.expander("📊 Loan Amount Distribution"):
        if "loan_amnt" in df.columns:
            fig2 = px.histogram(df, x="loan_amnt", nbins=50, title="Loan Amount Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Column 'loan_amnt' not present.")

    with st.expander("📌 DTI by Loan Status"):
        if {"dti", "loan_status"}.issubset(df.columns):
            fig3 = px.box(df, x="loan_status", y="dti", title="Debt-to-Income by Loan Status")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Required columns missing for this visualization.")

    with st.expander("📋 Loan Status vs Categorical Features"):
        categorical_cols = ["term", "grade", "home_ownership", "purpose", "verification_status"]
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        if categorical_cols:
            selected_cat = st.selectbox("Select a categorical feature", categorical_cols, key="cat_feature_select")
            if selected_cat:
                fig = px.histogram(
                    df,
                    x=selected_cat,
                    color="loan_status" if "loan_status" in df.columns else None,
                    barmode="group",
                    category_orders={selected_cat: sorted(df[selected_cat].dropna().unique())},
                    title=f"{selected_cat.replace('_',' ').title()} vs Loan Status",
                    labels={selected_cat: selected_cat.replace("_", " ").title(), "count": "Loan Count"}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available for this section.")

    with st.expander("🔍 Pairwise Relationships of Selected Numerical Features"):
        selected_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment']
    available = [c for c in selected_cols if c in df.columns]
    if len(available) >= 2:
        color_arg = "loan_status" if "loan_status" in df.columns else None
        fig = px.scatter_matrix(
            df,
            dimensions=available,
            color=color_arg,
            title="Pairwise Relationships of Selected Numerical Features",
            labels={col: col.replace("_", " ").title() for col in available},
            height=600
        )
        fig.update_traces(diagonal_visible=True)
        fig.update_layout(
            dragmode="select",
            hovermode="closest",
            margin=dict(t=50, l=25, r=25, b=25),
            legend_title_text="Loan Status" if color_arg else None
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least two numerical features to visualize the pairwise relationships.")

else:
    st.warning("⚠️ Loan data not loaded; interactive visuals unavailable.")

# ---------- TABLEAU EMBED ----------
st.markdown("---")
st.header("Tableau Loan Default Dashboard")
# Center the tableau embed wrapper explicitly
st.markdown("<div class='tableau-container-wrapper'>", unsafe_allow_html=True)
components.html("""
<div class="tableau-container">
  <div class='tableauPlaceholder' id='viz1753947164439' style='position: relative'>
    <noscript>
      <a href='#'>
        <img alt='Dashboard' src='https://public.tableau.com/static/images/pr/project_17539471101380/Dashboard1/1_rss.png' style='border: none' />
      </a>
    </noscript>
    <object class='tableauViz' style='display:none;'>
      <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
      <param name='embed_code_version' value='3'/>
      <param name='site_root' value=''/>
      <param name='name' value='project_17539471101380/Dashboard1'/>
      <param name='tabs' value='no'/>
      <param name='toolbar' value='yes'/>
      <param name='static_image' value='https://public.tableau.com/static/images/pr/project_17539471101380/Dashboard1/1.png'/>
      <param name='animate_transition' value='yes'/>
      <param name='display_static_image' value='yes'/>
      <param name='display_spinner' value='yes'/>
      <param name='display_overlay' value='yes'/>
      <param name='display_count' value='yes'/>
      <param name='language' value='en-US'/>
      <param name='filter' value='publish=yes'/>
    </object>
  </div>
</div>
<script type='text/javascript'>
  var divElement = document.getElementById('viz1753947164439');
  var vizElement = divElement.getElementsByTagName('object')[0];
  if ( divElement.offsetWidth > 800 ) {
    vizElement.style.width='1000px';vizElement.style.height='827px';
  } else if ( divElement.offsetWidth > 500 ) {
    vizElement.style.width='100%';vizElement.style.height='827px';
  } else {
    vizElement.style.width='100%';vizElement.style.height='827px';
  }
  var scriptElement = document.createElement('script');
  scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
  vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
""", height=900)
st.markdown("</div>", unsafe_allow_html=True)  # close Tableau wrapper

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🔗 Built using Streamlit, Plotly & Tableau | ML Model: Gradient Boosting Pipeline | © 2025")

# Close main container
st.markdown("</div>", unsafe_allow_html=True)

