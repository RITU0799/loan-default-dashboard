import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import base64
import os
from datetime import datetime

# ---------- CONFIG & STYLING ----------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg, #0f1e35 0%, #1f3358 60%, #006d77 100%);
color: white;
}
.card {
background: rgba(255,255,255,0.07);
border-radius: 15px;
padding: 1.5rem;
margin: 1rem 0;
}
.stButton>button {
background-color: #ffffff;
color: #1f3358;
font-weight: bold;
border-radius: 8px;
}
.note-box {
background-color: rgba(255,255,255,0.1);
border-left: 5px solid #FFD700;
padding: 1rem;
border-radius: 8px;
margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.header("Loan Default Risk Prediction")
st.markdown("An Intelligent ML System to assess your **Loan Default Risk** using Financial and Credit Parameters.")
st.markdown("---")

# ---------- SVG helpers for Visualization Insights (unaligned intentionally) ----------
def svg_to_data_uri(svg_str: str) -> str:
    """Convert an SVG string to a base64 data URI."""
    encoded = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
    return "data:image/svg+xml;base64," + encoded

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
@st.cache_data
def load_data():
    path = "C:/Users/RITUL/OneDrive/Desktop/loan-default-dashboard/data/loan_data_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.warning("⚠️ loan_data_cleaned.csv not found in data/ directory.")
        return pd.DataFrame()

df = load_data()
filtered_df = df.copy()

# ---------- MODEL LOADING ----------
model = None
feature_columns = None
model_path = os.path.join("models", "gbm_pipeline.pkl")
columns_path = os.path.join("models", "model_columns.pkl")
if os.path.exists(model_path) and os.path.exists(columns_path):
    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(columns_path)
        if isinstance(feature_columns, dict) and "columns" in feature_columns:
            feature_columns = feature_columns["columns"]
    except Exception as e:
        st.warning(f"Model loading failed: {e}")

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
st.markdown("---")

# ---------- CUSTOMER FORM ----------
st.subheader("Customer Loan Application")

with st.form("customer_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        customer_name = st.text_input("Customer Name")
        customer_city = st.text_input("City")
        bank_name = st.text_input("Bank Name")
        loan_amnt_input = st.number_input("Loan Amount", min_value=0, value=10000, step=1000)
        annual_inc_input = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
        revol_util_input = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=50.0)
        mort_acc_input = st.number_input("Mortgage Accounts", min_value=0, value=0, step=1)
    with col2:
        term_options = sorted(df['term'].dropna().unique()) if 'term' in df.columns else [36, 60]
        grade_options = sorted(df['grade'].dropna().unique()) if 'grade' in df.columns else ["A", "B", "C"]
        term_input = st.selectbox("Term (months)", options=term_options)
        grade_input = st.selectbox("Grade", options=grade_options)
        zip_code_input = st.text_input("ZIP Code")
        interest_rate_input = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
        issue_d_month_input = st.selectbox("Issue Month", options=list(range(1, 13)))
        issue_d_year_input = st.number_input("Issue Year", min_value=2000, max_value=datetime.now().year, value=2020)
    submitted = st.form_submit_button("Submit & Save")

if submitted:
    # Store session state values
    st.session_state.update({
        "customer_name": customer_name,
        "customer_city": customer_city,
        "bank_name": bank_name,
        "loan_amnt": loan_amnt_input,
        "annual_inc": annual_inc_input,
        "revol_util": revol_util_input,
        "mort_acc": mort_acc_input,
        "term": term_input,
        "grade": grade_input,
        "zip_code": zip_code_input,
        "interest_rate": interest_rate_input,
        "issue_d_month": issue_d_month_input,
        "issue_d_year": issue_d_year_input
    })

    # Calculate Risk
    loan_amnt = loan_amnt_input
    annual_inc = annual_inc_input
    risk_label = "High Risk (Charged Off)" if loan_amnt > annual_inc * 0.4 else "Low Risk (Fully Paid)"
    risk_score = f"{min(100, int((loan_amnt / max(annual_inc, 1)) * 100))}%"
    status_color = "red" if "High Risk" in risk_label else "green"
    application_status = "❌ Application Requires Review" if "High Risk" in risk_label else "✅ Application Approved"

    # ===== Save prediction log =====
    log_file = "prediction_logs.csv"
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Customer Name": customer_name,
        "City": customer_city,
        "Bank Name": bank_name,
        "Loan Amount": loan_amnt_input,
        "Annual Income": annual_inc_input,
        "Revolving Utilization (%)": revol_util_input,
        "Mortgage Accounts": mort_acc_input,
        "Term": term_input,
        "Grade": grade_input,
        "ZIP Code": zip_code_input,
        "Interest Rate (%)": interest_rate_input,
        "Issue Month": issue_d_month_input,
        "Issue Year": issue_d_year_input,
        "Risk Level": risk_label,
        "Risk Score": risk_score,
        "Application Status": application_status
    }

    # Append or create CSV
    if os.path.exists(log_file):
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
    else:
        pd.DataFrame([log_data]).to_csv(log_file, mode='w', header=True, index=False)

    st.success(f"✅ Data saved to {log_file}")

    # Display Risk Profile & Insights
    risk_html = f"""
    <div style='text-align:center; font-size:18px;'>
        <b>Risk Level:</b> <span style='color:{status_color};'>{risk_label}</span><br>
        <b>Risk Score:</b> {risk_score}<br>
        <b>Application Status:</b> <span style='color:{status_color}; font-weight:bold;'>{application_status}</span>
    </div>
    """
    st.markdown(risk_html, unsafe_allow_html=True)

    # NOTE Section
    st.markdown("""
    <div class='note-box'>
    <b>NOTE:</b><br>
    <b>Risk Interpretation:</b> The simple heuristic compares loan size to income; higher loan-to-income suggests elevated risk.<br>
    <b>Suggested Actions:</b><br>
    - High Risk: Request supplemental documentation, reduce loan amount, or tighten terms.<br>
    - Low Risk: Consider fast-tracking with standard conditions.<br>
    <b>Credit Behavior Check:</b> Review revolving utilization and historical grade performance.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- INTERACTIVE VISUALS ----------
st.header("A Dive Into Loan Data: Dynamic Charts & Insights")

if df is not None:
    with st.expander("📈 Income vs Loan Amount by Status"):
        if {"annual_inc", "loan_amnt", "loan_status"}.issubset(df.columns):
            fig1 = px.scatter(df, x="annual_inc", y="loan_amnt", color="loan_status",
                            title="Income vs Loan Amount by Loan Status")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(
                """
**Insight:** Higher-income borrowers tend to request larger loans, but default behavior varies—look for clusters where large loans combined with relatively low income appear in the default group, indicating potential overleveraging.  

**Actionable:** Consider adding a loan_to_income_ratio feature or flagging borrowers whose loan size is disproportionate to income for additional review.
"""
            )
        else:
            st.info("Required columns missing for this visualization.")

    with st.expander("📊 Loan Amount Distribution"):
        if "loan_amnt" in filtered_df.columns:
            fig2 = px.histogram(
                filtered_df,
                x="loan_amnt",
                nbins=50,
                title="Loan Amount Distribution",
                labels={"loan_amnt": "Loan Amount"},
                opacity=0.8
            )
            fig2.update_traces(marker_line_color="black", marker_line_width=1)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                """
**Insight:** Concentration of loan sizes may reflect product tiers or underwriting thresholds.  

**Actionable:** Reevaluate criteria for rare loan-size bins showing defaults.
"""
            )
        else:
            st.info("Column 'loan_amnt' not present.")

    with st.expander("📌 DTI by Loan Status"):
        if {"dti", "loan_status"}.issubset(df.columns):
            fig3 = px.box(df, x="loan_status", y="dti", title="Debt-to-Income by Loan Status")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                """
**Insight:** Defaulted loans often show higher median DTI, signaling borrower overextension; wide variance or outliers in the default group may hide inconsistent risk behaviors.  

**Actionable:** Introduce soft DTI thresholds or composite risk scores combining DTI with income stability to preempt high-risk approvals.
"""
            )
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
                if selected_cat == "grade":
                    st.markdown(
                        """
**Insight:** Lower grades (e.g., E–G) tend to have higher default incidence, reinforcing the predictive power of the grading system.  

**Actionable:** Consider grade-aware pricing or tighter cutoffs for lower grades.
"""
                    )
                elif selected_cat == "term":
                    st.markdown(
                        """
**Insight:** Longer-term loans may reduce monthly burden but extend exposure; compare default rates between terms to balance risk vs borrower affordability.  

**Actionable:** Calibrate term-related thresholds or offer incentives for safer term choices.
"""
                    )
                elif selected_cat == "purpose":
                    st.markdown(
                        """
**Insight:** Different loan purposes carry distinct risk profiles; some purposes may systematically exhibit higher default rates.  

**Actionable:** Tailor underwriting rules or interest adjustments based on purpose segmentation.
"""
                    )
                elif selected_cat == "verification_status":
                    st.markdown(
                        """
**Insight:** Unverified income borrowers often show elevated default rates, highlighting the value of income verification.  

**Actionable:** Weight verification status in the risk score or require additional validation for unverified applications.
"""
                    )
                else:
                    st.markdown(
                        """
**Insight:** This view reveals how the selected categorical borrower attribute correlates with default behavior.  

**Actionable:** Identify categories with disproportionate default counts and consider targeted policy adjustments.
"""
                    )
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

            st.markdown(
                """
**Insight:** This matrix reveals how key numerical features relate to each other and to loan outcomes. Strong correlations (e.g., between income and loan amount or interest rate and default-prone profiles) can surface multicollinearity or compounded risk factors. Clusters visible by loan status help identify feature combinations that separate defaulted vs non-defaulted loans.  

**Actionable:** Use highly correlated pairs to engineer interaction features or to simplify the model (e.g., via dimensionality reduction), and consider creating composite risk signals where separability by status is clear.
"""
            )
        else:
            st.info("Please select at least two numerical features to visualize the pairwise relationships.")
else:
    st.warning("⚠️ Loan data not loaded; interactive visuals unavailable.")
st.markdown("---")

# ---------- TABLEAU DASHBOARD ----------
st.markdown("<h2 style='text-align:center;'>📊 Tableau Loan Default Dashboard</h2>", unsafe_allow_html=True)
left, center, right = st.columns([1, 3, 1])
with center:
    components.html("""
    <div style="display:flex; justify-content:center;">
    <div style='width:100%; max-width:1000px;'>
        <div class='tableauPlaceholder' id='viz1754289973057' style='position: relative; width:100%;'>
        <noscript>
            <a href='#'>
            <img alt='Dashboard 1' src='https://public.tableau.com/static/images/pr/projectdashboard_17542809820500/Dashboard1/1_rss.png' style='border: none; width:100%;' />
            </a>
        </noscript>
        <object class='tableauViz' style='display:none; width:100%;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='projectdashboard_17542809820500/Dashboard1' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/pr/projectdashboard_17542809820500/Dashboard1/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
            <param name='filter' value='publish=yes' />
        </object>
        </div>
    </div>
    </div>
    <script type='text/javascript'>
    (function() {
        var divElement = document.getElementById('viz1754289973057');
        var vizElement = divElement.getElementsByTagName('object')[0];
        function resizeViz() {
        var width = divElement.offsetWidth;
        vizElement.style.width = '100%';
        var height = Math.max(500, Math.round(width * 0.75));
        vizElement.style.height = height + 'px';
        }
        window.addEventListener('resize', resizeViz);
        resizeViz();
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    })();
    </script>
    """, height=800)
st.markdown("---")

# ---------- FOOTER ----------
st.markdown("""
<footer style="text-align:center; margin-top:3rem; color:#ddd;">
Loan Default Prediction System © 2025 | Built using Streamlit, Plotly & Tableau | ML Model: Gradient Boosting Pipeline | Developed by Ritul Gaikwad
</footer>
""", unsafe_allow_html=True)

