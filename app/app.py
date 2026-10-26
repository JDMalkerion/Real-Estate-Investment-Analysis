"""
Real Estate Investment Advisor - Streamlit Web Application.
Interactive property valuation, investment screening, and market insights.
"""

from pathlib import Path
import sys

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    BEST_CLASSIFIER_PATH,
    BEST_REGRESSOR_PATH,
    CLEAN_DATA_PATH,
    FEATURES_CLASSIFIER_PATH,
    FEATURES_REGRESSOR_PATH,
    PREPROCESSOR_CLASSIFIER_PATH,
    PREPROCESSOR_REGRESSOR_PATH,
    PUBLIC_TRANSPORT_MAP,
)

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design styling
ACCENT_COLOR = "#2563EB"
st.markdown(
    f"""
    <style>
    .main-title {{
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }}
    .sub-title {{
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }}
    .stat-card {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    .stat-val {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}
    .stat-lbl {{
        font-size: 0.85rem;
        color: #64748B;
    }}
    .verdict-good {{
        background: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        font-weight: 600;
        color: #065F46;
    }}
    .verdict-bad {{
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        font-weight: 600;
        color: #92400E;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load cleaned dataset from Data/housing_clean.csv."""
    if not CLEAN_DATA_PATH.exists():
        st.error(f"Data file not found at {CLEAN_DATA_PATH}. Please run preprocessing first.")
        st.stop()
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_resource
def load_models():
    """Load trained models, preprocessors, and feature schemas."""
    clf = joblib.load(BEST_CLASSIFIER_PATH)
    preproc_clf = joblib.load(PREPROCESSOR_CLASSIFIER_PATH)
    feats_clf = joblib.load(FEATURES_CLASSIFIER_PATH)

    reg = joblib.load(BEST_REGRESSOR_PATH)
    preproc_reg = joblib.load(PREPROCESSOR_REGRESSOR_PATH)
    feats_reg = joblib.load(FEATURES_REGRESSOR_PATH)

    return clf, preproc_clf, feats_clf, reg, preproc_reg, feats_reg


def main() -> None:
    """Main Streamlit application logic."""
    df = load_data()
    clf, preproc_clf, feats_clf, reg, preproc_reg, feats_reg = load_models()

    # Title & Header
    st.markdown("<div class='main-title'>🏢 Real Estate Investment Advisor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Data-driven property screening, valuation forecasting, and location intelligence across India.</div>",
        unsafe_allow_html=True,
    )

    # Mandatory Methodology Callout
    st.info(
        "ℹ️ **Model Assumption Note**: Labels and 5-year prices are model-constructed from assumed growth rules, not observed returns. This tool serves as an analytical decision aid and not formal financial advice."
    )

    # Sidebar Filter Controls
    st.sidebar.header("🔍 Property Explorer Filters")

    # State filter
    all_states = sorted(df["State"].unique().tolist())
    selected_states = st.sidebar.multiselect("State", options=all_states, default=[])

    # City filter (dynamically filtered by state if selected)
    city_options = df[df["State"].isin(selected_states)]["City"].unique().tolist() if selected_states else df["City"].unique().tolist()
    city_options = sorted(city_options)
    selected_cities = st.sidebar.multiselect("City", options=city_options, default=[])

    # Locality filter
    loc_df = df
    if selected_cities:
        loc_df = loc_df[loc_df["City"].isin(selected_cities)]
    elif selected_states:
        loc_df = loc_df[loc_df["State"].isin(selected_states)]
    locality_options = sorted(loc_df["Locality"].unique().tolist())
    selected_localities = st.sidebar.multiselect("Locality", options=locality_options, default=[])

    # Price range
    min_p, max_p = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
    price_range = st.sidebar.slider(
        "Price Range (₹ Lakhs)",
        min_value=min_p,
        max_value=max_p,
        value=(min_p, max_p),
        step=5.0,
    )

    # BHK filter
    all_bhk = sorted(df["BHK"].unique().tolist())
    selected_bhk = st.sidebar.multiselect("BHK Configuration", options=all_bhk, default=[])

    # Property Type filter
    all_prop_types = sorted(df["Property_Type"].unique().tolist())
    selected_prop_types = st.sidebar.multiselect("Property Type", options=all_prop_types, default=[])

    # Size range
    min_s, max_s = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
    size_range = st.sidebar.slider(
        "Size Range (SqFt)",
        min_value=min_s,
        max_value=max_s,
        value=(min_s, max_s),
        step=100,
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]
    if selected_cities:
        filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]
    if selected_localities:
        filtered_df = filtered_df[filtered_df["Locality"].isin(selected_localities)]
    filtered_df = filtered_df[
        (filtered_df["Price_in_Lakhs"] >= price_range[0])
        & (filtered_df["Price_in_Lakhs"] <= price_range[1])
    ]
    if selected_bhk:
        filtered_df = filtered_df[filtered_df["BHK"].isin(selected_bhk)]
    if selected_prop_types:
        filtered_df = filtered_df[filtered_df["Property_Type"].isin(selected_prop_types)]
    filtered_df = filtered_df[
        (filtered_df["Size_in_SqFt"] >= size_range[0])
        & (filtered_df["Size_in_SqFt"] <= size_range[1])
    ]

    # Main Tabs
    tab_overview, tab_predict, tab_insights = st.tabs([
        "📋 Market Inventory",
        "🎯 Property Advisor (Prediction)",
        "📊 Market Insights & Model Explanations",
    ])

    # TAB 1: Market Inventory & Filters
    with tab_overview:
        st.subheader("Filtered Property Listings")
        if len(filtered_df) == 0:
            st.warning("⚠️ No properties match the selected filter criteria. Please broaden your selection.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            good_inv_pct = (filtered_df["Good_Investment"].mean() * 100) if "Good_Investment" in filtered_df else 0
            with c1:
                st.markdown(
                    f"<div class='stat-card'><div class='stat-val'>{len(filtered_df):,}</div><div class='stat-lbl'>Matching Properties</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='stat-card'><div class='stat-val'>₹{filtered_df['Price_in_Lakhs'].median():.1f}L</div><div class='stat-lbl'>Median Price</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='stat-card'><div class='stat-val'>₹{filtered_df['Price_per_SqFt'].median():,.0f}</div><div class='stat-lbl'>Median ₹/SqFt</div></div>",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"<div class='stat-card'><div class='stat-val'>{good_inv_pct:.1f}%</div><div class='stat-lbl'>Good Investment Share</div></div>",
                    unsafe_allow_html=True,
                )

            st.write("")
            display_cols = [
                "ID", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
                "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
                "Furnished_Status", "Availability_Status", "Good_Investment"
            ]
            st.dataframe(
                filtered_df[display_cols].head(500),
                use_container_width=True,
                height=350,
            )
            st.caption(f"Showing up to 500 of {len(filtered_df):,} filtered records.")

    # TAB 2: Prediction Form
    with tab_predict:
        st.subheader("Evaluate Property Investment Potential")
        st.write("Enter details of a property to evaluate its investment potential and forecast 5-year valuation:")

        with st.form("investment_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                form_state = st.selectbox("State", options=sorted(df["State"].unique().tolist()), index=0)
                city_subset = sorted(df[df["State"] == form_state]["City"].unique().tolist())
                form_city = st.selectbox("City", options=city_subset if city_subset else sorted(df["City"].unique().tolist()), index=0)
                form_locality = st.selectbox("Locality", options=sorted(df["Locality"].unique().tolist()), index=0)
                form_prop_type = st.selectbox("Property Type", options=["Apartment", "Independent House", "Villa"], index=0)
                form_bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=2)
                form_size = st.number_input("Size in SqFt", min_value=300, max_value=8000, value=1800, step=50)

            with col2:
                form_price = st.number_input("Current Price (₹ Lakhs)", min_value=5.0, max_value=1000.0, value=150.0, step=5.0)
                form_age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=5, step=1)
                form_floor = st.number_input("Floor Number", min_value=0, max_value=40, value=4, step=1)
                form_total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=40, value=12, step=1)
                form_furnished = st.selectbox("Furnished Status", options=["Furnished", "Semi-furnished", "Unfurnished"], index=1)
                form_avail = st.selectbox("Availability Status", options=["Ready_to_Move", "Under_Construction"], index=0)

            with col3:
                form_schools = st.slider("Nearby Schools Count", min_value=1, max_value=10, value=5)
                form_hospitals = st.slider("Nearby Hospitals Count", min_value=1, max_value=10, value=4)
                form_transit = st.selectbox("Public Transport Accessibility", options=["Low", "Medium", "High"], index=2)
                form_parking = st.selectbox("Parking Space", options=["Yes", "No"], index=0)
                form_security = st.selectbox("Security Guard/System", options=["Yes", "No"], index=0)
                form_facing = st.selectbox("Facing Direction", options=["East", "North", "West", "South"], index=0)
                form_owner = st.selectbox("Owner Type", options=["Owner", "Builder", "Broker"], index=0)
                form_amenities = st.multiselect(
                    "Amenities Available",
                    options=["Clubhouse", "Garden", "Gym", "Playground", "Pool"],
                    default=["Clubhouse", "Gym", "Garden"],
                )

            submitted = st.form_submit_button("🚀 Run Investment Analysis", use_container_width=True)

        if submitted:
            # Construct input dictionary for model evaluation
            total_floors_fixed = max(form_total_floors, form_floor)
            floor_ratio_val = form_floor / max(1, total_floors_fixed)
            amenity_count_val = len(form_amenities)

            # Compute infrastructure score for regressor
            norm_sc = (form_schools - 1.0) / 9.0
            norm_ho = (form_hospitals - 1.0) / 9.0
            norm_tr = PUBLIC_TRANSPORT_MAP.get(form_transit, 0.5)
            infra_score_val = (norm_sc + norm_ho + norm_tr) / 3.0

            # Determine age bucket
            if form_age <= 5:
                age_bkt = "0-5 Years"
            elif form_age <= 10:
                age_bkt = "6-10 Years"
            elif form_age <= 20:
                age_bkt = "11-20 Years"
            else:
                age_bkt = "20+ Years"

            input_dict = {
                "State": form_state,
                "City": form_city,
                "Locality": form_locality,
                "Property_Type": form_prop_type,
                "BHK": form_bhk,
                "Size_in_SqFt": form_size,
                "Price_in_Lakhs": form_price,
                "Year_Built": 2025 - form_age,
                "Furnished_Status": form_furnished,
                "Floor_No": form_floor,
                "Total_Floors": total_floors_fixed,
                "Age_of_Property": form_age,
                "Nearby_Schools": form_schools,
                "Nearby_Hospitals": form_hospitals,
                "Public_Transport_Accessibility": form_transit,
                "Parking_Space": form_parking,
                "Security": form_security,
                "Facing": form_facing,
                "Owner_Type": form_owner,
                "Availability_Status": form_avail,
                "amenity_count": amenity_count_val,
                "security_feature_count": 1 if form_security == "Yes" else 0,
                "floor_ratio": floor_ratio_val,
                "school_density_score": norm_sc,
                "infrastructure_score": infra_score_val,
                "age_bucket": age_bkt,
            }
            sample_row = pd.DataFrame([input_dict])

            # 1. Classifier Prediction
            sample_clf_proc = preproc_clf.transform(sample_row[feats_clf])
            pred_class = int(clf.predict(sample_clf_proc)[0])
            pred_prob = float(clf.predict_proba(sample_clf_proc)[0][1])

            # 2. Regressor Prediction
            sample_reg_proc = preproc_reg.transform(sample_row[feats_reg])
            pred_future_price = float(reg.predict(sample_reg_proc)[0])

            # Implied appreciation & growth
            appreciation_abs = pred_future_price - form_price
            appreciation_pct = (appreciation_abs / form_price) * 100
            cagr = ((pred_future_price / form_price) ** (1.0 / 5.0) - 1.0) * 100

            st.write("---")
            st.subheader("Analysis & Recommendation Verdict")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("#### Investment Classification Verdict")
                if pred_class == 1:
                    st.markdown(
                        f"""
                        <div class='verdict-good'>
                            ✅ <strong>RECOMMENDED: GOOD INVESTMENT POTENTIAL</strong><br>
                            Confidence Score: {pred_prob * 100:.1f}%<br>
                            <small>Meets favorable price-to-size positioning and strong local infrastructure criteria.</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='verdict-bad'>
                            ⚠️ <strong>CAUTIOUS: BELOW TARGET INVESTMENT THRESHOLD</strong><br>
                            Confidence Score: {(1.0 - pred_prob) * 100:.1f}% (Chance of Good Investment: {pred_prob * 100:.1f}%)<br>
                            <small>Pricing or property infrastructure profile does not meet target return thresholds.</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Confidence Gauge / Progress Bar
                st.write("")
                st.write(f"**Investment Attractiveness Probability: {pred_prob * 100:.1f}%**")
                st.progress(pred_prob)

            with res_col2:
                st.markdown("#### 5-Year Valuation Forecast")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(
                        label="Current Valuation",
                        value=f"₹{form_price:.2f}L",
                        help="Spot property purchase price entered.",
                    )
                with m2:
                    st.metric(
                        label="Forecasted 5-Year Price",
                        value=f"₹{pred_future_price:.2f}L",
                        delta=f"+₹{appreciation_abs:.2f}L ({appreciation_pct:.1f}%)",
                        delta_color="normal",
                    )

                st.info(
                    f"📈 **Implied Annual Growth Rate (CAGR)**: **{cagr:.2f}% per year** over the 5-year holding horizon."
                )

    # TAB 3: Market Insights & Visualizations
    with tab_insights:
        st.subheader("Market Insights & Feature Importance")

        st.markdown("#### 1. Location-wise Pricing Analysis")
        ins_col1, ins_col2 = st.columns(2)

        with ins_col1:
            city_avg_sqft = df.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False).head(15).reset_index()
            fig_city = px.bar(
                city_avg_sqft,
                x="City",
                y="Price_per_SqFt",
                title="Top 15 Cities by Average ₹/SqFt",
                labels={"Price_per_SqFt": "Avg Price per SqFt (₹)"},
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_city.update_layout(yaxis=dict(rangemode="tozero"), template="plotly_white")
            st.plotly_chart(fig_city, use_container_width=True)

        with ins_col2:
            state_avg_sqft = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).head(15).reset_index()
            fig_state = px.bar(
                state_avg_sqft,
                x="State",
                y="Price_per_SqFt",
                title="Top 15 States by Average ₹/SqFt",
                labels={"Price_per_SqFt": "Avg Price per SqFt (₹)"},
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_state.update_layout(yaxis=dict(rangemode="tozero"), template="plotly_white")
            st.plotly_chart(fig_state, use_container_width=True)

        st.markdown("#### 2. Vintage Trends & Infrastructure Impact")
        ins_col3, ins_col4 = st.columns(2)

        with ins_col3:
            age_price = df.groupby("Age_of_Property")["Price_per_SqFt"].mean().reset_index()
            fig_age = px.line(
                age_price,
                x="Age_of_Property",
                y="Price_per_SqFt",
                markers=True,
                title="Property Age vs Average ₹/SqFt",
                labels={"Age_of_Property": "Age (Years)", "Price_per_SqFt": "Mean Price per SqFt (₹)"},
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_age.update_layout(yaxis=dict(rangemode="tozero"), template="plotly_white")
            st.plotly_chart(fig_age, use_container_width=True)

        with ins_col4:
            infra_bins = pd.cut(df["infrastructure_score"], bins=5).astype(str)
            infra_summary = df.groupby(infra_bins)["Good_Investment"].mean().reset_index()
            infra_summary.columns = ["Infra_Score_Bucket", "Good_Inv_Ratio"]
            infra_summary["Good_Inv_Pct"] = infra_summary["Good_Inv_Ratio"] * 100
            fig_infra = px.bar(
                infra_summary,
                x="Infra_Score_Bucket",
                y="Good_Inv_Pct",
                title="Infrastructure Score vs Good Investment Qualification (%)",
                labels={"Infra_Score_Bucket": "Infrastructure Score Tier", "Good_Inv_Pct": "Good Investment (%)"},
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_infra.update_layout(yaxis=dict(rangemode="tozero"), template="plotly_white")
            st.plotly_chart(fig_infra, use_container_width=True)

        st.markdown("#### 3. Top Feature Importances (Trained Models)")
        fi_col1, fi_col2 = st.columns(2)

        with fi_col1:
            st.markdown("**Investment Classifier (XGBoost) Drivers**")
            clf_fi = clf.feature_importances_
            # Reconstruct feature names
            ohe = preproc_clf.named_transformers_["low_card"]
            low_cats = [
                "State", "City", "Property_Type", "Parking_Space",
                "Security", "Facing", "Owner_Type", "Availability_Status"
            ]
            names_clf = list(ohe.get_feature_names_out(low_cats)) + ["Locality"] + [
                "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Floor_No", "Total_Floors",
                "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "amenity_count", "floor_ratio"
            ]
            idx_c = np.argsort(clf_fi)[::-1][:10]
            top_clf_df = pd.DataFrame({
                "Feature": [names_clf[i] for i in idx_c][::-1],
                "Importance": clf_fi[idx_c][::-1],
            })
            fig_fi_c = px.bar(
                top_clf_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_fi_c.update_layout(xaxis=dict(rangemode="tozero"), template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi_c, use_container_width=True)

        with fi_col2:
            st.markdown("**5-Year Price Regressor (XGBoost) Drivers**")
            reg_fi = reg.feature_importances_
            ohe_r = preproc_reg.named_transformers_["low_card"]
            low_cats_r = [
                "State", "City", "Property_Type", "Furnished_Status",
                "Public_Transport_Accessibility", "Parking_Space", "Security",
                "Facing", "Owner_Type", "Availability_Status", "age_bucket"
            ]
            names_reg = list(ohe_r.get_feature_names_out(low_cats_r)) + ["Locality"] + [
                "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No", "Total_Floors",
                "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "infrastructure_score",
                "amenity_count", "security_feature_count", "floor_ratio", "school_density_score"
            ]
            idx_r = np.argsort(reg_fi)[::-1][:10]
            top_reg_df = pd.DataFrame({
                "Feature": [names_reg[i] for i in idx_r][::-1],
                "Importance": reg_fi[idx_r][::-1],
            })
            fig_fi_r = px.bar(
                top_reg_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_fi_r.update_layout(xaxis=dict(rangemode="tozero"), template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi_r, use_container_width=True)


if __name__ == "__main__":
    main()
