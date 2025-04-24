import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings


# --- Cached data loading ---
@st.cache_data
def load_sales_data():
    # Get the path to the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the relative path to the data file
    data_file_path = os.path.join(
        current_dir, "../data/processed_sales_data_for_discount.parquet"
    )

    # Load the data
    df = pd.read_parquet(data_file_path)
    df["discount_bin"] = (df["Discount%"] / 5).round() * 5
    return df


# --- Cached discount curve preprocessing ---
@st.cache_data
def preprocess_discount_curve(sales_data: pd.DataFrame):
    group = (
        sales_data.groupby(
            ["articleGroupDescription", "brandDescription", "discount_bin"]
        )
        .apply(
            lambda g: pd.Series(
                {
                    "monthly_qty": g["quantity"].sum() / 7,
                }
            )
        )
        .reset_index()
    )
    return group


def variableResistanceModel(context, alpha=0.05, max_discount=90):
    inventory = context.get("inventory_level", 100)
    days_remaining = context.get("days_remaining", 30)
    sales_velocity = context.get("sales_velocity", 1)

    target_sales_per_day = inventory / days_remaining

    for discount in range(0, max_discount + 1, 5):
        predicted_sales = sales_velocity * (1 + alpha * discount)
        if predicted_sales >= target_sales_per_day:
            return discount

    # Fallback: pick best performing discount
    best_discount = 0
    max_units_sold = -float("inf")
    for discount in range(max_discount, -1, -5):
        predicted_sales = sales_velocity * (1 + alpha * discount)
        if predicted_sales > max_units_sold:
            best_discount = discount
            max_units_sold = predicted_sales

    return best_discount


def compute_discount_scenario(
    article: str,
    brand: str,
    leftover_units: int,
    discount_curve: pd.DataFrame,
    remaining_days: int = None,
    remaining_months: int = None,
):
    if remaining_days:
        days_remaining = remaining_days
    elif remaining_months:
        days_remaining = remaining_months * 30
    else:
        days_remaining = 150  # default value

    # Filter data for the article-brand combo
    subset = discount_curve[
        (discount_curve["articleGroupDescription"] == article)
        & (discount_curve["brandDescription"] == brand)
        & (discount_curve["discount_bin"] > 0)
    ].copy()

    if subset.empty or len(subset) < 2:
        return {
            "articleGroupDescription": article,
            "brandDescription": brand,
            "leftover_qty": leftover_units,
            "best_discount_bin": np.nan,
            "expected_margin%": np.nan,
            "note": "Insufficient historical data",
        }

    # Estimate daily sales velocity from historical monthly_qty
    monthly_avg_sales = subset["monthly_qty"].mean()
    sales_velocity = monthly_avg_sales / 30  # convert to daily

    context = {
        "inventory_level": leftover_units,
        "days_remaining": days_remaining,
        "sales_velocity": sales_velocity,
    }

    best_discount = variableResistanceModel(context)

    return {
        "articleGroupDescription": article,
        "brandDescription": brand,
        "leftover_qty": leftover_units,
        "best_discount_bin": best_discount,
        "note": "Heuristic resistance model used",
    }


# --- Streamlit UI ---
st.set_page_config(page_title="Discount Optimization Tool", layout="centered")
st.title("📉 Discount Scenario Optimizer")

# Load & preprocess (cached)
sales_data = load_sales_data()
discount_curve = preprocess_discount_curve(sales_data)

# Sidebar Inputs
st.sidebar.header("📁 Input Parameters")

# Get unique combinations of article and brand
article_brand_combinations = (
    sales_data[["articleGroupDescription", "brandDescription"]]
    .dropna()
    .drop_duplicates()
)

# Select Article
selected_article = st.sidebar.selectbox(
    "Select Article Group",
    sorted(sales_data["articleGroupDescription"].dropna().unique()),
)

# Filter brands based on the selected article
valid_brands_for_article = article_brand_combinations[
    article_brand_combinations["articleGroupDescription"] == selected_article
]["brandDescription"].unique()
selected_brand = st.sidebar.selectbox("Select Brand", valid_brands_for_article)

# Filter articles based on the selected brand
valid_articles_for_brand = article_brand_combinations[
    article_brand_combinations["brandDescription"] == selected_brand
]["articleGroupDescription"].unique()

# Remaining units, days, and months inputs
leftover_units = st.sidebar.number_input("Remaining Units", min_value=1, value=100)
remaining_days = st.sidebar.number_input("Remaining Days", min_value=0, value=100)
remaining_months = st.sidebar.number_input(
    "OR Remaining Months", min_value=0.0, value=0.0, step=0.1
)

if st.sidebar.button("🔍 Calculate Optimal Discount"):
    result = compute_discount_scenario(
        article=selected_article,
        brand=selected_brand,
        leftover_units=leftover_units,
        remaining_days=remaining_days if remaining_days > 0 else None,
        remaining_months=remaining_months if remaining_months > 0 else None,
        discount_curve=discount_curve,
    )

    st.markdown("---")
    st.header("📊 Suggested Discount Scenario")

    if result.get("best_discount_bin") is not np.nan:
        st.success("🎯 Optimal Discount Recommendation")
        col1, col2 = st.columns(2)
        col1.metric("📉 Discount", f"{int(result['best_discount_bin'])}%")
        col2.metric("📦 Stock Left", f"{int(result['leftover_qty'])} units")

        st.markdown(f"""
        **🧾 Article:** `{result["articleGroupDescription"]}`
        **🏷️ Brand:** `{result["brandDescription"]}`
        """)

    else:
        st.error("⚠️ Could not generate a reliable recommendation.")
        st.markdown(f"**Reason:** {result.get('note', 'Unknown issue')}")
