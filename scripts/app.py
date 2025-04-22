import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os


# --- Logistic function for curve fitting ---
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# --- Weighted average utility ---
def weighted_avg(group, value_col, weight_col="quantity"):
    d, w = group[value_col], group[weight_col]
    return (d * w).sum() / w.sum() if w.sum() != 0 else np.nan


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
    df = df[df["Inhouse_Brand"] == True].copy()
    df["salesDate"] = pd.to_datetime(df["salesDate"])
    df = df[df["salesDate"].dt.month.isin([3, 4, 5, 6, 7, 8, 9])]
    df["discount_bin"] = (df["Discount%"] / 5).round() * 5
    return df


# --- Cached discount curve preprocessing ---
@st.cache_data
def preprocess_discount_curve(
    sales_data: pd.DataFrame, min_margin_required: float = 50
):
    group = (
        sales_data.groupby(
            ["articleGroupDescription", "brandDescription", "discount_bin"]
        )
        .apply(
            lambda g: pd.Series(
                {
                    "monthly_qty": g["quantity"].sum() / 7,
                    "avg_margin%": (
                        (
                            weighted_avg(g, "retailPrice")
                            - weighted_avg(g, "purchasePrice_barcode")
                        )
                        / weighted_avg(g, "retailPrice")
                        * 100
                    ),
                }
            )
        )
        .reset_index()
    )
    return group[group["avg_margin%"] >= min_margin_required]


# --- Discount suggestion logic ---
def compute_discount_scenario(
    article: str,
    brand: str,
    leftover_units: int,
    discount_curve: pd.DataFrame,
    remaining_days: int = None,
    remaining_months: int = None,
    min_margin_required: float = 50,
):
    if remaining_days:
        remaining_months = remaining_days / 30
    elif not remaining_months:
        remaining_months = 5

    subset = discount_curve[
        (discount_curve["articleGroupDescription"] == article)
        & (discount_curve["brandDescription"] == brand)
        & (discount_curve["discount_bin"] > 0)
    ].copy()

    if len(subset) < 4:
        return {
            "articleGroupDescription": article,
            "brandDescription": brand,
            "leftover_qty": leftover_units,
            "best_discount_bin": np.nan,
            "expected_margin%": np.nan,
            "note": "Insufficient data after filtering",
        }

    X = subset["discount_bin"].values
    y = subset["monthly_qty"].values

    try:
        popt, _ = curve_fit(logistic, X, y, p0=[max(y), 0.3, np.median(X)], maxfev=5000)
        L, k, x0 = popt

        target_qty_per_month = leftover_units / remaining_months
        ratio = L / target_qty_per_month

        if ratio <= 1:
            required_discount = 80
            note = "Target exceeds modeled capacity — max discount applied"
        else:
            discount_needed = x0 - (1 / k) * np.log(ratio - 1)
            required_discount = np.clip(discount_needed, 0, 90)
            required_discount = round(required_discount / 5) * 5
            note = ""

        margin_match = subset[subset["discount_bin"] == required_discount]
        if not margin_match.empty:
            margin = margin_match["avg_margin%"].values[0]
        else:
            closest = subset.iloc[
                (subset["discount_bin"] - required_discount).abs().argsort()[:1]
            ]
            margin = closest["avg_margin%"].values[0]

        if margin < min_margin_required:
            note = f"Margin {margin:.2f}% below min required {min_margin_required}%"

        return {
            "articleGroupDescription": article,
            "brandDescription": brand,
            "leftover_qty": leftover_units,
            "best_discount_bin": required_discount,
            "expected_margin%": margin,
            "note": note,
        }

    except RuntimeError:
        return {
            "articleGroupDescription": article,
            "brandDescription": brand,
            "leftover_qty": leftover_units,
            "best_discount_bin": np.nan,
            "expected_margin%": np.nan,
            "note": "Model fitting failed",
        }


# --- Streamlit UI ---
st.set_page_config(page_title="Discount Optimization Tool", layout="centered")
st.title("📉 Discount Scenario Optimizer")

# Load & preprocess (cached)
sales_data = load_sales_data()
discount_curve = preprocess_discount_curve(sales_data)

# Sidebar Inputs
st.sidebar.header("📁 Input Parameters")
articles = sales_data["articleGroupDescription"].dropna().unique()
brands = sales_data["brandDescription"].dropna().unique()
article = st.sidebar.selectbox("Select Article Group", sorted(articles))
brand = st.sidebar.selectbox("Select Brand", sorted(brands))
leftover_units = st.sidebar.number_input("Remaining Units", min_value=1, value=500)
remaining_days = st.sidebar.number_input("Remaining Days", min_value=0, value=150)
remaining_months = st.sidebar.number_input(
    "OR Remaining Months", min_value=0.0, value=0.0, step=0.1
)

if st.sidebar.button("🔍 Calculate Optimal Discount"):
    result = compute_discount_scenario(
        article=article,
        brand=brand,
        leftover_units=leftover_units,
        remaining_days=remaining_days if remaining_days > 0 else None,
        remaining_months=remaining_months if remaining_months > 0 else None,
        discount_curve=discount_curve,
    )

    st.markdown("---")
    st.header("📊 Suggested Discount Scenario")

    if result.get("best_discount_bin") is not np.nan:
        st.success("🎯 Optimal Discount Recommendation")
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 Discount", f"{int(result['best_discount_bin'])}%")
        col2.metric("💰 Expected Margin", f"{result['expected_margin%']:.2f}%")
        col3.metric("📦 Stock Left", f"{int(result['leftover_qty'])} units")

        st.markdown(f"""
        **🧾 Article:** `{result["articleGroupDescription"]}`  
        **🏷️ Brand:** `{result["brandDescription"]}`  
        """)

    else:
        st.error("⚠️ Could not generate a reliable recommendation.")
        st.markdown(f"**Reason:** {result.get('note', 'Unknown issue')}")
