import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="New Customer Prediction",
    layout="centered"
)

st.title("🆕 New Customer Prediction")
st.write(
    "Predict customer segment, future purchase behavior, "
    "and product recommendations for a new customer."
)

# =====================================
# LOAD TRAINED MODELS (SMALL PKL FILES)
# =====================================
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
future_pipeline = pickle.load(open("future_pipeline.pkl", "rb"))
segment_map = pickle.load(open("segment_map.pkl", "rb"))

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.dropna(subset=["InvoiceDate"], inplace=True)
    return df

df = load_data()

# =====================================
# BUILD RECOMMENDER (CACHED)
# =====================================
@st.cache_resource
def build_recommender(df):

    customer_product = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity = cosine_similarity(customer_product)

    customer_similarity_df = pd.DataFrame(
        similarity,
        index=customer_product.index,
        columns=customer_product.index
    )

    return customer_product, customer_similarity_df


customer_product, customer_similarity_df = build_recommender(df)

# =====================================
# RECOMMENDATION FUNCTION
# =====================================
def recommend_products(customer_id, top_n=4):
    customer_id = str(customer_id)

    if customer_id not in customer_similarity_df.columns:
        return []

    similar_customers = (
        customer_similarity_df.loc[customer_id]
        .sort_values(ascending=False)
        .iloc[1:6]
    )

    product_scores = customer_product.loc[similar_customers.index].mean()
    purchased = customer_product.loc[customer_id]

    product_scores = product_scores[purchased == 0]

    return product_scores.sort_values(ascending=False).head(top_n).index.tolist()

# =====================================
# USER INPUT (NEW CUSTOMER)
# =====================================
st.sidebar.header("Enter New Customer RFM Values")

recency = st.sidebar.number_input(
    "Recency (days since last purchase)", min_value=0
)
frequency = st.sidebar.number_input(
    "Frequency (number of purchases)", min_value=0
)
monetary = st.sidebar.number_input(
    "Monetary Value (total spend)", min_value=0.0
)

# =====================================
# PREDICTION
# =====================================
if st.sidebar.button("Predict New Customer"):

    new_customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    # Segment prediction
    seg_id = kmeans.predict(
        scaler.transform(new_customer)
    )[0]

    segment_name = segment_map[seg_id]

    # Future purchase prediction
    purchase_prob = future_pipeline.predict_proba(
        new_customer
    )[0][1]

    st.subheader("📊 Prediction Results")

    st.write("**Customer Segment:**", segment_name)
    st.write(
        "**Purchase Probability (Next 30 Days):**",
        round(purchase_prob, 2)
    )

    if purchase_prob >= 0.5:
        st.success("Likely to Purchase Again")
    else:
        st.warning("Unlikely to Purchase Again")

    # Product recommendations (reference-based)
    st.subheader("🛍 Recommended Products")

    sample_customer_id = customer_product.index[0]
    recommendations = recommend_products(sample_customer_id)

    if recommendations:
        for i, product in enumerate(recommendations, 1):
            st.write(f"{i}. {product}")
    else:
        st.write("No recommendations available")
