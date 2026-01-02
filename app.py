import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Customer Intelligence System",
    layout="centered"
)

st.title("🛒 Customer Intelligence System")
st.write(
    "End-to-end customer segmentation, future purchase prediction, "
    "and product recommendation system."
)

# =====================================
# LOAD TRAINED MODELS (SMALL PKL FILES)
# =====================================
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
future_pipeline = pickle.load(open("future_pipeline.pkl", "rb"))
segment_map = pickle.load(open("segment_map.pkl", "rb"))

# =====================================
# LOAD TRANSACTION DATA
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
# BUILD RECOMMENDER SYSTEM (CACHED)
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
# CREATE RFM TABLE FROM DATA
# =====================================
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# =====================================
# SIDEBAR – CUSTOMER SELECTION
# =====================================
st.sidebar.header("Select Customer")

customer_id = st.sidebar.selectbox(
    "Choose Customer ID",
    rfm["CustomerID"].unique()
)

selected_customer = rfm[
    rfm["CustomerID"] == customer_id
][["Recency", "Frequency", "Monetary"]]

st.sidebar.subheader("RFM Values")
st.sidebar.write(selected_customer)

# =====================================
# PREDICTION BUTTON
# =====================================
if st.sidebar.button("Predict Customer Insights"):

    # -----------------------------
    # CUSTOMER SEGMENT PREDICTION
    # -----------------------------
    seg_id = kmeans.predict(
        scaler.transform(selected_customer)
    )[0]

    segment_name = segment_map[seg_id]

    # -----------------------------
    # FUTURE PURCHASE PREDICTION
    # -----------------------------
    purchase_prob = future_pipeline.predict_proba(
        selected_customer
    )[0][1]

    # =================================
    # DISPLAY RESULTS
    # =================================
    st.subheader("📊 Customer Insights")

    st.write("**Customer ID:**", customer_id)
    st.write("**Customer Segment:**", segment_name)
    st.write(
        "**Purchase Probability (Next 30 Days):**",
        round(purchase_prob, 2)
    )

    if purchase_prob >= 0.5:
        st.success("Likely to Purchase Again")
    else:
        st.warning("Unlikely to Purchase Again")

    # =================================
    # PRODUCT RECOMMENDATIONS
    # =================================
    st.subheader("🛍 Recommended Products")

    recommendations = recommend_products(customer_id)

    if recommendations:
        for i, product in enumerate(recommendations, 1):
            st.write(f"{i}. {product}")
    else:
        st.write("No recommendations available")
