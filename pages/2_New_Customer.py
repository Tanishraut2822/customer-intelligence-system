import streamlit as st
import pandas as pd
import pickle

# =====================================
# LOAD MODELS
# =====================================
kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
future_pipeline = pickle.load(open('future_pipeline.pkl', 'rb'))
segment_map = pickle.load(open('segment_map.pkl', 'rb'))
customer_product = pickle.load(open('customer_product.pkl', 'rb'))
customer_similarity_df = pickle.load(open('customer_similarity.pkl', 'rb'))

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
# PAGE UI
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
# INPUTS
# =====================================
st.sidebar.header("Enter RFM Values")

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
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })

    # Segment
    seg_id = kmeans.predict(
        scaler.transform(new_customer)
    )[0]
    segment_name = segment_map[seg_id]

    # Future purchase
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

    # Recommendations (fallback logic)
    st.subheader("🛍 Recommended Products")

    # Use a reference customer for similarity
    sample_customer_id = customer_product.index[0]
    recommendations = recommend_products(sample_customer_id)

    if recommendations:
        for i, product in enumerate(recommendations, 1):
            st.write(f"{i}. {product}")
    else:
        st.write("No recommendations available")
