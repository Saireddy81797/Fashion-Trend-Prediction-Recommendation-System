#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Fashion Trend & Recommendation", layout="wide")

# --------------------------
# Load Data
# --------------------------
df = pd.read_csv("fashion_trend_data_small.csv")

st.title("👗 Fashion Trend Prediction & Recommendation System")
st.markdown("**Discover trending products and get recommendations for similar fashion items!**")

# --------------------------
# Sidebar: Select product
# --------------------------
st.sidebar.header("Filters")
selected_product = st.sidebar.selectbox("Select a Product", df['product_id'].tolist())
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# --------------------------
# Compute Top Trending Products
# --------------------------
top_trending = df.sort_values(by="sales_last_month", ascending=False).head(10)

st.subheader("🔥 Top 10 Trending Products")
st.bar_chart(top_trending.set_index('product_id')['sales_last_month'])

# --------------------------
# TF-IDF + Cosine Similarity
# --------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
product_indices = pd.Series(df.index, index=df['product_id'])

# --------------------------
# Recommendation Function
# --------------------------
def recommend_products(product_id, top_n=5):
    idx = product_indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip the product itself
    product_indices_list = [i[0] for i in sim_scores]
    return df['product_id'].iloc[product_indices_list].tolist()

# --------------------------
# Show Recommendations
# --------------------------
st.subheader(f"✅ Recommended Products for {selected_product}")
recommended = recommend_products(selected_product, top_n)
st.write(recommended)

# --------------------------
# Professional Plots
# --------------------------
st.subheader("📊 Category-wise Sales")
category_sales = df.groupby('category')['sales_last_month'].sum().sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(8,4))
sns.barplot(x=category_sales.index, y=category_sales.values, palette='coolwarm', ax=ax1)
ax1.set_ylabel("Total Sales Last Month")
ax1.set_xlabel("Category")
st.pyplot(fig1)

st.subheader("📈 Sales Distribution")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.histplot(df['sales_last_month'], bins=10, kde=True, color='green', ax=ax2)
ax2.set_xlabel("Sales Last Month")
st.pyplot(fig2)

st.subheader("🟢 Product Similarity Heatmap")
fig3, ax3 = plt.subplots(figsize=(10,8))
sns.heatmap(cosine_sim, cmap='YlGnBu', ax=ax3)
st.pyplot(fig3)

