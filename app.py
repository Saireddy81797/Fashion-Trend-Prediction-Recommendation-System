#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Fashion Trend Prediction & Recommendation System")

st.subheader("Top Trending Products")
st.write(top_trending)

selected_product = st.selectbox("Select Product for Recommendations", df['product_id'].unique())
st.subheader(f"Recommended Products similar to {selected_product}")
st.write(recommend_products(selected_product))

st.subheader("Category-wise Sales Distribution")
fig, ax = plt.subplots()
sns.barplot(data=df.groupby('category')['sales_last_month'].sum().reset_index(), 
            x='category', y='sales_last_month', ax=ax)
st.pyplot(fig)

