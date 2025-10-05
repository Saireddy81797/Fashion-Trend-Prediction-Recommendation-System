#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Fashion Trend Prediction & Recommendation System")

user_id = st.selectbox("Select User", df['user_id'].unique())

st.subheader("Top Trending Products")
st.write(top_trending)

st.subheader(f"Personalized Recommendations for {user_id}")
st.write(recommend_products(model, dataset, user_id))

st.subheader("Category-wise Sales Distribution")
fig, ax = plt.subplots()
sns.barplot(data=df.groupby('category')['sales_last_month'].sum().reset_index(), x='category', y='sales_last_month', ax=ax)
st.pyplot(fig)

