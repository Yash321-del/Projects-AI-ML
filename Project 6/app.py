import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Iris Flower Clustering",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower Clustering using K-Means")

st.write("""
This application clusters the Iris flowers using only:

- Petal Length
- Petal Width

It also demonstrates the Elbow Method to find the optimal value of K.
""")

# Load Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

species = iris.target

df["Species"] = species

# Keep only required columns
df = df[[
    "petal length (cm)",
    "petal width (cm)",
    "Species"
]]

st.subheader("Dataset")

st.dataframe(df.head(10), use_container_width=True)

# Scatter Plot Before Scaling

st.subheader("Original Dataset")

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(
    df["petal length (cm)"],
    df["petal width (cm)"],
    color="blue"
)

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")

st.pyplot(fig)

# Scaling

scaler = MinMaxScaler()

scaled = scaler.fit_transform(
    df[["petal length (cm)", "petal width (cm)"]]
)

scaled_df = pd.DataFrame(
    scaled,
    columns=["Petal Length", "Petal Width"]
)

st.subheader("Scaled Dataset")

st.dataframe(scaled_df.head(), use_container_width=True)

# Elbow Method

st.subheader("Elbow Method")

sse = []

K = range(1,11)

for k in K:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(scaled_df)
    sse.append(model.inertia_)

fig2, ax2 = plt.subplots(figsize=(7,5))

ax2.plot(K, sse, marker='o')

ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("SSE")
ax2.set_title("Elbow Method")

st.pyplot(fig2)

st.success("Optimal Value of K = 3")

# Select K

k = st.slider(
    "Choose Number of Clusters",
    2,
    10,
    3
)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

prediction = model.fit_predict(scaled_df)

scaled_df["Cluster"] = prediction

st.subheader("Clustered Data")

st.dataframe(scaled_df.head(20), use_container_width=True)

# Cluster Plot

fig3, ax3 = plt.subplots(figsize=(7,6))

colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "cyan",
    "olive"
]

for i in range(k):
    cluster = scaled_df[scaled_df["Cluster"] == i]

    ax3.scatter(
        cluster["Petal Length"],
        cluster["Petal Width"],
        color=colors[i],
        label=f"Cluster {i}"
    )

centers = model.cluster_centers_

ax3.scatter(
    centers[:,0],
    centers[:,1],
    marker="*",
    s=250,
    color="black",
    label="Centroids"
)

ax3.set_xlabel("Scaled Petal Length")
ax3.set_ylabel("Scaled Petal Width")
ax3.legend()

st.pyplot(fig3)

st.info("""
Project Summary

• Dataset: Iris Dataset from sklearn

• Features Used:
    - Petal Length
    - Petal Width

• Algorithm:
    - K-Means Clustering

• Preprocessing:
    - MinMax Scaling

• Optimal Number of Clusters:
    - K = 3
""")