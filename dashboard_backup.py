
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import joblib

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Food Delivery Operations Analytics",
    layout="wide",
    page_icon="🚚"
)

# ======================
# THEME
# ======================

st.markdown("""
<style>

.stApp{
    background-color:#FDF6EC;
}

[data-testid="stSidebar"]{
    background-color:#FFF8F0;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.1);
}

h1,h2,h3{
    color:#4B3F35;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA
# ======================

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data/food_delivery_cleaned.csv")

df = load_data()

# ======================
# TITLE
# ======================

st.title("🚚 Food Delivery Operations Analytics & Prediction System")
st.markdown("---")

# ======================
# SIDEBAR FILTERS
# ======================

st.sidebar.header("Filters")

vehicle = st.sidebar.multiselect(
    "Vehicle Type",
    df["Type_of_vehicle"].unique(),
    default=df["Type_of_vehicle"].unique()
)

order = st.sidebar.multiselect(
    "Order Type",
    df["Type_of_order"].unique(),
    default=df["Type_of_order"].unique()
)

rating = st.sidebar.slider(
    "Minimum Rating",
    float(df["Delivery_person_Ratings"].min()),
    float(df["Delivery_person_Ratings"].max()),
    3.0
)

filtered = df[
    (df["Type_of_vehicle"].isin(vehicle)) &
    (df["Type_of_order"].isin(order)) &
    (df["Delivery_person_Ratings"] >= rating)
]

# ======================
# KPI CARDS
# ======================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Deliveries",
    f"{len(filtered):,}"
)

c2.metric(
    "Average Time",
    f"{filtered['Time_taken(min)'].mean():.1f} min"
)

c3.metric(
    "Average Distance",
    f"{filtered['Distance_km'].mean():.2f} km"
)

c4.metric(
    "Average Rating",
    f"{filtered['Delivery_person_Ratings'].mean():.2f}"
)

st.markdown("---")

# ======================
# TABS
# ======================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Analytics",
    "Geographic Intelligence",
    "Network Analysis",
    "Prediction"
])

# =====================================================
# OVERVIEW
# =====================================================

with tab1:

    st.subheader("Delivery Performance Overview")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.violin(
            filtered,
            y="Time_taken(min)",
            box=True,
            points="outliers",
            title="Delivery Time Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        vehicle_count = (
            filtered["Type_of_vehicle"]
            .value_counts()
            .reset_index()
        )

        fig = px.pie(
            vehicle_count,
            names="Type_of_vehicle",
            values="count",
            title="Vehicle Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ANALYTICS
# =====================================================

with tab2:

    st.subheader("Delivery Analytics")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered.sample(min(5000, len(filtered))),
            x="Distance_km",
            y="Time_taken(min)",
            color="Type_of_vehicle",
            size="Delivery_person_Ratings",
            title="Distance vs Delivery Time"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        corr = filtered[
            [
                "Delivery_person_Age",
                "Delivery_person_Ratings",
                "Distance_km",
                "Time_taken(min)"
            ]
        ].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# GEOGRAPHIC
# =====================================================

with tab3:

    st.subheader("Geographic Intelligence")

    sample = filtered.sample(min(3000, len(filtered)))

    st.markdown("### Delivery Heatmap")

    fig = px.density_map(
        sample,
        lat="Delivery_location_latitude",
        lon="Delivery_location_longitude",
        radius=20,
        zoom=4,
        map_style="carto-positron"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Restaurant Hotspots")

    fig = px.scatter_map(
        sample,
        lat="Restaurant_latitude",
        lon="Restaurant_longitude",
        color="Distance_km",
        size="Time_taken(min)",
        zoom=4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Delivery Route Visualization")

    route = sample.sample(min(25, len(sample)))

    fig = go.Figure()

    for _, row in route.iterrows():

        fig.add_trace(
            go.Scattermap(
                lat=[
                    row["Restaurant_latitude"],
                    row["Delivery_location_latitude"]
                ],
                lon=[
                    row["Restaurant_longitude"],
                    row["Delivery_location_longitude"]
                ],
                mode="lines+markers",
                showlegend=False
            )
        )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            zoom=4
        ),
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# NETWORK
# =====================================================

with tab4:

    st.subheader("Delivery Network Analysis")

    sample = filtered.sample(min(200, len(filtered)))

    G = nx.Graph()

    for _, row in sample.iterrows():

        source = (
            round(row["Restaurant_latitude"], 2),
            round(row["Restaurant_longitude"], 2)
        )

        target = (
            round(row["Delivery_location_latitude"], 2),
            round(row["Delivery_location_longitude"], 2)
        )

        G.add_edge(source, target)

    c1, c2, c3 = st.columns(3)

    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())

    if G.number_of_nodes() > 0:
        c3.metric(
            "Density",
            round(nx.density(G), 4)
        )

    pos = nx.spring_layout(G)

    nodes = np.array([pos[k] for k in G.nodes()])

    fig = px.scatter(
        x=nodes[:, 0],
        y=nodes[:, 1],
        title="Delivery Network Graph"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================

with tab5:

    st.subheader("Delivery Time Prediction")

    age = st.slider(
        "Delivery Person Age",
        18,
        50,
        30
    )

    rating = st.slider(
        "Rating",
        1.0,
        5.0,
        4.0
    )

    distance = st.slider(
        "Distance (km)",
        1,
        30,
        10
    )

    order_type = st.selectbox(
        "Order Type",
        df["Type_of_order"].unique()
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        df["Type_of_vehicle"].unique()
    )

    if st.button("Predict Delivery Time"):

        try:

            model = joblib.load(
                "models/rf_model.pkl"
            )

            order_encoded = (
                list(df["Type_of_order"].unique())
                .index(order_type)
            )

            vehicle_encoded = (
                list(df["Type_of_vehicle"].unique())
                .index(vehicle_type)
            )

            X = [[
                age,
                rating,
                distance,
                order_encoded,
                vehicle_encoded
            ]]

            pred = model.predict(X)[0]

            st.success(
                f"Estimated Delivery Time: {pred:.1f} minutes"
            )

        except:

            estimate = (
                distance * 2.2 +
                (5 - rating) * 4 +
                8
            )

            st.info(
                f"Estimated Delivery Time: {estimate:.1f} minutes"
            )

st.markdown("---")
st.caption(
    "Food Delivery Operations Analytics & Prediction System | Python • Streamlit • Plotly • NetworkX • Machine Learning"
)
