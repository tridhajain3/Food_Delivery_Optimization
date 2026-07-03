import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Food Delivery Operations Analytics",
    page_icon="🚚",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data/food_delivery_cleaned.csv")

df = load_data()

# =====================================
# TITLE
# =====================================

st.title("🚚 Food Delivery Operations Analytics & Prediction System")

st.markdown("""
Interactive dashboard for:

- 📊 Delivery Performance Analytics
- 🌍 Geographic Delivery Insights
- 📈 Operations Analysis
- 🤖 Delivery Time Prediction
""")

st.divider()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Filters")

vehicle = st.sidebar.multiselect(
    "Vehicle Type",
    sorted(df["Type_of_vehicle"].unique()),
    default=sorted(df["Type_of_vehicle"].unique())
)

order = st.sidebar.multiselect(
    "Order Type",
    sorted(df["Type_of_order"].unique()),
    default=sorted(df["Type_of_order"].unique())
)

rating = st.sidebar.slider(
    "Minimum Rider Rating",
    float(df["Delivery_person_Ratings"].min()),
    float(df["Delivery_person_Ratings"].max()),
    3.0
)

filtered = df[
    (df["Type_of_vehicle"].isin(vehicle)) &
    (df["Type_of_order"].isin(order)) &
    (df["Delivery_person_Ratings"] >= rating)
]

# =====================================
# KPI
# =====================================

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

st.divider()

# =====================================
# TABS
# =====================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Operations Analytics",
    "Geographic Intelligence",
    "Prediction Engine"
])

# =====================================
# EXECUTIVE OVERVIEW
# =====================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered,
            x="Time_taken(min)",
            nbins=30,
            title="Delivery Time Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.pie(
            filtered,
            names="Type_of_vehicle",
            title="Vehicle Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        filtered,
        x="Type_of_order",
        y="Time_taken(min)",
        color="Type_of_order",
        title="Delivery Time by Order Type"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# OPERATIONS ANALYTICS
# =====================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered.sample(min(5000, len(filtered))),
            x="Distance_km",
            y="Time_taken(min)",
            color="Type_of_vehicle",
            size="Delivery_person_Ratings",
            hover_data=["Type_of_order"],
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

    fig = px.bar(
        filtered.groupby("Type_of_vehicle")["Time_taken(min)"]
        .mean()
        .reset_index(),
        x="Type_of_vehicle",
        y="Time_taken(min)",
        title="Average Delivery Time by Vehicle"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# GEOGRAPHIC INTELLIGENCE
# =====================================

with tab3:

    sample = filtered.sample(min(3000, len(filtered)))

    fig = px.scatter_map(
        sample,
        lat="Delivery_location_latitude",
        lon="Delivery_location_longitude",
        color="Time_taken(min)",
        size="Distance_km",
        zoom=4,
        title="Delivery Density Map",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter_map(
        sample,
        lat="Restaurant_latitude",
        lon="Restaurant_longitude",
        color="Delivery_person_Ratings",
        size="Time_taken(min)",
        zoom=4,
        title="Restaurant Hotspots",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        sample,
        x="Distance_km",
        y="Time_taken(min)",
        color="Type_of_vehicle",
        size="Delivery_person_Ratings",
        hover_data=["Type_of_order"],
        title="Delivery Flow Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# PREDICTION ENGINE
# =====================================

with tab4:

    st.subheader("Predict Delivery Time")

    age = st.slider(
        "Delivery Person Age",
        18,
        50,
        30
    )

    rating = st.slider(
        "Delivery Person Rating",
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

            order_enc = list(
                df["Type_of_order"].unique()
            ).index(order_type)

            vehicle_enc = list(
                df["Type_of_vehicle"].unique()
            ).index(vehicle_type)

            prediction = model.predict(
                [[
                    age,
                    rating,
                    distance,
                    order_enc,
                    vehicle_enc
                ]]
            )[0]

            st.success(
                f"Estimated Delivery Time: {prediction:.1f} minutes"
            )

        except:

            estimate = (
                distance * 2 +
                (5 - rating) * 4 +
                8
            )

            st.info(
                f"Estimated Delivery Time: {estimate:.1f} minutes"
            )

st.divider()

st.caption(
    "Food Delivery Operations Analytics & Prediction System | Python • Streamlit • Plotly • Machine Learning"
)
