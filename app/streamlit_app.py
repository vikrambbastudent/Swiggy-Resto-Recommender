# Streamlit-based Recommendation System (cover-only hero, no text overlay)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# ---------------------------
# Page config MUST be first
# ---------------------------
st.set_page_config(
    page_title="Swiggy Eats Recommender",
    page_icon="🍴",
    layout="wide"
)

# ---------------------------
# Global styles
# ---------------------------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%); }
      .block-container { padding-top: 0rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar Logo
# ---------------------------
logo_path = r"C:\Users\HP\SwiggyEats-Recommender\ChatGPT Image Sep 4, 2025, 03_02_30 PM.png"
st.sidebar.image(logo_path, width=150)
st.sidebar.markdown("## Filter Your Preferences")

# ---------------------------
# Utilities
# ---------------------------
def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------------------
# Hero (cover image at very top, NO text overlay)
# ---------------------------
cover_path = r"C:\Users\HP\SwiggyEats-Recommender\Your paragraph text.png"
cover_base64 = get_base64_image(cover_path)

st.markdown(
    f"""
    <div style="text-align: center;
                background-image: url('data:image/png;base64,{cover_base64}');
                background-size: cover;
                background-position: center;
                padding: 200px;   /* Increase padding to make banner taller */
                border-radius: 15px;
                margin-bottom: 12px;">
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Subtitle (below cover)
# ---------------------------
st.markdown("### Personalized Restaurant Recommendations based on City, Cuisine, Cost, and Rating")

# ---------------------------
# Load Data & Models
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\HP\SwiggyEats-Recommender\dataset\cleaned_csv1.csv")

@st.cache_resource
def load_models():
    with open(r"C:\Users\HP\SwiggyEats-Recommender\models\encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(r"C:\Users\HP\SwiggyEats-Recommender\models\kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    return encoders, kmeans

cleaned_df = load_data()
encoders, kmeans = load_models()

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🔍 Filter Your Preferences")
city = st.sidebar.selectbox('🏙️ Select City', options=cleaned_df['city'].unique())
cuisine = st.sidebar.selectbox('🍲 Select Cuisine', options=cleaned_df['cuisine'].unique())
cost = st.sidebar.number_input('💰 Enter Cost', min_value=0.0, step=0.1, value=500.0)
rating = st.sidebar.slider("⭐ Select a Rating", min_value=0.0, max_value=5.0, step=0.1, value=4.0)

# ---------------------------
# Recommendations
# ---------------------------
if st.sidebar.button('🚀 Get Recommendations'):
    # Encode inputs
    city_encoded = pd.DataFrame(
        encoders["city_encoder"].transform([[city]]),
        columns=encoders["city_encoder"].get_feature_names_out(["city"])
    )
    cuisine_encoded = pd.DataFrame(
        [encoders["cuisine_encoder"].transform([cuisine])],
        columns=["cuisine_encoded"]
    )
    rating_scaled = pd.DataFrame(
        encoders["rating_scaler"].transform([[rating]]),
        columns=["rating_scaled"]
    )
    cost_scaled = pd.DataFrame(
        encoders["cost_scaler"].transform([[cost]]),
        columns=["cost_scaled"]
    )

    # Combine inputs
    user_input_df = pd.concat([cuisine_encoded, rating_scaled, cost_scaled, city_encoded], axis=1)

    # Predict cluster
    predicted_cluster = kmeans.predict(user_input_df)[0]

    # Filter recommendations
    cluster_indices = (kmeans.labels_ == predicted_cluster)
    result = cleaned_df.loc[cluster_indices & (cleaned_df["city"] == city)]

    if not result.empty:
        st.subheader(
            f"🎯 Recommended Restaurants in {city} "
            f"(Cuisine: {cuisine}, Rating ≥ {rating}, Cost ~ ₹{cost:.0f})"
        )

        recommended_restaurants = result[["name", "address", "rating", "cost", "link"]].rename(
            columns={
                "name": "Restaurant_Name",
                "address": "Address",
                "rating": "Rating",
                "cost": "Cost",
                "link": "Restaurant_Link"
            }
        )

        # Compact address
        recommended_restaurants["Address"] = recommended_restaurants["Address"].apply(
            lambda x: ", ".join(str(x).split(",")[-3:]).strip() if pd.notnull(x) else "N/A"
        )

        # Sort
        recommended_restaurants["Cost_Diff"] = (recommended_restaurants["Cost"] - cost).abs()
        recommended_restaurants = recommended_restaurants.sort_values(
            by=["Rating", "Cost_Diff"], ascending=[False, True]
        )

        # Indexing
        recommended_restaurants.reset_index(drop=True, inplace=True)
        recommended_restaurants.index = recommended_restaurants.index + 1
        recommended_restaurants.index.name = "S.No"

        # Emojis
        medals = {1: "🏆 ", 2: "🥈 ", 3: "🥉 "}
        recommended_restaurants["Restaurant_Name"] = [
            f"{medals.get(i, '')}{name}" for i, name in enumerate(recommended_restaurants["Restaurant_Name"], start=1)
        ]

        # Hyperlinks
        recommended_restaurants["Restaurant_Link"] = recommended_restaurants["Restaurant_Link"].apply(
            lambda x: f"[Visit]({x})" if pd.notnull(x) else "N/A"
        )

        recommended_restaurants = recommended_restaurants.drop(columns=["Cost_Diff"])

        st.markdown("### 🍽️ Your Recommendations")
        st.write(
            recommended_restaurants[["Restaurant_Name", "Address", "Rating", "Restaurant_Link"]]
            .head(10)
            .to_markdown(index=True)
        )
    else:
        st.warning("⚠️ Sorry, no restaurants match your preferences. Try adjusting filters.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("📌 *Powered by Machine Learning & Streamlit* | Created with ❤️ for Food Lovers")
