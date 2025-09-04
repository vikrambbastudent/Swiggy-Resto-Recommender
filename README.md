# 🍽️ Swiggy-Resto-Recommender  

This project is a **machine learning-based recommendation system** built on real-world Swiggy-like restaurant data.  
It predicts and recommends restaurants to users based on **city, cuisine, cost, and rating**, leveraging clustering techniques.  

The project includes both a **Jupyter Notebook (model training)** and a **Streamlit web application (deployment)**.  

---

## 🚀 Features
- 📊 Data cleaning & preprocessing of raw restaurant data
- 🍕 Recommendation system that suggests top restaurants
- 🏙️ City-wise restaurant filtering
- 📈 Model training & evaluation (Silhouette Score for clustering quality)
- 🖥️ Interactive UI built with Streamlit

---

## 🗂️ Dataset
- **File:** `cleaned_csv1.csv`  
- **Columns used:**
  - `name` → Restaurant name  
  - `city` → City where the restaurant is located  
  - `cuisine` → Primary cuisine served  
  - `cost` → Average cost for two (normalized)  
  - `rating` → Customer rating (scaled)  
  - `link` → Swiggy restaurant link  

---

## 🧠 Machine Learning Models
### 1. **KMeans Clustering**
- **Parameters used:**
  - `n_clusters = 5`  
  - `init = 'k-means++'`  
  - `n_init = 'auto'`  
  - `max_iter = 300`  
  - `tol = 0.0001`  
  - `random_state = 42`  
  - `algorithm = 'lloyd'`  

- **Silhouette Score:** `0.6963`  
  (indicating good separation between clusters)

- Model saved as `kmeans_model.pkl`.  

### 2. **Encoders & Scalers**
- **OneHotEncoder** for `city`  
- **LabelEncoder** for `cuisine`  
- **StandardScaler** for `rating` and `cost`  
- Saved as `encoder.pkl`.  

---

## 📊 Results
Example recommendation for:  
- **City:** Patna  
- **Cuisine:** Indian  
- **Cost:** 1000.00  
- **Rating:** 5.0  

**Recommended Restaurants:**
- Garage Kitchen  
- Hotpot  
- The Napples - Hotel Amalfi Grand  
- Dangus 
- Meal On Time
- Sundae Everyday Ice Creams
- Bollywood Treats - Hotel Maurya 
- Hot Grills
- Foodie 18 
- Hi Q Foods

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Pandas, NumPy** → Data preprocessing  
- **Scikit-learn** → ML models & evaluation  
- **Matplotlib / Seaborn / Plotly** → Visualization  
- **Streamlit** → Web app deployment  
- **Docker** (optional) → Containerized deployment  

---

## ⚙️ How to Run
### 1. Clone Repository
```bash
git clone https://github.com/yourusername/swiggy-recommendation-system.git
cd swiggy-recommendation-system
