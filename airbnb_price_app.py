import streamlit as st
import pandas as pd
import joblib

# 加载模型和特征列
model = joblib.load("airbnb_price_model.pkl")
columns = joblib.load("model_columns.pkl")

def main():
    st.set_page_config(page_title="Airbnb Price Estimator")
    st.title("🏠 Airbnb Price Estimator")
    st.write("Enter your listing details:")

    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
    neighbourhood_group = st.selectbox("Neighbourhood Group", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    minimum_nights = st.slider("Minimum Nights", 1, 30, 3)
    number_of_reviews = st.slider("Number of Reviews", 0, 500, 10)
    reviews_per_month = st.slider("Reviews Per Month", 0.0, 10.0, 0.5)
    availability_365 = st.slider("Availability (Days)", 0, 365, 180)

    if st.button("Start"):
        input_dict = {
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'availability_365': availability_365,
            f'room_type_{room_type}': 1,
            f'neighbourhood_group_{neighbourhood_group}': 1
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        pred = model.predict(input_df)[0]
        st.success(f"Estimated price: ${pred:.2f}")
        st.write(f"Estimated range: ${pred*0.85:.0f} ~ ${pred*1.15:.0f}")

if __name__ == '__main__':
    main()