import joblib
import streamlit as st
import pandas as pd

def show_prediction():
    # Load dataset
    df = pd.read_csv('retail_store_inventory.csv', sep='\t')

    # Load trained model
    lr = joblib.load("Price_Predict_Linear")

    st.title("Retail Price Prediction")

    with st.form("Prediction Form"):
        st.subheader("Please Input Your Data")
        
        # Selectbox Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            category_option = st.selectbox('Choose Category:', ('Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys'))
            category_map = {'Clothing': 0, 'Electronics': 1, 'Furniture': 2, 'Groceries': 3, 'Toys': 4}
            Category = category_map[category_option]

        with col2:
            region_option = st.selectbox('Choose Region:', ('East', 'North', 'South', 'West'))
            region_map = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
            Region = region_map[region_option]

        with col3:
            weather_option = st.selectbox('Choose Weather Condition:', ('Cloudy', 'Rainy', 'Snowy', 'Sunny'))
            weather_map = {'Cloudy': 0, 'Rainy': 1, 'Snowy': 2, 'Sunny': 3}
            weatherCondition = weather_map[weather_option]

        with col4:
            season_option = st.selectbox('Choose Seasonality:', ('Autumn', 'Spring', 'Summer', 'Winter'))
            season_map = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
            Seasonality = season_map[season_option]

        # Numeric Inputs
        col5, col6 = st.columns(2)
        with col5:
            sold = st.number_input('Unit Sold:', min_value=0, step=1, value=0)
        with col6:
            order = st.number_input('Unit Ordered:', min_value=0, step=1, value=0)

        col7, col8 = st.columns(2)
        with col7:
            inventory = st.number_input('Inventory Level:', min_value=0, step=1, value=0)
        with col8:
            discount_percent = st.slider('Discount (%):', min_value=0, max_value=100, value=0, step=1)
            discount = discount_percent / 100  # Convert to decimal for model

        comp = st.number_input('Competitor Price ($):', min_value=0.0, step=0.01, format="%.2f")

        # Checkbox for Holiday/Promotion
        is_holiday = st.checkbox('Holiday / Promotion')
        holiday = 1 if is_holiday else 0

        # Submit button
        submit = st.form_submit_button("Predict Optimal Price")

        if submit:
            # Prepare input features as DataFrame
            input_data = pd.DataFrame([[
                Category, Region, inventory, sold, order, discount, weatherCondition, is_holiday, comp, Seasonality
            ]], columns=['Category', 'Region', 'Inventory Level', 'Units Sold', 'Units Ordered',
                        'Discount', 'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing',
                        'Seasonality'
                    ])

            # Predict using model
            predicted_price = lr.predict(input_data)[0]

            # Show result
            st.success(f"✅ The predicted optimal price is: **${predicted_price:.2f}**")
