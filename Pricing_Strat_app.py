import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import streamlit_folium as st_folium

# Set page configuration (this must be the first Streamlit command)
st.set_page_config(page_title="Dynamic Pricing App", layout="wide")

# Load Data
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

# Load the uploaded file
file_path = 'TM_Data.xlsx'
data = load_data(file_path)

# Multi-Page Application
# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Introduction", "Dynamic Pricing Concepts", "Data Overview", "Pricing Analysis", "Predictive Modeling","Suggestions"]
selected_page = st.sidebar.radio("Go to:", pages)

relevant_columns = ['Event Type', 'Genre', 'State', 'City', 'Min Price $', 'Max Price $']

# Page 0: Introduction
if selected_page == "Introduction":
    st.title(" This is my attempt to build an Application to understand Dynamic Pricing Startegies at TicketMaster ")
    
    st.write("""
    This application is designed to showcase how **dynamic pricing strategies** can be effectively implemented and analyzed in the context of live events, such as concerts, sports, and theater. 
    It combines **data-driven insights** with **predictive modeling** to recommend optimal pricing strategies for maximum revenue and sell-through rates.

    ### Key Features of This Application:
    1. **Dynamic Pricing Concepts**:
        - Provides an in-depth explanation of key pricing strategies, such as demand-based pricing, time-based pricing, and surge pricing
    2. **Data Overview**:
        - Offers a detailed analysis of the dataset, including missing value handling, statistical summaries, and trends.
    3. **Pricing Analysis**:
        - Highlights actionable insights, including high-value event types, pricing volatility, genre-based opportunities, and geographical pricing dynamics.
    4. **Predictive Modeling**:
        - Predicts the **optimal ticket price** for an event using **machine learning models** and user-provided inputs. 
        - Incorporates dynamic pricing factors, such as event type, genre, location, and historical price ranges.

    ### How did I built it:
    - Pulled Data using TicketMaster API and cleaned it
    - Data Discovery by using Exploratory Data Analyis to understand the spread of Data
    - A deeper dive into the Pricing Analysis to understand High Value locations, Seasonal Pricing Trends, Volatality, Elasticity and features related to Pricing for the spread of Data
    - A Machine learning Model to predict/forecast accurate optimal pricing given other features for various upcoming event's  

    Explore the app using the navigation menu on the left! 🎉
    """)

# Page 1: Dynamic Pricing Concepts
if selected_page == "Dynamic Pricing Concepts":
    st.title("My Understanding of Dynamic Pricing Strategies")
    
    st.write("""
    **Dynamic pricing** is the strategy of adjusting prices in real time based on market conditions, such as demand, supply, competitor pricing, and customer behavior. Unlike fixed pricing, dynamic pricing is flexible, allowing businesses to maximize revenue by aligning prices with current market dynamics.

    ### Key Types of Dynamic Pricing Strategies
    1. **Time-Based Pricing** 🕒:
        - Prices vary depending on the time of day, day of the week, or season.
        - Example: Hotels and concert venues charge more on peak weekends than during off-peak weekdays.

    2. **Demand-Based Pricing** 📈:
        - Prices are adjusted based on supply and demand.
        - Example: Airlines increase ticket prices as the departure date approaches and fewer seats remain.

    3. **Surge Pricing** ⚡:
        - Prices spike during periods of unusually high demand.
        - Example: Uber charges higher fares during rainy weather or major events to incentivize more drivers to work.

    4. **Personalized Pricing** 🛒:
        - Prices are tailored to individual customers based on their purchase history, loyalty, or browsing habits.
        - Example: E-commerce platforms offer personalized discounts to repeat customers.

    5. **Segmented Pricing** 🎟️:
        - Prices are customized for broader customer groups.
        - Example: Students and senior citizens receive discounted ticket prices for events.

    ### Application in the Live Events Industry 🎤🎶
    Dynamic pricing in live events, such as concerts and sports games, involves several unique factors:
    - **Historical Sales Data** 📊: Analyzing past demand patterns for similar events or performers.
    - **Seating Segmentation** 🎭: Adjusting prices based on seat proximity to the stage and other value-perception factors.
    - **Competitor Pricing** 💲: Comparing prices on secondary markets.
    - **Seasonality** 🌟: Higher prices during holidays or weekends compared to off-peak periods.
    - **Inventory Levels** 🎫: Adjusting prices dynamically as tickets sell out.

    ### Challenges in Dynamic Pricing 🚧  
             
    1. **Ticket Bots** 🤖:
        - Automated bots bulk-purchase tickets and resell them at inflated prices on secondary markets.
        - Although laws like the Better Online Ticket Sales (BOTS) Act exist, enforcement remains a challenge.

    2. **High Surcharges** 💸:
        - Additional fees (e.g., service or delivery charges) can increase ticket prices by as much as 75%.
    
    ### Why It Matters 🌍
    Dynamic pricing is about balancing analytics, market trends, and consumer psychology to optimize revenue while meeting customer expectations. In live events, it ensures ticket prices reflect real-time demand, competition, and other factors, enabling businesses to maximize profitability.
    """)

elif selected_page == "Data Overview":
    st.title("Data Overview 🧮")
    st.write("This section provides an in-depth analysis of the dataset, with a focus on pricing, event-specific trends, and location insights.")

    # Display the first few rows of the dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Basic statistics
    st.write("### Summary Statistics 📊")
    st.write(data.describe(include="all"))

    # Check for missing values
    st.write("### Missing Values Overview")
    missing_values = data.isnull().sum()
    missing_data = missing_values[missing_values > 0]
    if not missing_data.empty:
        st.write("Columns with missing values:")
        st.dataframe(missing_data)
    else:
        st.write("No missing values found in the dataset.")

    # Pricing analysis
    st.write("### Pricing Analysis 💲")
    if 'Min Price $' in data.columns and 'Max Price $' in data.columns and 'Average Price $' in data.columns:
        # Price distribution (average price)
        st.write("#### Average Price Distribution")
        fig_avg_price_dist, ax_avg_price_dist = plt.subplots()
        sns.histplot(data['Average Price $'], kde=False, ax=ax_avg_price_dist, bins=30, color="blue")
        ax_avg_price_dist.set_title("Average Price Distribution")
        st.pyplot(fig_avg_price_dist)

        # Price ranges (Min and Max)
        st.write("#### Price Range (Min vs Max)")
        fig_price_range, ax_price_range = plt.subplots()
        sns.scatterplot(data=data, x='Min Price $', y='Max Price $', hue='Event Type', ax=ax_price_range)
        ax_price_range.set_title("Min vs Max Price by Event Type")
        st.pyplot(fig_price_range)
    else:
        st.warning("Pricing columns ('Min Price $', 'Max Price $', 'Average Price $') are missing or incomplete.")

    # Event-specific insights
    st.write("### Event-Specific Insights 🎭")
    if 'Event Type' in data.columns and 'Genre' in data.columns:
        # Count of events by type
        st.write("#### Number of Events by Type")
        event_count = data['Event Type'].value_counts()
        st.bar_chart(event_count)

        # Average price by genre
        if 'Average Price $' in data.columns:
            genre_price = data.groupby('Genre')['Average Price $'].mean().sort_values(ascending=False).reset_index()
            st.write("#### Average Price by Genre")
            st.dataframe(genre_price)
            fig_genre_price, ax_genre_price = plt.subplots()
            sns.barplot(data=genre_price, x='Average Price $', y='Genre', palette="viridis", ax=ax_genre_price)
            ax_genre_price.set_title("Average Price by Genre")
            st.pyplot(fig_genre_price)
    else:
        st.warning("'Event Type' or 'Genre' columns are missing.")

    # Location-based analysis
    st.write("### Location-Based Analysis 🌍")
    if 'State' in data.columns and 'City' in data.columns and 'Average Price $' in data.columns:
        # Average price by state
        state_price = data.groupby('State')['Average Price $'].mean().sort_values(ascending=False).reset_index()
        st.write("#### Average Price by State")
        st.dataframe(state_price)
        fig_state_price, ax_state_price = plt.subplots(figsize=(10, 6))
        sns.barplot(data=state_price, x='Average Price $', y='State', palette="coolwarm", ax=ax_state_price)
        ax_state_price.set_title("Average Price by State")
        st.pyplot(fig_state_price)

        # Event distribution by city
        city_events = data['City'].value_counts().head(10)
        st.write("#### Top 10 Cities by Number of Events")
        st.bar_chart(city_events)
    else:
        st.warning("'State', 'City', or 'Average Price $' columns are missing.")

    # Time-based analysis
    st.write("### Time-Based Analysis 🕒")
    if 'Dates' in data.columns and 'Average Price $' in data.columns:
        data['Dates'] = pd.to_datetime(data['Dates'], errors='coerce')
        time_price = data.groupby(data['Dates'].dt.to_period('M'))['Average Price $'].mean().reset_index()
        time_price['Dates'] = time_price['Dates'].dt.to_timestamp()
        fig_time_price, ax_time_price = plt.subplots()
        ax_time_price.plot(time_price['Dates'], time_price['Average Price $'], marker='o')
        ax_time_price.set_title("Average Price Over Time")
        ax_time_price.set_xlabel("Date")
        ax_time_price.set_ylabel("Average Price")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_time_price)
    else:
        st.warning("Time-based analysis requires both 'Dates' and 'Average Price $' columns.")


elif selected_page == "Pricing Analysis":
    st.title("Top 10 Pricing Insights 🔍")
    st.write("This section highlights actionable insights derived from the dataset to optimize pricing strategies and revenue potential.")

    # 1. High-Performing Event Types
    st.write("### 1. High-Performing Event Types 🎭")
    if 'Event Type' in data.columns and 'Average Price $' in data.columns:
        avg_price_event = data.groupby('Event Type')['Average Price $'].mean().sort_values(ascending=False).reset_index()
        st.write("#### Average Price by Event Type")
        st.dataframe(avg_price_event)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=avg_price_event, x='Average Price $', y='Event Type', palette="coolwarm", ax=ax1)
        ax1.set_title("Average Price by Event Type")
        ax1.set_xlabel("Average Price ($)")
        st.pyplot(fig1)

        st.write("**Insight:** Focus on top-performing event types to maximize revenue.")
    else:
        st.warning("Columns 'Event Type' or 'Average Price $' are missing.")

    # 2. High-Value Locations with Folium
    st.write("### 2. High-Value Locations 🌍")
    if {'City', 'Latitude', 'Longitude', 'Average Price $'}.issubset(data.columns):
        st.write("#### Map of High-Value Cities")
        city_avg_prices = data.groupby(['City', 'Latitude', 'Longitude'])['Average Price $'].mean().reset_index()

        # Create Folium map
        map_center = [city_avg_prices['Latitude'].mean(), city_avg_prices['Longitude'].mean()]
        pricing_map = folium.Map(location=map_center, zoom_start=5)
        for _, row in city_avg_prices.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=7,
                popup=(
                    f"City: {row['City']}<br>"
                    f"Average Price: ${row['Average Price $']:.2f}"
                ),
                color="blue",
                fill=True,
                fill_opacity=0.7,
            ).add_to(pricing_map)

        # Display map
        from streamlit_folium import st_folium
        st_folium(pricing_map, width=700, height=500)

        st.write("**Insight:** Cities with higher average prices represent opportunities for premium ticket pricing.")
    else:
        st.warning("Columns 'City', 'Latitude', 'Longitude', or 'Average Price $' are missing.")

    # 3. Seasonal Pricing Trends
    st.write("### 3. Seasonal Pricing Trends 📅")
    if 'Dates' in data.columns and 'Average Price $' in data.columns:
        data['Dates'] = pd.to_datetime(data['Dates'], errors='coerce')
        data['Month'] = data['Dates'].dt.month
        monthly_avg_prices = data.groupby('Month')['Average Price $'].mean().reset_index()

        st.write("#### Monthly Average Prices")
        st.dataframe(monthly_avg_prices)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=monthly_avg_prices, x='Month', y='Average Price $', marker='o', ax=ax3)
        ax3.set_title("Seasonal Trends in Average Prices")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Average Price ($)")
        st.pyplot(fig3)

        st.write("**Insight:** Use time-based pricing strategies to increase prices during peak months.")
    else:
        st.warning("Columns 'Dates' or 'Average Price $' are missing.")

    # 4. Pricing Volatility by Event Type
    st.write("### 4. Pricing Volatility by Event Type 📊")
    if 'Event Type' in data.columns and 'Average Price $' in data.columns:
        volatility_event = data.groupby('Event Type')['Average Price $'].std().sort_values(ascending=False).reset_index()
        st.write("#### Price Volatility by Event Type")
        st.dataframe(volatility_event)

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=volatility_event, x='Average Price $', y='Event Type', palette="viridis", ax=ax4)
        ax4.set_title("Price Volatility by Event Type")
        ax4.set_xlabel("Standard Deviation in Price ($)")
        st.pyplot(fig4)

        st.write("**Insight:** High-volatility event types benefit from dynamic pricing adjustments to stabilize revenue.")
    else:
        st.warning("Columns 'Event Type' or 'Average Price $' are missing.")

    # 5. Genre-Driven Revenue Opportunities
    st.write("### 5. Genre-Driven Revenue Opportunities 🎶")
    if 'Genre' in data.columns and 'Average Price $' in data.columns:
        avg_price_genre = data.groupby('Genre')['Average Price $'].mean().sort_values(ascending=False).reset_index()
        st.write("#### Average Price by Genre")
        st.dataframe(avg_price_genre)

        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=avg_price_genre, x='Average Price $', y='Genre', palette="magma", ax=ax5)
        ax5.set_title("Average Price by Genre")
        ax5.set_xlabel("Average Price ($)")
        st.pyplot(fig5)

        st.write("**Insight:** Focus on high-performing genres for targeted marketing and promotions.")
    else:
        st.warning("Columns 'Genre' or 'Average Price $' are missing.")

    # 6. Top-Performing Events
    st.write("### 6. Top-Performing Events 🌟")
    if 'Event Name' in data.columns and 'Average Price $' in data.columns:
        top_events = data[['Event Name', 'Average Price $']].sort_values(by='Average Price $', ascending=False).head(10)
        st.write("#### Top 10 Events by Average Price")
        st.dataframe(top_events)

        st.write("**Insight:** Highlight these events to strategize promotional efforts.")
    else:
        st.warning("Columns 'Event Name' or 'Average Price $' are missing.")

    # 7. Price Sensitivity and Elasticity
    st.write("### 7. Price Sensitivity and Elasticity 📈")
    if 'Min Price $' in data.columns and 'Average Price $' in data.columns:
        data['Price Elasticity'] = data['Average Price $'] / data['Min Price $']
        st.write("#### Price Elasticity Distribution")
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Price Elasticity'], kde=True, ax=ax7, color="purple")
        ax7.set_title("Price Elasticity Distribution")
        st.pyplot(fig7)

        st.write("**Insight:** Events with low elasticity are ideal candidates for premium pricing.")
    else:
        st.warning("Columns 'Min Price $' or 'Average Price $' are missing.")

    # 8. Event-Genre Combinations
    st.write("### 8. Event-Genre Combinations 🎟️")
    if 'Event Type' in data.columns and 'Genre' in data.columns and 'Average Price $' in data.columns:
        top_combinations = data.groupby(['Event Type', 'Genre'])['Average Price $'].mean().sort_values(ascending=False).reset_index().head(10)
        st.write("#### Top 10 Event-Genre Combinations by Price")
        st.dataframe(top_combinations)

        st.write("**Insight:** These combinations have the highest pricing potential.")
    else:
        st.warning("Columns 'Event Type', 'Genre', or 'Average Price $' are missing.")

    # 9. Pricing and Location Dynamics
    st.write("### 9. Pricing and Location Dynamics 🌎")
    if 'State' in data.columns and 'Average Price $' in data.columns:
        avg_price_state = data.groupby('State')['Average Price $'].mean().sort_values(ascending=False).reset_index()
        st.write("#### Average Price by State")
        st.dataframe(avg_price_state)

        st.write("**Insight:** States with higher average prices represent lucrative markets for premium pricing.")
    else:
        st.warning("Columns 'State' or 'Average Price $' are missing.")

    # 10. High-Value Price Ranges
    st.write("### 10. High-Value Price Ranges 💵")
    if 'Min Price $' in data.columns and 'Max Price $' in data.columns and 'Average Price $' in data.columns:
        st.write("#### Price Range Analysis")
        fig10, ax10 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data[['Min Price $', 'Max Price $', 'Average Price $']], ax=ax10)
        ax10.set_title("Distribution of Min, Max, and Average Prices")
        st.pyplot(fig10)

        st.write("**Insight:** Identify events with excessively broad price ranges to optimize pricing strategies.")
    else:
        st.warning("Columns 'Min Price $', 'Max Price $', or 'Average Price $' are missing.")


# Page 5: Predictive Modeling
elif selected_page == "Predictive Modeling":
    st.title("Dynamic Pricing Model 💡")
    st.write("Predict the optimal ticket price based on event details, incorporating dynamic pricing strategies to maximize revenue.")

    # Check if necessary columns exist in the dataset
    if {'Event Type', 'Genre', 'State', 'City', 'Min Price $', 'Max Price $'}.issubset(data.columns):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        import numpy as np

        # Step 1: Dynamic filtering for user input
        # Select Event Type
        event_type = st.selectbox("Event Type", data['Event Type'].dropna().unique())

        # Filter genres based on the selected Event Type
        filtered_genres = data[data['Event Type'] == event_type]['Genre'].dropna().unique()
        genre = st.selectbox("Genre", filtered_genres)

        # Filter states based on the selected Event Type
        filtered_states = data[data['Event Type'] == event_type]['State'].dropna().unique()
        state = st.selectbox("State", filtered_states)

        # Filter cities based on the selected State
        filtered_cities = data[(data['Event Type'] == event_type) & (data['State'] == state)]['City'].dropna().unique()
        city = st.selectbox("City", filtered_cities)

        # Input for Minimum and Maximum Price
        min_price = st.number_input("Minimum Price ($)", min_value=0.0, step=1.0)
        max_price = st.number_input("Maximum Price ($)", min_value=min_price, step=1.0)

        # Step 2: Modeling
        relevant_columns = ['Event Type', 'Genre', 'State', 'City', 'Min Price $', 'Max Price $']
        target_column = 'Average Price $'
        filtered_data = data.dropna(subset=relevant_columns + [target_column])

        # Prepare features and target for modeling
        X = filtered_data[relevant_columns]
        y = filtered_data[target_column]

        # One-hot encode categorical features
        categorical_features = ['Event Type', 'Genre', 'State', 'City']
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        # Train a Gradient Boosting Regressor
        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, random_state=528)

        # Preprocess and fit the model
        X_encoded = preprocessor.fit_transform(X)
        model.fit(X_encoded, y)

        # Create a custom input DataFrame for prediction
        custom_input = pd.DataFrame({
            'Event Type': [event_type],
            'Genre': [genre],
            'State': [state],
            'City': [city],
            'Min Price $': [min_price],
            'Max Price $': [max_price]
        })

        # Encode the custom input
        custom_input_encoded = preprocessor.transform(custom_input)

        # Predict the optimal price
        if st.button("Predict Optimal Price"):
            optimal_price = model.predict(custom_input_encoded)[0]
            st.write(f"### Recommended Optimal Price: ${optimal_price:.2f}")

            # Explain dynamic pricing context
            st.write("""
            #### How This Price is Determined:
            - **Demand-Based Pricing**: Factors like event type, genre, and location are used to adjust pricing based on historical demand trends.
            - **Location-Specific Pricing**: States and cities with historically higher pricing trends influence the recommended price.
            - **Price Range Influence**: The input minimum and maximum prices serve as bounds for optimal price prediction.
            """)
    else:
        st.warning("The dataset does not contain all the required columns: 'Event Type', 'Genre', 'State', 'City', 'Min Price $', and 'Max Price $'.")

# Page 6: Suggestions and Feedback
elif selected_page == "Suggestions":
    st.title("Suggestions and Feedback 💡")
    st.write("I value your feedback! Please share your thoughts on how I can improve this application.")

    # User input for suggestions
    suggestions = st.text_area("Your Suggestions:", placeholder="Write your suggestions here...", height=150)

    # Submit button
    if st.button("Submit Feedback"):
        if suggestions.strip():  # Check if the user entered any text
            st.success("Thank you for your feedback! Your suggestions have been recorded.")
            # Save the feedback to a file or database
            with open("user_feedback.txt", "a") as feedback_file:
                feedback_file.write(suggestions + "\n")
        else:
            st.warning("Please enter a suggestion before submitting.")

    # Display previous feedback (Optional)
    st.write("---")
    st.write("### Previously Submitted Feedback")
    try:
        with open("user_feedback.txt", "r") as feedback_file:
            feedback_data = feedback_file.readlines()
            if feedback_data:
                for i, feedback in enumerate(feedback_data[-5:][::-1], 1):  # Show the last 5 suggestions
                    st.write(f"**{i}.** {feedback.strip()}")
            else:
                st.info("No feedback has been submitted yet.")
    except FileNotFoundError:
        st.info("No feedback has been submitted yet.")

# Footer
st.sidebar.write("---")
st.sidebar.write("Developed by Atharva Deshpande")
