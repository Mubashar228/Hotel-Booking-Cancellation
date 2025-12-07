import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Spark Session
# -------------------------------
spark = SparkSession.builder \
    .appName("Hotel Booking Cancellation") \
    .getOrCreate()

# -------------------------------
# Load Data
# -------------------------------
data_path = "hotel_booking.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# -------------------------------
# Selected columns
# -------------------------------
selected_col = ['lead_time', 'hotel', 'previous_cancellations',
                'previous_bookings_not_canceled', 'booking_changes',
                'deposit_type', 'customer_type', 'market_segment',
                'required_car_parking_spaces', 'total_of_special_requests',
                'is_canceled']

df1 = df.select(selected_col).dropna()

# -------------------------------
# Indexing categorical features
# -------------------------------
indexers = [
    StringIndexer(inputCol=col, outputCol=col+"_indexed", handleInvalid="keep")
    for col in ['hotel', 'deposit_type', 'customer_type', 'market_segment']
]

# -------------------------------
# Feature columns
# -------------------------------
feature_cols = ['lead_time', 'previous_cancellations', 'previous_bookings_not_canceled',
                'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests',
                'hotel_indexed', 'deposit_type_indexed', 'customer_type_indexed', 'market_segment_indexed']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="is_canceled")
pipeline = Pipeline(stages=indexers + [assembler, lr])

# -------------------------------
# Train Model
# -------------------------------
model = pipeline.fit(df1)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="wide")
st.title("🏨 Hotel Booking Cancellation Prediction (PySpark + Streamlit)")

st.markdown("""
This application predicts **hotel booking cancellations** using Machine Learning and PySpark.  
Use the input fields below to test booking scenarios and see the likelihood of cancellation.
""")

# -------------------------------
# Sidebar: Data Analysis
# -------------------------------
st.sidebar.header("📊 Data Analysis")
show_analysis = st.sidebar.checkbox("Show Booking Data Analysis")

if show_analysis:
    st.subheader("Booking Data Overview")
    st.dataframe(df1.limit(10).toPandas())  # Show first 10 rows

    st.subheader("Cancellation Distribution")
    cancel_counts = df1.groupBy("is_canceled").count().toPandas()
    st.bar_chart(cancel_counts.set_index('is_canceled'))

    st.subheader("Hotel Type vs Cancellation Rate")
    hotel_cancel = df1.groupBy("hotel", "is_canceled").count().toPandas()
    fig, ax = plt.subplots()
    sns.barplot(x="hotel", y="count", hue="is_canceled", data=hotel_cancel, ax=ax)
    st.pyplot(fig)

# -------------------------------
# Input Fields for Prediction
# -------------------------------
st.markdown("### 📋 Enter Booking Details")

lead_time = st.slider("Lead Time (days)", 0, 700, 100)
previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", 0, 100, 0)
booking_changes = st.slider("Booking Changes", 0, 20, 0)
deposit_type = st.selectbox("Deposit Type", ['No Deposit', 'Refundable', 'Non Refund'])
customer_type = st.selectbox("Customer Type", ['Transient', 'Contract', 'Transient-Party', 'Group'])
market_segment = st.selectbox("Market Segment", ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups'])
hotel = st.selectbox("Hotel Type", ['Resort Hotel', 'City Hotel'])
required_car_parking_spaces = st.slider("Car Parking Spaces", 0, 5, 0)
total_special_requests = st.slider("Special Requests", 0, 5, 0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Cancellation"):
    sample_dict = {
        'lead_time': [lead_time],
        'hotel': [hotel],
        'previous_cancellations': [previous_cancellations],
        'previous_bookings_not_canceled': [previous_bookings_not_canceled],
        'booking_changes': [booking_changes],
        'deposit_type': [deposit_type],
        'customer_type': [customer_type],
        'market_segment': [market_segment],
        'required_car_parking_spaces': [required_car_parking_spaces],
        'total_of_special_requests': [total_special_requests]
    }

    sample_df = spark.createDataFrame(pd.DataFrame(sample_dict))
    prediction_result = model.transform(sample_df)
    prediction = prediction_result.select("prediction").collect()[0][0]

    if prediction == 1.0:
        st.error("❌ This booking is likely to be CANCELLED.")
    else:
        st.success("✅ This booking is likely to be HONORED (not cancelled).")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
---
Made with ❤️ by **Mubashar Ul Hassan** | PySpark & Streamlit
""")
