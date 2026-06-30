import streamlit as st
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Data Viewer Dashboard", page_icon="📊", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Pandas Data Frame")
st.write("Displaying uploaded dataset using Pandas DataFrame")

# -----------------------------
# Upload Excel File
# -----------------------------
uploaded_file = r"C:\Users\hp\Desktop\Real-Time_Sentimental_and_Trend_Analysis_of_social_media\tests\Social Media Data Collection.xlsx"

if uploaded_file:

    # Read Excel
    df = pd.read_excel(uploaded_file)

    # -----------------------------
    # Display DataFrame
    # -----------------------------
    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True, height=600)

    # -----------------------------
    # Full Data View
    # -----------------------------
    with st.expander("View Complete Dataset"):
        st.write(df)

    # -----------------------------
    # Download Button
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV", data=csv, file_name="dataset.csv", mime="text/csv"
    )

else:
    st.info("Please upload an Excel file to display data.")
