
import streamlit as st
import pandas as pd
import numpy as np
import base64




st.set_page_config(layout="wide")
import base64

def show_circular_logo(image_path, size=100):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    logo_html = f"""
    <div style='display: flex; align-items: center; justify-content: center;'>
        <img src='data:image/jpeg;base64,{encoded}' style='width: {size}px; height: {size}px; border-radius: 50%; border: 2px solid #ddd;'/>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    show_circular_logo("Auction-system_logo.jpg", size=100)  # Change file name if needed

with col2:
    st.markdown("<h1 style='margin-top: 20px;'>Online Auction Prediction System</h1>", unsafe_allow_html=True)

st.markdown("---")



    

# Sidebar input
st.sidebar.header("🔧 Auction Inputs")
item = st.sidebar.selectbox("Select Item", ["CartierWristwatch", "iPhone14", "DiamondRing", "SmartTV", "Laptop"])
auction_type = st.sidebar.selectbox("Auction Type", ["OneDay", "TwoDay", "ThreeDay"])
open_price = st.sidebar.number_input("Open Price (₹)", min_value=1, step=1)

if st.sidebar.button("Start Auction"):
    st.subheader(f"📦 Auction for: {item} ({auction_type})")

    num_bids = int(open_price * 1.3 + 3)
    np.random.seed(42)

# Step 2: Generate bid data
    bid_values = [round(open_price * np.random.uniform(1.5, 3.0) + i * 10, 2) for i in range(num_bids)]
    bid_times = [round(np.random.uniform(0.1, 1.0), 2) for _ in range(num_bids)]
    bidder_rates = [round(np.random.uniform(0.3, 1.0), 2) for _ in range(num_bids)]

# Step 3: Calculate closing prices
    closing_prices = []
    for i in range(num_bids):
       price = bid_values[i] + (bidder_rates[i] * 20) + (1 - bid_times[i]) * 10
       closing_prices.append(round(price, 2))

    final_price = max(closing_prices)

# Step 4: Build DataFrame
    result_data = {
    "item_id": [item] * num_bids,
    "auction_type": [auction_type] * num_bids,
    "bid_value": bid_values,
    "bid_time": bid_times,
    "bidder_rate": bidder_rates,
    "open_price": [open_price] * num_bids,
    "closing_price": closing_prices
   }

    df = pd.DataFrame(result_data)

# Round all numeric columns to 2 decimal places
    numeric_cols = ["bid_value", "bid_time", "bidder_rate", "open_price", "closing_price"]
    df[numeric_cols] = df[numeric_cols].round(2)

# Add highlight column (used for logic only)
    df["highlight"] = df["closing_price"] == final_price
    highlight_mask = df["highlight"]
    df_display = df.drop(columns=["highlight"])

# Define highlight function using row index
    def highlight_row(row_idx):
       return ['background-color: lightcoral'] * len(df_display.columns) if highlight_mask.iloc[row_idx] else [''] * len(df_display.columns)

# Style DataFrame
    styled_df = (
      df_display.style
      .format({col: "{:.2f}" for col in numeric_cols})
      .apply(lambda row: highlight_row(row.name), axis=1)
      .set_properties(**{"font-size": "18px"})
    )

# Show styled table
# Show number of bids
    st.info(f"🔢 Total Number of Bids: {num_bids}")

    st.dataframe(styled_df, use_container_width=True)

# Final closing price
    st.success("🏆 Final Closing Price: ₹{:.2f}".format(final_price))

# Show winner row only
    winner = df[df["closing_price"] == final_price].drop(columns=["highlight"]).reset_index(drop=True)
    st.write("🟩 Winning Bid Details:")
    st.dataframe(winner)
