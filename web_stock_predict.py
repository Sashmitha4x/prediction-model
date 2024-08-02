import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(page_title="Stock Market Predictor", page_icon="💹", layout="wide")

st.title("Stock Market Predictor")

# Keep the text input empty for the user to type in their desired stock ticker
stock = st.text_input("Enter the Stock ID")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

try:
    if stock:  # Only attempt to download data if stock is not empty
        google_data = yf.download(stock, start, end)
        if google_data.empty:
            st.error("No data found for the given stock ID. Please enter a valid stock ID.")
        else:
            model = load_model("Latest_stock_price_model.keras")
            st.subheader("Stock Data")
            st.write(google_data)

            splitting_len = int(len(google_data) * 0.7)
            x_test = pd.DataFrame(google_data.Close[splitting_len:])

            def plot_graph(x_values, y_values, title, xaxis_title, yaxis_title, line_data=None, line_labels=None):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Close Price', line=dict(color='blue')))

                if line_data:
                    for i, (line_y_values, line_label) in enumerate(zip(line_data, line_labels)):
                        fig.add_trace(go.Scatter(x=x_values, y=line_y_values, mode='lines', name=line_label,
                                                 line=dict(color='orange' if i == 0 else 'green',
                                                           dash='dash' if i == 0 else 'dot')))

                fig.update_layout(
                    title=title,
                    xaxis_title=xaxis_title,
                    yaxis_title=yaxis_title,
                    height=500,  # Increased height
                    width=1000,  # Increased width
                    template='plotly_dark'
                )
                return fig

            # Options for user to select which graph to display
            options = ["None",
                       "Original Close Price and MA for 250 days",
                       "Original Close Price and MA for 200 days",
                       "Original Close Price and MA for 100 days",
                       "Original Close Price and MA for 100 days and MA for 250 days",
                       "Original Close Price vs Predicted Close price"]

            selected_option = st.selectbox("Select the graph to display", options)

            if selected_option == "Original Close Price and MA for 250 days":
                google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
                fig = plot_graph(
                    google_data.index,
                    google_data.Close,
                    title="Close Price and 250-Day Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    line_data=[google_data['MA_for_250_days']],
                    line_labels=['250-Day MA']
                )
                st.plotly_chart(fig)

            elif selected_option == "Original Close Price and MA for 200 days":
                google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
                fig = plot_graph(
                    google_data.index,
                    google_data.Close,
                    title="Close Price and 200-Day Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    line_data=[google_data['MA_for_200_days']],
                    line_labels=['200-Day MA']
                )
                st.plotly_chart(fig)

            elif selected_option == "Original Close Price and MA for 100 days":
                google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
                fig = plot_graph(
                    google_data.index,
                    google_data.Close,
                    title="Close Price and 100-Day Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    line_data=[google_data['MA_for_100_days']],
                    line_labels=['100-Day MA']
                )
                st.plotly_chart(fig)

            elif selected_option == "Original Close Price and MA for 100 days and MA for 250 days":
                google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
                google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
                fig = plot_graph(
                    google_data.index,
                    google_data.Close,
                    title="Close Price with 100-Day and 250-Day Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    line_data=[google_data['MA_for_100_days'], google_data['MA_for_250_days']],
                    line_labels=['100-Day MA', '250-Day MA']
                )
                st.plotly_chart(fig)

            elif selected_option == "Original Close Price vs Predicted Close price":
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(x_test[['Close']])

                x_data = []
                y_data = []

                for i in range(100, len(scaled_data)):
                    x_data.append(scaled_data[i - 100:i])
                    y_data.append(scaled_data[i])

                x_data, y_data = np.array(x_data), np.array(y_data)

                predictions = model.predict(x_data)

                inv_pre = scaler.inverse_transform(predictions)
                inv_y_test = scaler.inverse_transform(y_data)

                ploting_data = pd.DataFrame(
                    {
                        'original_test_data': inv_y_test.reshape(-1),
                        'predictions': inv_pre.reshape(-1)
                    },
                    index=google_data.index[splitting_len + 100:]
                )
                st.subheader("Original values vs Predicted values")
                st.write(ploting_data)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=google_data.index[:splitting_len + 100], y=google_data.Close[:splitting_len + 100],
                                         mode='lines', name='Data - Not Used', line=dict(color='gray', dash='dash')))
                fig.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['original_test_data'], mode='lines',
                                         name='Original Test Data', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=ploting_data.index, y=ploting_data['predictions'], mode='lines',
                                         name='Predicted Test Data', line=dict(color='red')))
                fig.update_layout(
                    title="Close Price vs Predicted Close Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500,  # Increased height
                    width=1000,  # Increased width
                    template='plotly_dark'
                )
                st.plotly_chart(fig)
    else:
        st.info("Please enter a stock ID to get started.")
except Exception as e:
    st.error(f"An error occurred: {e}. Please enter a valid stock ID.")

# Initialize session state for chatbot
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def get_chatbot_response(query):
    # This is a placeholder function. Replace it with chatbot logic.
    responses = {
        "What is the stock price of Apple?": "The current stock price of Apple is $150.",
        "How did the market perform today?": "The market closed higher today with the S&P 500 up 1.5%.",
        # Add more predefined responses or integrate with chatbot model
    }
    return responses.get(query, "Sorry, I don't have an answer for that question.")

def submit_callback():
    if st.session_state.input_text:
        st.session_state.messages.append({"role": "user", "text": st.session_state.input_text})
        response = get_chatbot_response(st.session_state.input_text)
        st.session_state.messages.append({"role": "bot", "text": response})
        st.session_state.input_text = ""  # Reset the input field
        # Trigger update without re-running the entire script
        st.session_state.show_chatbot = True

# Button to toggle chatbot visibility
if st.button("Chatbot"):
    st.session_state.show_chatbot = not st.session_state.show_chatbot

# Display the chatbot section based on the toggle state
if st.session_state.show_chatbot:
    st.subheader("Message with Chatbot")

    # Display messages in a UI
    chat_container = """
    <div style='background-color:#EAEDED;padding:10px;border-radius:10px;height:300px;overflow-y:scroll;'>
    """
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_container += f"<div style='background-color:#090979;padding:10px;margin:10px;border-radius:25px;float:right;clear:both;'>{message['text']}</div>"
        else:
            chat_container += f"<div style='background-color:#7B7D7D;padding:10px;margin:10px;border-radius:25px;float:left;clear:both;'>{message['text']}</div>"
    chat_container += "</div>"

    st.markdown(chat_container, unsafe_allow_html=True)

    # Chat input at the bottom
    query = st.text_input("Ask a question about the stock market:", key="input_text")

    # Disable the submit button if the input text is empty
    submit_button_disabled = not st.session_state.input_text.strip()

    st.button("Submit", on_click=submit_callback, disabled=submit_button_disabled)

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #888;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            border-top: 1px solid #ddd;
        }
    </style>
    <div class="footer">
        <p>Stock Market Predictor | Developed by [Sashmitha Nethranjana] | &copy; 2024</p>
    </div>
""", unsafe_allow_html=True)
