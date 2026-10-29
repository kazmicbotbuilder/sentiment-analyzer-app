import streamlit as st
import anthropic
import pandas as pd
import plotly.express as px

st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")

# Anthropic API Key input
anthropic_api_key = st.sidebar.text_input(
    "Enter your Anthropic API Key", 
    type="password", 
    help="You can find your API key at https://console.anthropic.com/settings/keys"
)

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

def classify_sentiment_claude(review_text):
    """
    Classify the sentiment of a customer review using Claude Haiku 5.5.
    Parameters:
        review_text (str): The customer review text to be classified.
    Returns:
        str: The sentiment classification of the review as a single word, "positive", "negative", or "neutral".
             Returns "unknown" if the model's answer is not one of those three words.
    """
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,  # thinking tokens count toward this limit
        output_config={"effort": "low"},
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = "".join(b.text for b in response.content if b.type == "text")
    answer = answer.strip().strip(".\"'").lower()
    return answer if answer in VALID_SENTIMENTS else "unknown"


# CSV file uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with restaurant reviews", 
    type=["csv"])

# Once the user uploads a csv file:
if uploaded_file is not None: 
    # Read the file
    reviews_df = pd.read_csv(uploaded_file)

    # Check if the data has a text column
    text_columns = reviews_df.select_dtypes(include="object").columns

    if len(text_columns) == 0:
        st.error("No text columns found in the uploaded file.")

    # Show a dropdown menu to select the review column
    review_column = st.selectbox(
        "Select the column with the customer reviews",
        text_columns
    )

    if not anthropic_api_key:
        st.warning("Enter your Anthropic API key in the sidebar to analyze the reviews.")
        st.stop()

    # Analyze the sentiment of the selected column
    reviews_df["sentiment"] = reviews_df[review_column].apply(classify_sentiment_claude)
    
    # Display the sentiment distribution in metrics in 3 columns: Positive, Negative, Neutral
    # Make the strings in the sentiment column titled
    reviews_df["sentiment"] = reviews_df["sentiment"].str.title()
    sentiment_counts = reviews_df["sentiment"].value_counts()
    # st.write(reviews_df)
    # st.write(sentiment_counts)

    # Create 3 columns to display the 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        # Show the number of positive reviews and the percentage
        positive_count = sentiment_counts.get("Positive", 0)
        st.metric("Positive", 
                  positive_count, 
                  f"{positive_count / len(reviews_df) * 100:.2f}%")
    
    with col2:
        # Show the number of neutral reviews and the percentage
        neutral_count = sentiment_counts.get("Neutral", 0)
        st.metric("Neutral", 
                  neutral_count, 
                  f"{neutral_count / len(reviews_df) * 100:.2f}%")
    
    with col3:
        # Show the number of negative reviews and the percentage
        negative_count = sentiment_counts.get("Negative", 0)
        st.metric("Negative", 
                  negative_count, 
                  f"{negative_count / len(reviews_df) * 100:.2f}%")
        
    # Display pie chart
    fig = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title='Sentiment Distribution'
    )
    st.plotly_chart(fig)