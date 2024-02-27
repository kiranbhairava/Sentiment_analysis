import streamlit as st
import pandas as pd
import subprocess
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.express as px
import base64
import os

# Function to run the Scrapy spider with user input movie IDs
def run_scrapy_spider(movie_ids):
    try:
        for movie_id in movie_ids:
            subprocess.run(['scrapy', 'runspider', 'scrape.py', '-a', f'movie_ids={movie_id}', '-o', f'imdb_reviews_{movie_id}.csv'])
    except Exception as e:
        st.error(f"Error occurred during web scraping: {str(e)}")


def perform_sentiment_analysis(df):
    try:
        st.title("Movie Sentiment Analysis")

        # Load BERT model and tokenizer
        model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # List to store sentiment labels
        sentiment_labels = []

        for index, row in df.iterrows():
            text = row["reviews"]

            # Tokenize input text
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            inputs.to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_class = torch.argmax(outputs.logits).item()

            # Add sentiment label to the list
            if predicted_class == 4 or predicted_class == 3:
                sentiment_labels.append('Positive')
            elif predicted_class == 0 or predicted_class == 1:
                sentiment_labels.append('Negative')
            else:
                sentiment_labels.append('Neutral')

        # Add sentiment labels to DataFrame
        df['sentiment'] = sentiment_labels

        # Calculate sentiment counts
        sentiment_counts = df['sentiment'].value_counts()

        # Plot pie chart for sentiment distribution
        fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Display the sentiment analysis results and classification
        st.write("Sentiment Analysis Results:")
        for sentiment, count in sentiment_counts.items():
            st.write(f"{sentiment}: {count} reviews")

        # Classification based on sentiment counts
        total_reviews = sum(sentiment_counts)
        if 'Positive' in sentiment_counts:
            positive_percentage = sentiment_counts['Positive'] / total_reviews * 100
        else:
            positive_percentage = 0

        if 'Negative' in sentiment_counts:
            negative_percentage = sentiment_counts['Negative'] / total_reviews * 100
        else:
            negative_percentage = 0

        if 'Neutral' in sentiment_counts:
            neutral_percentage = sentiment_counts['Neutral'] / total_reviews * 100
        else:
            neutral_percentage = 0

        if positive_percentage > 10 and positive_percentage > negative_percentage and positive_percentage > neutral_percentage:
            st.success("Hit")
        elif negative_percentage > 10 and negative_percentage > positive_percentage and negative_percentage > neutral_percentage:
            st.error("Flop")
        else:
            st.warning("Average")

        if not df.empty:
            csv_data = df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="all_reviews.csv">Download all reviews as a csv file</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.subheader("Reviews in CSV format:")
            st.markdown(df.to_html(index=False, header=True), unsafe_allow_html=True)
        else:
            st.write('No reviews found.')

    except Exception as e:
        st.error(f"Error occurred during sentiment analysis: {str(e)}")


def main():
    st.title('Sentiment_Analysis')
    st.write('Enter the ID of the movie')

    # Input field for user to enter movie IDs
    movie_ids_input = st.text_input("Movie ID")

    if st.button('Enter'):
        movie_ids = movie_ids_input.split(',') if movie_ids_input else []
        if movie_ids:
            run_scrapy_spider(movie_ids)
            st.write('Analysis has been initiated.')
            
            # Flag to check if any data scraping is initiated
            data_scraped = False


            # Once the spider completes, read the generated CSV file
            for movie_id in movie_ids:
                csv_file = f'imdb_reviews_{movie_id}.csv'
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)

                        if not df.empty:


                        # Perform sentiment analysis on the reviews
                            perform_sentiment_analysis(df)

                        else:
                            st.warning(f"No reviews found for movie ID: {movie_id}")

                    except pd.errors.EmptyDataError:
                        st.warning("Enter the correct input.")

                else:
                    st.warning(f"No reviews found for movie ID: {movie_id}")

                   

        else:
            st.error('Please enter at least one movie ID.')


if __name__ == "__main__":
    main()


