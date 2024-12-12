import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load cleaned data (update the path if necessary)
cleaned_data = pd.read_csv('../src/data.csv')  # Update with the actual path if needed

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to the 'headline' column
cleaned_data['sentiment_score'] = cleaned_data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment based on the score
def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

cleaned_data['sentiment'] = cleaned_data['sentiment_score'].apply(classify_sentiment)

# Save the results to a CSV file
cleaned_data.to_csv('sentiment_analysis_results.csv', index=False)

print("Sentiment analysis results saved to sentiment_analysis_results.csv")
