import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Set up paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))  # Adjust to project root
src_dir = os.path.join(root_dir, "src")
plots_dir = os.path.join(root_dir, "plots")
outputs_dir = os.path.join(root_dir, "outputs")  # Create an 'outputs' folder to save text results

# Ensure directories exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Load raw data
data_path = os.path.join(src_dir, 'data.csv')
raw_data = pd.read_csv(data_path)

# Clean the data
cleaned_data = raw_data.dropna(subset=['headline', 'publisher', 'date'])
cleaned_data['headline'] = cleaned_data['headline'].str.strip()
cleaned_data['publisher'] = cleaned_data['publisher'].str.lower()
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
cleaned_data = cleaned_data.dropna(subset=['date'])
cleaned_data = cleaned_data.drop_duplicates()

# Save cleaned data
cleaned_data_path = os.path.join(src_dir, 'cleaned_data.csv')
cleaned_data.to_csv(cleaned_data_path, index=False)

# -------------------------
# Sentiment Analysis
# -------------------------
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment
def calculate_sentiment(headline):
    sentiment_score = sia.polarity_scores(headline)
    return sentiment_score['compound']

# Apply sentiment analysis to the 'headline' column
cleaned_data['sentiment_score'] = cleaned_data['headline'].apply(calculate_sentiment)

# Save the data with sentiment scores
cleaned_data_with_sentiment_path = os.path.join(src_dir, 'cleaned_data_with_sentiment.csv')
cleaned_data.to_csv(cleaned_data_with_sentiment_path, index=False)

# Save the Sentiment Analysis Results
sentiment_analysis_path = os.path.join(outputs_dir, 'sentiment_analysis_results.txt')
with open(sentiment_analysis_path, 'w') as f:
    f.write("Sentiment Analysis Results (headline and sentiment score):\n")
    f.write(str(cleaned_data[['headline', 'sentiment_score']].head()))
print(f"Saved: {sentiment_analysis_path}")

# -------------------------
# Topic Modeling with LDA
# -------------------------
# Vectorize the headlines
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(cleaned_data['headline'])

# Fit the LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

# Function to get top words from each topic
def get_top_words(lda_model, vectorizer, n_top_words=10):
    terms = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [terms[i] for i in top_indices]
        topic_words.append(top_words)
    return topic_words

# Get the top words from each topic
n_top_words = 10  # Display the top 10 words
topic_words = get_top_words(lda_model, vectorizer, n_top_words)

# Save the Topic Modeling Results
topic_modeling_path = os.path.join(outputs_dir, 'topic_modeling_results.txt')
with open(topic_modeling_path, 'w') as f:
    f.write("Top Words from each Topic:\n")
    for idx, words in enumerate(topic_words):
        f.write(f"Topic {idx+1}: {', '.join(words)}\n")
print(f"Saved: {topic_modeling_path}")

# -------------------------
# EDA Analysis: Headline Length Statistics
# -------------------------
cleaned_data['headline_length'] = cleaned_data['headline'].apply(len)
headline_length_desc = cleaned_data['headline_length'].describe()

# Save the Headline Length Statistics
headline_length_stats_path = os.path.join(outputs_dir, 'headline_length_stats.txt')
with open(headline_length_stats_path, 'w') as f:
    f.write("Headline Length Statistics:\n")
    f.write(str(headline_length_desc))
print(f"Saved: {headline_length_stats_path}")

# Publisher Analysis: Count articles per publisher
publisher_counts = cleaned_data['publisher'].value_counts()
top_publishers = publisher_counts.head(10)

# Save the Publisher Counts
publisher_counts_path = os.path.join(outputs_dir, 'publisher_counts.txt')
with open(publisher_counts_path, 'w') as f:
    f.write("Top 10 Publishers by Article Count:\n")
    f.write(str(top_publishers))
print(f"Saved: {publisher_counts_path}")

# -------------------------
# Visualizations
# -------------------------
# Plot: Distribution of Headline Lengths
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data['headline_length'], bins=50, kde=True)
plt.title('Distribution of Headline Lengths')
plt.xlabel('Headline Length')
plt.ylabel('Frequency')
headline_length_plot_path = os.path.join(plots_dir, 'headline_length_distribution.png')
plt.savefig(headline_length_plot_path)
print(f"Saved: {headline_length_plot_path}")

# Plot: Top 10 Publishers by Article Count
plt.figure(figsize=(12, 6))
publisher_counts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Article Count')
plt.xticks(rotation=45, ha='right')
top_publishers_plot_path = os.path.join(plots_dir, 'top_publishers.png')
plt.savefig(top_publishers_plot_path)
print(f"Saved: {top_publishers_plot_path}")

# Plot: Distribution of Dates (Time Series Analysis)
monthly_publications = cleaned_data.groupby(cleaned_data['date'].dt.to_period('M')).size()
plt.figure(figsize=(12, 6))
monthly_publications.plot(kind='line', color='purple')
plt.title('Article Publications Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Publications')
monthly_publications_plot_path = os.path.join(plots_dir, 'article_publications_over_time.png')
plt.savefig(monthly_publications_plot_path)
print(f"Saved: {monthly_publications_plot_path}")

# Plot: Hourly Distribution of Articles
cleaned_data['hour'] = cleaned_data['date'].dt.hour
hourly_distribution = cleaned_data['hour'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
hourly_distribution.plot(kind='bar', color='orange')
plt.title('Hourly Distribution of Article Publications')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Publications')
hourly_distribution_plot_path = os.path.join(plots_dir, 'hourly_publications_distribution.png')
plt.savefig(hourly_distribution_plot_path)
print(f"Saved: {hourly_distribution_plot_path}")

# Plot: Top 10 Words from Topics
fig, axes = plt.subplots(len(topic_words), 1, figsize=(10, 6 * len(topic_words)))
for i, top_words in enumerate(topic_words):
    axes[i].barh(range(len(top_words)), [1] * len(top_words), align='center')
    axes[i].set_yticks(range(len(top_words)))
    axes[i].set_yticklabels(top_words)
    axes[i].set_xlabel('Frequency')
    axes[i].set_title(f'Topic {i + 1}')

topic_words_plot_path = os.path.join(plots_dir, 'topic_words.png')
plt.tight_layout()
plt.savefig(topic_words_plot_path)
print(f"Saved: {topic_words_plot_path}")

# Show plots
plt.show()
