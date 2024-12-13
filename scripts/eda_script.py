# Importing necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import re

# Define file paths
root_dir = os.path.abspath("../")  # Adjust if necessary
src_dir = os.path.join(root_dir, "src")
plots_dir = os.path.join(root_dir, "plots")
outputs_dir = os.path.join(root_dir, "outputs")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Load the data
data_path = os.path.join(src_dir, 'cleaned_data.csv')
raw_data = pd.read_csv(data_path)

# Clean data
cleaned_data = raw_data.dropna(subset=['headline', 'publisher', 'date'])
cleaned_data['headline'] = cleaned_data['headline'].str.strip()
cleaned_data['publisher'] = cleaned_data['publisher'].str.lower()
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
cleaned_data = cleaned_data.dropna(subset=['date']).drop_duplicates()

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
cleaned_data['sentiment_score'] = cleaned_data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Topic Modeling with LDA
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(cleaned_data['headline'])
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

# Get topic words
terms = vectorizer.get_feature_names_out()
n_top_words = 10
topic_words = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    topic_words.append([terms[i] for i in top_indices])

# Publisher Analysis
publisher_counts = cleaned_data['publisher'].value_counts()
top_publishers = publisher_counts.head(10)

# Publisher Domain Extraction
def extract_domain(publisher):
    match = re.search(r'@([a-zA-Z0-9.-]+)', publisher)
    return match.group(1) if match else 'unknown'

cleaned_data['publisher_domain'] = cleaned_data['publisher'].apply(extract_domain)
domain_counts = cleaned_data['publisher_domain'].value_counts()
top_domains = domain_counts.head(10)

# Visualization: Top 10 Publishers by Article Count
plt.figure(figsize=(12, 6))
publisher_counts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Article Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_publishers.png'))
plt.show()

# Visualization: Top 10 Domains by Publisher Count
plt.figure(figsize=(12, 6))
domain_counts.head(10).plot(kind='bar', color='lightgreen')
plt.title('Top 10 Domains by Publisher Count')
plt.xlabel('Domain')
plt.ylabel('Publisher Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_domains.png'))
plt.show()

# Visualization: Headline Length Distribution
cleaned_data['headline_length'] = cleaned_data['headline'].apply(len)
plt.figure(figsize=(12, 6))
sns.histplot(cleaned_data['headline_length'], bins=50, kde=True)
plt.title('Headline Length Distribution')
plt.xlabel('Headline Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'headline_length_distribution.png'))
plt.show()

# Visualization: Topic Words from LDA
fig, axes = plt.subplots(len(topic_words), 1, figsize=(10, 6 * len(topic_words)))
for i, top_words in enumerate(topic_words):
    axes[i].barh(range(len(top_words)), [1] * len(top_words), align='center')
    axes[i].set_yticks(range(len(top_words)))
    axes[i].set_yticklabels(top_words)
    axes[i].set_xlabel('Frequency')
    axes[i].set_title(f'Topic {i + 1}')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'topic_words.png'))
plt.show()

# Saving the results to the outputs directory
topic_modeling_path = os.path.join(outputs_dir, 'topic_modeling_results.txt')
with open(topic_modeling_path, 'w', encoding='utf-8') as f:  # Set encoding to 'utf-8'
    f.write("Top Words from each Topic:\n")
    for idx, words in enumerate(topic_words):
        f.write(f"Topic {idx+1}: {', '.join(words)}\n")
print(f"Saved: {topic_modeling_path}")

# Saving the sentiment analysis results
sentiment_analysis_path = os.path.join(outputs_dir, 'sentiment_analysis_results.txt')
with open(sentiment_analysis_path, 'w', encoding='utf-8') as f:  # Set encoding to 'utf-8'
    f.write("Sentiment Analysis Results (headline and sentiment score):\n")
    f.write(str(cleaned_data[['headline', 'sentiment_score']].head()))
print(f"Saved: {sentiment_analysis_path}")

# Saving the top publishers and domains analysis results
publisher_counts_path = os.path.join(outputs_dir, 'publisher_counts.txt')
with open(publisher_counts_path, 'w', encoding='utf-8') as f:  # Set encoding to 'utf-8'
    f.write("Top 10 Publishers by Article Count:\n")
    f.write(str(top_publishers))
print(f"Saved: {publisher_counts_path}")

domain_counts_path = os.path.join(outputs_dir, 'domain_counts.txt')
with open(domain_counts_path, 'w', encoding='utf-8') as f:  # Set encoding to 'utf-8'
    f.write("Top 10 Domains by Publisher Count:\n")
    f.write(str(top_domains))
print(f"Saved: {domain_counts_path}")
