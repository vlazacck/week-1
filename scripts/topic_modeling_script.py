# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the cleaned data (make sure this path is correct based on your file location)
cleaned_data = pd.read_csv('../src/cleaned_data.csv')

# Preprocess the text (remove stopwords)
stop_words = set(stopwords.words('english'))
cleaned_data['processed_headline'] = cleaned_data['headline'].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
)

# Vectorize the headlines using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_data['processed_headline'])

# Perform LDA for topic modeling
n_topics = 5  # Number of topics you want to extract
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(tfidf_matrix)

# Extract topics and their words
n_top_words = 10  # Number of words per topic
feature_names = tfidf_vectorizer.get_feature_names_out()
topics = []

for topic_idx, topic in enumerate(lda_model.components_):
    topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topics.append(f"Topic {topic_idx + 1}: " + " ".join(topic_words))

# Output the topics
for topic in topics:
    print(topic)
