import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))  # Adjust to project root
src_dir = os.path.join(root_dir, "src")
plots_dir = os.path.join(root_dir, "plots")

# Ensure plots directory exists
os.makedirs(plots_dir, exist_ok=True)

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

# EDA Analysis: Headline Length Statistics
cleaned_data['headline_length'] = cleaned_data['headline'].apply(len)
print("Headline Length Statistics:")
print(cleaned_data['headline_length'].describe())

# Publisher Analysis: Count articles per publisher
publisher_counts = cleaned_data['publisher'].value_counts()
print("\nTop 10 Publishers by Article Count:")
print(publisher_counts.head(10))

# Visualizations
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

plt.show()
