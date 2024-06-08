import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from textblob import TextBlob
from fpdf import FPDF

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the dataset
dataframe = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Select the 'review.text' column and drop missing values
reviews_data = dataframe['reviews.text']
clean_data = reviews_data.dropna()

# Preprocess text data
def preprocess_text(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.text.lower() not in STOP_WORDS and token.text not in string.punctuation:
            tokens.append(token.lemma_)
    return " ".join(tokens)

clean_data = clean_data.apply(preprocess_text)

# Choose two product reviews for comparison
review1 = clean_data.iloc[0]
review2 = clean_data.iloc[1]

# Calculate similarity between the two reviews
similarity_score = nlp(review1).similarity(nlp(review2))

# Sentiment analysis function using TextBlob polarity attribute
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment

# Test the model on sample reviews
sample_reviews = [
    "This product is excellent! It exceeded my expectations.",
    "I am very disappointed with this product. It broke after one use.",
    "It's okay, not great but not terrible either."
]

for review in sample_reviews:
    preprocessed_review = preprocess_text(review)
    sentiment = analyze_sentiment(preprocessed_review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")

# Generate report
summary = """
Amazon Product Reviews Sentiment Analysis Report

Description of the Dataset:
The dataset consists of consumer reviews of Amazon products. Each review is represented by a column 'reviews.text', which contains the text of the review.

Preprocessing Steps:
1. Loaded the dataset and selected the 'reviews.text' column.
2. Removed missing values.
3. Preprocessed the text data by removing stopwords and punctuation, and lemmatizing the tokens.

Evaluation of Results:
The sentiment analysis function was tested on sample product reviews. The model classified the sentiment of each review as positive, negative, or neutral based on the sentiment score.

Insights:
The model performed reasonably well on the sample reviews. However, the accuracy of the sentiment analysis can be improved by fine-tuning the model and using a more sophisticated sentiment analysis approach. The current model may struggle with detecting nuanced sentiments or handling complex sentences.

Strengths:
- Simple and quick preprocessing steps.
- Easy to understand and implement sentiment analysis.

Limitations:
- Limited accuracy due to the basic sentiment analysis model.
- May not handle complex sentences and nuanced sentiments well.
"""

# Create PDF using FPDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Sentiment Analysis Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

pdf = PDF()
pdf.set_left_margin(10)
pdf.set_right_margin(10)
pdf.add_chapter('Summary', summary)

pdf.output('sentiment_analysis_report.pdf')
