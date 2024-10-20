import pandas as pd
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon')


#modulo 1: preprocesar el dataset
data=pd.read_csv("/home/cesaralonso/fuzzy-rule-based-sentiment-analysis/test_data.csv",encoding='ISO-8859-1')  
doc=data.sentence
sentidoc=data.sentiment

tweets=[]
senti=[]

for j in range(len(doc)):
    str1=data.sentence[j]
    tweets.append(str1)
    senti.append(data.sentiment[j])

def decontracted(phrase):   # text pre-processing 
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"@", "" , phrase)         # removal of @
        phrase =  re.sub(r"http\S+", "", phrase)   # removal of URLs
        phrase = re.sub(r"#", "", phrase)          # hashtag processing
    
        # general
        phrase = re.sub(r" t ", " not ", phrase)
        phrase = re.sub(r" re ", " are ", phrase)
        phrase = re.sub(r" s ", " is ", phrase)
        phrase = re.sub(r" d ", " would ", phrase)
        phrase = re.sub(r" ll ", " will ", phrase)
        phrase = re.sub(r" t ", " not ", phrase)
        phrase = re.sub(r" ve ", " have ", phrase)
        phrase = re.sub(r" m ", " am ", phrase)
        return phrase
    
for k in range(len(doc)):
    tweets[k]=decontracted(tweets[k])

#Modulo 2: Lexicon de sentimientos

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate positive and negative sentiment using VADER
def calculate_nltk_scores(text):
    # Get the sentiment scores for the text
    sentiment_scores = sia.polarity_scores(text)
    
    # Extract positive and negative scores
    pos_score = sentiment_scores['pos']
    neg_score = sentiment_scores['neg']
    
    return pos_score, neg_score

# Apply the sentiment analysis to each row and store results in new columns
data['TweetPos'], data['TweetNeg'] = zip(*data['sentence'].apply(calculate_nltk_scores))

# Save the updated DataFrame back to a new CSV file
output_file_path = '/home/cesaralonso/fuzzy-rule-based-sentiment-analysis/modulo2.csv'  # Update with the desired output path
data.to_csv(output_file_path, index=False)

print(f"Sentiment scores added and saved to {output_file_path}")