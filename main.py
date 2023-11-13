from __future__ import division  # Python 2 users only

import re

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import brown
import pandas as pd

# Make sure you've downloaded the necessary data from NLTK
nltk.download('brown')
nltk.download('punkt')  # for tokenization

# Retrieve all text IDs
text_ids = brown.fileids()

# Prepare data for the DataFrame
data = {'text_id': [], 'text': [], 'word': [], 'word_count': []}
for text_id in text_ids:
    words = brown.words(text_id)
    text = ' '.join(words)
    tokens = word_tokenize(text)  # Tokenizing the text
    token_count = len(tokens)  # Counting the tokens
    data['text_id'].append(text_id)
    data['text'].append(text)
    data['word'].append(tokens)  # Adding the tokens to the data
    data['word_count'].append(token_count)  # Adding the token count to the data

# Create DataFrame
df = pd.DataFrame(data)

# Sort the DataFrame by 'token_count' in descending order
df_sorted = df.sort_values(by='word_count', ascending=False)

# Save the sorted DataFrame to CSV
df_sorted.to_csv('brown_corpus_with_tokens_and_count_sorted.csv', index=False)

# Create a separate DataFrame for the top 10 highest token counts
top_12_df = df_sorted.head(12)

# Save the top_10_df DataFrame to a CSV file
top_12_df.to_csv('top_12_token_count.csv', index=False)

# Create a new DataFrame for sentences
sentence_data = {'text_id': [], 'text': []}
for index, row in top_12_df.iterrows():
    sentences = sent_tokenize(row['text'])
    for sentence in sentences:
        # Only add the sentence if it is not composed entirely of punctuation
        if not re.fullmatch(r'[^\w\s]|_', sentence):
            sentence_data['text_id'].append(row['text_id'])
            sentence_data['text'].append(sentence)

# Create the new DataFrame with each sentence as a separate record
sentences_df = pd.DataFrame(sentence_data)

# Remove any records that are only punctuation
sentences_df = sentences_df[~sentences_df['text'].str.fullmatch(r'^[\W_]*$')]

# Reset the index to get a linenumber column
sentences_df = sentences_df.reset_index(drop=True)
sentences_df.index = sentences_df.index + 1  # to start counting from 1
sentences_df['linenumber'] = sentences_df.index

# Save the sentences_df DataFrame to a CSV file
sentences_df.to_csv('brown_corpus_top_12_token_count.csv', index=False)


# Download the Gutenberg corpus
nltk.download('gutenberg')
from nltk.corpus import gutenberg

# Retrieve all text IDs
text_ids = gutenberg.fileids()

# Create a dictionary to hold our data
corpus_data = {'text_id': [], 'text': []}

# Loop through the text IDs and save the ID and the corresponding text
for text_id in text_ids:
    # Retrieve the words for the current text ID
    text_words = gutenberg.words(text_id)
    # Convert the words to a single string
    text = ' '.join(text_words)
    # Append the data to the dictionary
    corpus_data['text_id'].append(text_id)
    corpus_data['text'].append(text)

# Create a DataFrame from our dictionary
df_corpus = pd.DataFrame(corpus_data)

# Save the DataFrame to a CSV file
df_corpus.to_csv('gutenberg_corpus.csv', index=False)