# finalCapstone
Task 21 in my Data Science Bootcamp - NLP Applications

# Sentiment Analysis of Amazon Product Reviews

This Python script performs sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) techniques and tools such as spaCy and TextBlob. The analysis provides insights into the sentiment polarity of individual reviews and the similarity between pairs of reviews.

## Prerequisites

Before running the script, ensure you have the required dependencies installed. You can install them using:

> pip install spacy pandas textblob

Additionally, download the spaCy English language model:

> python -m spacy download en_core_web_sm

## Usage

1. Clone the repository:

> git clone https://github.com/davemakebeer/finalCapstone.git  
> cd finalCapstone

2. Run the script:

> python sentiment_analysis.py


3. Follow the on-screen prompts to choose between displaying the sentiment polarity of a single review or comparing the similarity of two reviews. If you issues with the en_core_web_sm model when running .similarity, you may wish to close the program, change the code from _sm to _md and run again. However, this may not be necessary.

## Features

- **Sentiment Analysis:** Understand the sentiment polarity of individual Amazon product reviews, ranging from extremely positive to extremely negative.
- **Review Similarity Comparison:** Compare the similarity between two selected reviews to gauge how closely related their sentiments are.

## Functions

1. **Display Review Polarity:**
   - Select a review by index to view its sentiment polarity.

2. **Compare Similarity of Two Reviews:**
   - Choose two reviews by index to compare their sentiment similarity.

3. **Exit Program:**
   - Terminate the program.

## Error Handling

The script includes error handling to ensure valid inputs from the user, preventing issues related to choosing an index outside the bounds of the available reviews.
