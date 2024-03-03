"""
<<< Overview >>>
This program imports a dataset of Amazon reviews, isolates and cleans the
review data ready for NLP processing and sentiment analysis.
The main program offers the user a menu screen where they can select viewing
the polarity of an individual review or comparing the similarity of two.
The program then calculates and returns polarity or similarioty values as
requested, along with the original text and tokens for user assessment.

<<< Note to the reviewer >>>
From my reading of the task, it's my belief that the main purpose of this
program (beyond demonstrating my Python skills) is to facilitate my learning
and assessment of the model's strengths and limitations.
This is why I've taken the menu screen approach, rather than running sentiment
analysis for the entire DataFrame and exporting this data for visualisation.
"""

import spacy
import pandas as pd
from textblob import TextBlob

# Load a small-sized langauge model for polarity analysis
nlp_sm = spacy.load('en_core_web_sm')

# Load a medium-sized langauge model for similarity analysis
nlp_md = spacy.load('en_core_web_md')


# Read the .csv file and store into a Pandas DataFrame
df = pd.read_csv('amazon_product_reviews.csv')

# Drop rows with missing values in the 'reviews.text' column
clean_data = df.dropna(subset=['reviews.text'])

# Isolate and store the reviews.text column into a series for processing
reviews_data = clean_data['reviews.text']


def get_int(display_string: str) -> int:
    """
    Continues to ask the user for an int, until a valid int is entered

    Parameters:
    - display_string (str): Prompts user to enter an int.

    Returns:
    - int: A valid integer
    """
    # Continuously prompt the user until a valid integer is entered.
    while True:
        user_input = input(display_string)
        try:
            user_input = int(user_input)
            return user_input
        except ValueError:
            print("<<< Please enter an integer >>>")

def find_review(display_string: str) -> str:
    """
    Uses user-input index to retrieve and return review text from a list.

    Parameters:
    - display_string (input[str]): Prompts user to choose an index.

    Calls:
    - get_int(): Ensures user input is a valid int.

    Returns:
    - str: The review located at the chosen index.
    """
    while True:
        index = get_int(display_string)     # Prompt user to select index
        if index <= len(reviews_data):
            review = reviews_data[index]    # Locate user-selected review
            return review
        else:                               # Ensure input is in bounds
            print(f"<<< Max index: {len(reviews_data)} >>>")

def filter_and_tokenize(text: str) -> str:
    """
    Converts string to an NLP object using the small model.
    Tokenizes, lemmatizes and filters text to produce a clean list of relevant
    words.
    Joins the list into a str for further NLP processing.

    Parameters:
    - text (str): A string comprising review text to be processed.

    Returns:
    - str: A string comprised of cleaned lemmas.
    """
    # Make text an nlp object using the small model
    sentence = nlp_sm(text)

    tokens = [
        token.lemma_ for token in sentence       # Tokenize and lemmatize
        if not token.is_stop                     # Remove stop words
        if not token.is_punct or token.is_space  # Remove punct and whitespace
    ]

    # Join to str and make lowercase
    return ' '.join(tokens).lower()

def print_max_index() -> print:
    """Prints the number of indexes in DataFrame"""
    print(
        f"There are {len(reviews_data)} reviews in the DataFrame."
    )

def polarity_description(polarity_value: float) -> str:
    """
    Assigns a textual score to the polarity value.
    This gives the user a clear idea of the model's prediction of sentiment.

    Parameters:
    - polarity_value (float): A float between -1.000 and 1.000.

    Returns:
    - description (str): A string comprising the textual description.
    """
    # Assign description for polarity value
    if polarity_value >= 0.800:
        description = "Extremely positive"
    elif polarity_value >= 0.400:
        description = "Very positive"
    elif polarity_value >= 0.100:
        description = "Somewhat positive"
    elif polarity_value >= -0.100:
        description = "Neutral"
    elif polarity_value >= -0.400:
        description = "Somewhat negative"
    elif polarity_value >= -0.800:
        description = "Very negative"
    elif polarity_value >= -1.000:
        description = "Extremely negative"
    return description

def similarity_description(similarity_value: float) -> str:
    """
    Assigns a textual score to the similarity value.
    This gives the user a clear idea of the model's prediction of sentiment.

    Parameters:
    - similarity_value (float): A float between -1.000 and 1.000.

    Returns:
    - description (str): A string comprising the textual description.
    """
    # Assign description for similarity value
    if similarity_value >= 0.800:
        description = "Extremely similar"
    elif similarity_value >= 0.400:
        description = "Very similar"
    elif similarity_value >= 0.100:
        description = "Somewhat similar"
    elif similarity_value >= -0.100:
        description = "Neutral"
    elif similarity_value >= -0.400:
        description = "Somewhat dissimilar"
    elif similarity_value >= -0.800:
        description = "Very dissimilar"
    elif similarity_value >= -1.000:
        description = "Extremely dissimilar"
    return description

def get_polarity(display_string: str) -> print:
    """
    Locates review text, tokenizes and cleans text, calculates polarity value.
    Assigns a description for the polarity value.
    Displays results along with original text and all cleaned tokens.

    Parameters:
    - display_string (str): Prompts user to choose a review by index.

    Calls:
    - find_review(): Locates and returns user-chosen review.
    - filter_and_tokenize(): Tokenizes and cleans a given review text.
    - polarity_description(): Assigns a textual score to the polarity value.

    Returns:
    - None.
    """
    text = find_review(display_string)      # Locate user-selected review
    tokens = filter_and_tokenize(text)      # Clean and tokenize text
    text_blob = TextBlob(tokens)            # Convert text to TextBlob object
    value = text_blob.polarity              # Calculate polarity
    description = polarity_description(value) # Give value a textual score

    # Summarise and display all information for assessment
    print(
        f"\nOriginal text:\t{text}\n"
        f"Tokens only:\t{tokens}\n"
        f"Polarity value:\t{value:.3f}\n"
        f"Description:\t{description}"
    )

def get_similarity(display_string: str) -> print:
    """
    Locates review text, tokenizes and cleans text, calculates similarity
    value.
    Displays results along with all cleaned tokens.

    Parameters:
    - display_string (str): Prompts user to choose two reviews by index.

    Calls:
    - find_review(): Locates and returns user-chosen review.
    - filter_and_tokenize(): Tokenizes and cleans a given review text.
    - similarity_description(): Assigns textual score to similarity value.

    Returns:
    - None.
    """
    # Locate user-selected reviews
    text_1 = find_review(display_string)
    text_2 = find_review(display_string)

    # Tokenize and clean text
    tokens_1 = filter_and_tokenize(text_1)
    tokens_2 = filter_and_tokenize(text_2)

    # Convert to nlp objects using the medium model
    nlp_review_1 = nlp_md(tokens_1)
    nlp_review_2 = nlp_md(tokens_2)

    # Calculate similarity value and assign description
    value = nlp_review_1.similarity(nlp_review_2)
    description = similarity_description(value)

    # Summarise and display all information for assessment
    print(
        f"\nReview 1 Lemmas:\t{tokens_1}\n"
        f"Review 2 Lemmas:\t{tokens_2}\n"
        f"Similarity value:\t{value:.3f}\n"
        f"Description:\t\t{description}"
    )

def options_screen() -> print:
    """
    Displays main menu options to the user.

    Parameters:
    - None.

    Calls:
    - print_max_index(): Prints the number of indexes in DataFrame

    Returns:
    - None.
    """
    print()
    print_max_index()                   # Remind user of data series limits
    print()
    print("1. Display Review Polarity")
    print("2. Compare Similarity of Two Reviews")
    print("3. Exit Program")

def display_review_polarity():
    """
    Displays the polarity score of a selected review.
    Function is repeated until user chooses to break.

    Parameters:
    - None

    Calls:
    - get_polarity(): Finds, prepares text; calculates, returns polarity

    Returns:
    - None.
    """
    while True:
        get_polarity(
            f"Select an index up to {len(reviews_data)} "
            "for polarity analysis: "
        )
        user_input = input(
            "\nAnalysis complete. Hit enter to continue "
            "or type 'stop' for main menu: "
        )
        if user_input.lower() == 'stop':
            break

def display_reviews_similarity():
    """
    Displays the similarity value of two selected reviews.
    Function is repeated until user chooses to break.

    Parameters:
    - None

    Calls:
    - get_similarity(): Finds, prepares text; calculates, returns similarity

    Returns:
    - None
    """
    while True:
        get_similarity(
            f"Select an index up to {len(reviews_data)} "
            "for similarity analysis: "
        )
        user_input = input(
            "\nAnalysis complete. Hit enter to continue "
            "or type 'stop' for main menu: "
        )
        if user_input.lower() == 'stop':
            break

def main():
    """
    Main program loop.
    Offers user an option screen.
    Loop is broken by user selecting option 3.
    Error handling ensures valid options are chosen.

    Parameters:
    - None.

    Calls:
    - display_review_polarity(): Displays polarity score of a selected review.
    - display_reviews_similarity(): Displays similarity value of two reviews.

    Returns:
    - None
    """
    option_choice = 0

    # Run options screen
    while option_choice != 3:
        options_screen()
        option_choice = get_int("\nChoose Option 1-3: ")
        print()
        if option_choice == 1:              # Display Review Polarity
            display_review_polarity()
        elif option_choice == 2:            # Compare Similarity of Two Reviews
            display_reviews_similarity()
        elif option_choice == 3:            # Exit program
            print("Goodbye for now.")
        else:                               # Error handling
            print("Option not available.")

# Run main program
if __name__ == "__main__":
    main()
