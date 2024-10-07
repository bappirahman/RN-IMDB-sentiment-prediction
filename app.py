from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import streamlit as st

# Load the model
model = load_model('./assets/imdb_rnn_model.h5')

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
  """Transforms a sequence of word indices back into the original text.

  Args:
    text: A sequence of word indices.

  Returns:
    The original text.
  """
  return ' '.join([reverse_word_index.get(i, '?') for i in text])

def preprocess_text(text):
    """Preprocesses a text by converting it to lowercase, splitting it into words, replacing
    each word with its corresponding index in the word_index dictionary (or 2 if not present),
    adding 3 to the index, and padding the sequence to have a length of 500.

    Args:
        text: The text to be preprocessed.

    Returns:
        A 2D NumPy array of shape (1, 500) containing the preprocessed text.
    """
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    """Predicts the sentiment of a text, given as a string.

    Args:
        review: The text to be analyzed.

    Returns:
        A tuple containing the sentiment (either 'Positive' or 'Negative') and
        the probability of the sentiment, as a float between 0 and 1.
    """
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

st.title("IMDB Sentiment Analysis")

review = st.text_input("Enter a review")

sentiment, prediction = predict_sentiment(review)

st.write("Sentiment:", sentiment)
st.write('Prediction:', prediction)