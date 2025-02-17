import re
import pickle
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download necessary resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Global initialization of lemmatizer
wordlem = WordNetLemmatizer()

# Function to remove stopwords and perform complete preprocessing
def process_comment(comment):
    # Convert to lowercase
    comment = comment.lower()

    # Remove HTML tags
    comment = re.sub(r'<.*?>', '', comment)
    
    # Keep only letters and spaces
    comment = re.sub(r'[^a-z\s]', '', comment)

    # Remove multiple spaces
    comment = re.sub(r'\s+', ' ', comment).strip()

    # Tokenize the text
    tokens = word_tokenize(comment)

    # Lemmatization
    final_tokens = [wordlem.lemmatize(w) for w in tokens]

    return ' '.join(final_tokens)


# Function for preprocessing and tokenization
def preprocess_text(text, tokenizer, max_length):
    text_cleaned = process_comment(text)  # Cleaning
    sequence = tokenizer.texts_to_sequences([text_cleaned])  # Convert to numerical sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')  # Padding
    return padded_sequence

# Main function to test the model
def main(args):
    # Load the saved tokenizer
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the saved model
    model = load_model(args.model_path)

    # Preprocess the provided comment
    preprocessed_comment = preprocess_text(args.comment, tokenizer, args.max_length)

    # Make a prediction
    prediction = model.predict(preprocessed_comment)[0][0]  # [0][0] to extract the value
    toxicity = "Non-Toxic"
    if prediction > 0.8:
        toxicity = "Highly Toxic"
    elif prediction > 0.6:
        toxicity = "Toxic"
    elif prediction > 0.4:
        toxicity = "Slightly Toxic"

    # Display the result
    print(f"Comment: {args.comment}")
    print(f"Predicted Toxicity: {toxicity} (Score: {prediction:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a toxicity classification model.")
    parser.add_argument("--comment", type=str, required=True, help="Comment to test.")
    parser.add_argument("--tokenizer_path", type=str, required=False, default="src/tokenizer.pickle", help="Path to the saved tokenizer.")
    parser.add_argument("--model_path", type=str, required=False, default="src/model/toxicity_model_0.keras", help="Path to the saved model.")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length (default: 100).")
    args = parser.parse_args()

    main(args)

# Use the following command to run the test
# python src/tests/test_model.py --comment "I hate you" --tokenizer_path src/tokenizer.pickle --model_path src/model/toxicity_model_0.keras
