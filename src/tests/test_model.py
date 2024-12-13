import re
import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import nltk
import joblib  # Pour charger le modèle sauvegardé

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger les ressources nécessaires (à exécuter une seule fois)
nltk.download('punkt')
nltk.download('wordnet')

# Initialisation globale des stopwords et lemmatizer
# stop_words = set(stopwords.words('english'))
wordlem = WordNetLemmatizer()

# Fonction pour retirer les stopwords et effectuer un prétraitement complet
def process_comment(comment):
    # Convertir en minuscule
    comment = comment.lower()

    # Supprimer les balises HTML
    comment = re.sub(r'<.*?>', '', comment)
    
    # Garder uniquement les lettres et les espaces
    comment = re.sub(r'[^a-z\s]', '', comment)

    # Supprimer les espaces multiples
    comment = re.sub(r'\s+', ' ', comment).strip()

    # Tokeniser le texte
    tokens = word_tokenize(comment)

    # Lemmatization
    final_tokens = [wordlem.lemmatize(w) for w in tokens]

    return ' '.join(final_tokens)


# Fonction de prétraitement et tokenisation
def preprocess_text(text, tokenizer, max_length):
    text_cleaned = process_comment(text)  # Nettoyage
    sequence = tokenizer.texts_to_sequences([text_cleaned])  # Conversion en séquence numérique
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')  # Padding
    return padded_sequence

# # Fonction principale pour tester le modèle
# def main(args):
#     # Charger le tokenizer sauvegardé
#     with open(args.tokenizer_path, 'rb') as f:
#         tokenizer = pickle.load(f)

#     # Charger le modèle sauvegardé
#     model = joblib.load(args.model_path)

#     # Prétraiter le commentaire fourni
#     preprocessed_comment = preprocess_text(args.comment, tokenizer, args.max_length)

#     # Faire la prédiction
#     prediction = model.predict(preprocessed_comment)[0][0]  # [0][0] pour extraire la valeur
#     toxicity = "Toxic" if prediction > 0.5 else "Non-Toxic"

#     # Afficher le résultat
#     print(f"Comment: {args.comment}")
#     print(f"Predicted Toxicity: {toxicity} (Score: {prediction:.2f})")

def main(args):
    # Charger le tokenizer sauvegardé
    print(args.tokenizer_path)
    print(args.model_path)
    print(args.max_length)
    print(args.comment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test d'un modèle de classification de toxicité.")
    parser.add_argument("--comment", type=str, required=True, help="Commentaire à tester.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Chemin vers le tokenizer sauvegardé.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle sauvegardé.")
    parser.add_argument("--max_length", type=int, default=100, help="Longueur maximale des séquences (par défaut: 100).")
    args = parser.parse_args()

    main(args)

