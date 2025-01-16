import pandas as pd
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Import necessary library
import pandas as pd
import numpy as np

from globals import chat_words, stopword, known_proper_nouns, spell  # Import the global variables

# Libraries for text preprocessing
import re
import string
import contractions
import spacy
import emoji
from nltk.corpus import stopwords
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from spellchecker import SpellChecker
from wordsegment import load, segment
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from deep_translator import GoogleTranslator
load()

# English vocabulary list used for removing non-english text refering to https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
with open('words_alpha.txt') as f:
    words_alpha = set(word.strip().lower() for word in f)

# spacy.prefer_gpu()

# Function to create a language detector component
def create_language_detector(nlp, name):
    return LanguageDetector()

# Register the language detector factory if it's not already registered
if not Language.has_factory("language_detector"):
    Language.factory("language_detector", func=create_language_detector)
    
# Load the English model from spaCy
nlp = spacy.load('en_core_web_trf', disable=['tok2vec', 'morphologizer','ner'])

# Add the language detector to the pipeline if not already present
if 'language_detector' not in nlp.pipe_names:
    nlp.add_pipe('language_detector', last=True)
    
# Libraries for model training
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Suppress warnings
from transformers import logging
import warnings
warnings.simplefilter("ignore", UserWarning)
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import PreTrainedModel
# Under sampling
def under_sampling(df, target_column):
    
    # Separate the features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Define the undersampler
    rus = RandomUnderSampler(sampling_strategy='not minority', random_state=101) # resample all classes but the minority class;

    # Apply the undersampler
    X_res, y_res = rus.fit_resample(X, y)
    
   # Combine the resampled features and target back into a DataFrame
    df_undersampled = pd.DataFrame(X_res, columns=X.columns)
    df_undersampled[target_column] = y_res
    # print(df_undersampled[target_column].value_counts())
    return df_undersampled

# Load dataset
def load_imdb_dataset():
    # Define the list of CSV file paths
    file_paths = [
        'data/imdb_cleaned_1.csv',
        'data/imdb_cleaned_2.csv',
        'data/imdb_cleaned_3.csv',
        'data/imdb_cleaned_4_5_6.csv',
        'data/imdb_cleaned_7.csv',
        'data/imdb_cleaned_8.csv',
        'data/imdb_cleaned_9.csv',
    ]
    # Load all CSV files into a single DataFrame
    dataframes = [pd.read_csv(file, usecols=['review', 'sentiment']) for file in file_paths]

    # Concatenate all dataframes
    imdb_df = pd.concat(dataframes, ignore_index=True)

    # Drop any rows with missing values
    imdb_df.dropna(inplace=True)
    
    # Class balance
    imdb_df = under_sampling(imdb_df, 'sentiment')
    
    #Split into train and test
    train_set, test_set = train_test_split(imdb_df, test_size=0.2, random_state=42)
    
    review_df= imdb_df['review']
    sentiment_df = imdb_df['sentiment']
    train_set, test_set, train_labels, test_labels  = train_test_split(review_df, sentiment_df, test_size=0.2, random_state=101)
    
    return train_set, test_set, train_labels, test_labels

# ************************************** Data Preparation  (Vectorization, Embedding)**************************************
# Function to remove HTML Tags
def remove_html_tags(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    return text

# Function to remove URL
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

# Punctuations is referring to some special characters like !, ? etc
def remove_punc(text):
    # Remove punctuation and add a space after removing punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim any leading or trailing whitespace
    text = text.strip()
    
    return text

# Function to convert chat words
def chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.lower() in chat_words:
            new_text.append(chat_words[i.lower()])
        else:
            new_text.append(i)
    return " ".join(new_text)

def correct_spelling(text):
    # Process the text with spaCy
    doc = nlp(text)
    corrected_tokens = []
    
    for token in doc:
        # print(token.text, ' - ',token.pos_)
        
        # Add the token if it's not a proper noun
        if token.pos_ in {'PROPN'} or token.text in known_proper_nouns:
            corrected_tokens.append(token.text)
        else:
            # Correct the token if it's misspelled
            # Check if the token is a noun
            
            corrected_word = spell.correction(token.text)
            corrected_tokens.append(corrected_word if corrected_word else token.text)
    
    # Combine the corrected tokens into a single text string
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

# Function to normalize elongated words (e.g., "sooooo" -> "so")
def normalize_elongated_words(text):
    # Replace elongated words with their standard forms
    regex = r'(\w)\1{2,}'
    return re.sub(regex, r'\1\1', text)

# Function to tokenize the text using spaCy
def tokenize_text(text):
    #Replace non-ASCII characters with spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.strip()
    return [tok.text for tok in nlp.tokenizer(text)]

def lemmatize_text(text):
    # Check if 'text' is a list
    if isinstance(text, list):
        text = ' '.join(text)
        
    # Lemmatize the text 
    doc = nlp(text)
    # for token in doc:
    #     print(token.text, token.pos_, token.lemma_)
    return [token.lemma_ for token in doc]

def detect_language(text):
    doc = nlp(text)
    return doc._.language

# Function to translate text to English if it's not in English
def translate_text(text):
    language = detect_language(text)['language']
    # print(f"{text} - {language}")
    if language != 'en':
        translated_text =  GoogleTranslator(source='auto', target='english').translate(text)
        # print(f"Translated text: {translated_text}")
        return translated_text
    return text

def remove_non_english_text(text):
    doc = nlp(text)
    cleaned_tokens = []
    
    for token in doc:
        # Keep tokens that are in the words list, are named entities, or are non-alphabetic
        if token.lower_ in words_alpha or token.ent_type_ or not token.is_alpha:
            cleaned_tokens.append(token.text)
    
    return " ".join(cleaned_tokens)
def preprocess_text(text):
    #Translate text to English if it's not in English
    text = translate_text(text) if text is not None else None
    # print(f'Translated text: {text}')
    # Converting to lowercase
    text = text.lower() if text is not None else None
    # print ('1. Converting to lowercase: ', text)
    
    # Removing HTML Tags
    text = remove_html_tags(text) if text is not None else None
    # print('2. Removing HTML Tags: ', text)
    
    # Remove URLs
    text = remove_url(text) if text is not None else None
    # print('3. Remove URLs: ', text)
    
    # Handling emojis
    text = emoji.demojize(text) if text is not None else None
    # print('9. Handling emojis: ', text)
    
    # Handling chat words
    text = chat_conversion(text).lower() if text is not None else None
    # print('7. Handling chat words: ', text)
    
    # Removing numbers
    text = re.sub(r'\d+', '', text) if text is not None else None
    # print('4. Removing numbers: ', text)
    
    #Expand contractions
    text = expand_contractions(text) if text is not None else None
    # print('5. Expand contractions: ', text)
    
    text = normalize_elongated_words(text) if text is not None else None
    # print('6. Removing elongated words: ', text)
    
    # Removing punctuations
    text = remove_punc(text) if text is not None else None
    # print('6. Removing punctuations: ', text)
    
    # Split concatenated words
    text = split_concatenated_words(text) if text is not None else None
    # print('11. Split concatenated words: ', text)
    
    # Spelling correction
    # text = TextBlob(text).correct().string
    text = correct_spelling(text) if text is not None else None
    # print('7. Spelling correction: ', text)
    
    # Removing stopwords
    text = remove_stopwords(text) if text is not None else None
    # print('8. Removing stopwords: ', text)
    
    # Tokenization
    tokens = tokenize_text(text) if text is not None else None
    # print('10. Tokenization: ', tokens)
    
    # Lemmatization
    tokens = lemmatize_text(tokens) if tokens is not None else None
    # print('12. Lemmatization: ', tokens)

    # Remove non-English words
    text = remove_non_english_text(' '.join(tokens)) if tokens is not None else None
    # print('12. Remove non-English words: ', text)

    if text is None and tokens is None:
        return None
    # return ' '.join(tokens)
    return text

# Function to remove stopwords such as a, an, the, at, etc
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopword])

def expand_contractions(text):
    return contractions.fix(text)

def split_concatenated_words(text):
    # Convert input list of words to a single string
    if isinstance(text, list):
        text = ' '.join(text)
        
    split_words = segment(text)

    return ' '.join(split_words)
# ************************************** Data Preparation  (Vectorization, Embedding)**************************************
# Preprocess data for text classification

# Vectorize the text data
def feature_extraction(X_train, X_test):
    X_train = ["" if pd.isna(text) else text for text in X_train]
    X_test = ["" if pd.isna(text) else text for text in X_test]

    vectorizer= TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    return X_train_vectors, X_test_vectors

# Preprocess data for deep learning
def preprocess_data_for_deep_learning(data):
    # Unpack the data
    train_set, test_set, train_labels, test_labels = data
    
    # Ensure all texts are strings in case there are any non-string types like float
    train_set = [str(text) for text in train_set]
    test_set = [str(text) for text in test_set]

    # Define the maximum number of words and sequence length
    vocab_size = max(len(set(text.split())) for text in train_set + test_set)
    # Analyze sequence lengths and adjust max_length if necessary
    all_texts = train_set + test_set
    sequence_lengths = [len(text.split()) for text in all_texts]
    max_length = min(max(sequence_lengths), 500)
    
    oov_tok = '<OOV>'
    # Initialize the tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_set)

    # Convert texts to sequences
    train_sequences = tokenizer.texts_to_sequences(train_set)
    test_sequences = tokenizer.texts_to_sequences(test_set)

    # Pad sequences
    # padding='post', truncating='post'
    X_train = pad_sequences(train_sequences, maxlen=max_length)
    X_test = pad_sequences(test_sequences, maxlen=max_length)
    
    return X_train, X_test, train_labels, test_labels, vocab_size, max_length

#****************************************** Predict User Sentence *****************************************************

def pred_user_sentence(pred_sentences, model, tokenizer=None):
    """
    Parameters
    ----------
    pred_sentences : str
        The sentence to be predicted.
    model : The model to be used for prediction (svm_model, lr_model, rf_model, cnn_model, lstm_model, cnn_lstm_model).
    tokenizer : The tokenizer to be used for pretrained models (optional).
    
    Returns
    -------
    tuple
        A tuple containing the predicted label, the keywords, and the confidence score.
    """
    
    if model is None:
        return None, None, None

    train_set, _, _, _ = load_imdb_dataset()

    labels = ["Negative", "Neutral", "Positive"]
    processed_sentences = preprocess_text(pred_sentences)
    keywords = tokenize_text(processed_sentences)

    if isinstance(model, PreTrainedModel):
        # Tokenize the sentence for the model
        encoded_input = tokenizer(
            processed_sentences,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoded_input)
            predictions = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        confidence_score, predicted_class_idx = torch.max(probabilities, dim=1)
        
        label = labels[predicted_class_idx.item()]
        confidence_score = confidence_score.item()

    elif hasattr(model, "predict_proba"):
        # Assuming the model has a predict_proba method (e.g., scikit-learn models)
        _, transformed_sentence = feature_extraction(train_set, [processed_sentences])
        probabilities = model.predict_proba(transformed_sentence)[0]
        
        predicted_class_idx = np.argmax(probabilities)
        label = labels[predicted_class_idx]
        confidence_score = float(probabilities[predicted_class_idx])

    elif hasattr(model, "save"):
        # Assuming the model is a deep learning model (e.g., Keras/TensorFlow model)
        _, processed_sentences_sequences, _, _, _, _ = preprocess_data_for_deep_learning(data=(train_set, [processed_sentences], None, None))
        
        predictions = model.predict(processed_sentences_sequences)
        probabilities = predictions[0]
        
        predicted_class_idx = np.argmax(probabilities)
        label = labels[predicted_class_idx]
        confidence_score =  float(probabilities[predicted_class_idx])

    else:
        _, transformed_sentence = feature_extraction(train_set, [processed_sentences])
        predicted_class_idx = model.predict(transformed_sentence)[0]
        label = labels[predicted_class_idx]
        confidence_score = 1.0  # Assuming the model does not provide probabilities

    return label, keywords, confidence_score


def load_pretrained_model_and_tokenizer(model_name):
    model_dir = f"model/{model_name.lower()}"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_and_predict(pred_sentence, model_name):
    """
    Parameters:
    pred_sentence: str
    model_name: str
    """
    print(pred_sentence, model_name)
    tokenizer = None

    if model_name not in ["bert", "xlnet", "roberta", "lr", "svm", "rf", "cnn", "lstm", "cnn_lstm"]:
        raise ValueError("Invalid model name. Please choose from 'bert', 'xlnet', 'roberta', 'lr', 'svm', 'rf', 'cnn', 'lstm', 'cnn_lstm'.")

    print(f"Loading model: {model_name}")
    # Check if the model is a pre-trained model from Hugging Face Transformers
    if model_name.lower() in ["bert", "xlnet", "roberta"]:
        # Load the pre-trained model and tokenizer from the model directory
        model, tokenizer = load_pretrained_model_and_tokenizer(model_name.lower())

    elif model_name.lower() in ["lr", "svm", "rf"]:
        # Load a machine learning model saved with joblib
        filename = f"model/{model_name.lower()}.sav"
        model = joblib.load(filename)
    else:
        # Load a deep learning model saved in Keras format
        filename = f"model/{model_name.lower()}.keras"
        model = tf.keras.models.load_model(filename)

    # Call the prediction function
    sentiment, keywords, confidence_score = pred_user_sentence(pred_sentence, model=model, tokenizer=tokenizer)
    return sentiment, keywords, confidence_score