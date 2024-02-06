import os
import re
import string
string.punctuation
import spacy

parent_dir = os.getcwd()
reviews_preprocessed_dir = os.path.abspath(os.path.join(parent_dir, 'reviews_preprocessed'))

# Load spaCy model for english 
nlp = spacy.load('en_core_web_sm')

def drop_nan(my_dict):

    my_dict_nan = {}
    for platform, df in my_dict.items():

        my_dict_nan[platform] = df.dropna(subset=['text_preprocessed'], ignore_index=True)

    return my_dict_nan

def lemmatizer_spacy(text):
    """Lemmatize text using spaCy"""
    doc = nlp(' '.join(text))
    return [token.lemma_ for token in doc]

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenization(text):
    return re.split('\W+', text)

def remove_stopwords(text, stopwords):
    return [i for i in text if i not in stopwords]


def preprocess_only_english(df):

    '''Take a pandas df and a column name and return a preprocessed text
    (lowercasing, punctuation, stopwords, and lemmatized) but not tokenized.'''

    parent_dir = os.path.abspath(os.getcwd())

    relative_path = os.path.abspath(os.path.join(parent_dir, 'dataset/gist_stopwords.txt'))
    
    with open(relative_path, 'r') as f:
        stopwords = set(f.read().split(','))


    def combined_preprocessing(text):
        text = text.lower()
        text = remove_punctuation(text)
        tokens = tokenization(text)
        tokens = remove_stopwords(tokens, stopwords)
        lemmatized_tokens = lemmatizer_spacy(tokens)
        return ' '.join(lemmatized_tokens)
    
    all_reviews = {'android':df[df['device'] == 'android-all'].reset_index(drop=True),
                   'ios': df[df['device'] == 'ios-all'].reset_index(drop=True)}
    
    preprocessed_dict = {}

    for platform, df in all_reviews.items():
        
        df['text_preprocessed'] = df['english_translation'].apply(combined_preprocessing)
        preprocessed_dict[platform] = df

    return preprocessed_dict


def preprocess_to_excel(reviews_dict, start_date='', end_date=''):

    file_ios = f'reviews_preprocessed_ios_{start_date}_{end_date}.xlsx'
    file_android =  f'reviews_preprocessed_android_{start_date}_{end_date}.xlsx'

    filepath_ios = os.path.abspath(os.path.join(reviews_preprocessed_dir, file_ios))
    filepath_android = os.path.abspath(os.path.join(reviews_preprocessed_dir, file_android))  

    reviews_dict_preprocessed = preprocess_only_english(reviews_dict)

    reviews_dict_preprocessed['ios'].to_excel(filepath_ios, index=False)
    reviews_dict_preprocessed['android'].to_excel(filepath_android, index=False)

    return filepath_ios, filepath_android