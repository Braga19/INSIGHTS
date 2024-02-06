import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import CountVectorizer

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

def count_words(df, threshold, exclude_word):
    '''Takes a pandas df already preprocessed and with a col All_Text(Title + Content),
    and creates a pandas df with words and their counting, filtering out words that
    are under the threshold or appearing in the exclude_word list'''

    review_text = df['text_preprocessed'].dropna().values.tolist()
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(review_text)
    
    word_counts = X.sum(axis=0)
    word_count_dict = {
    'Word': vectorizer.get_feature_names_out(),
    'Count': word_counts.tolist()[0]
    }
    word_count_df = pd.DataFrame(word_count_dict)
    
    # Sort in descending order 
    word_count_df = word_count_df.sort_values('Count', ascending=False)

    # Set filter out words under the threshold
    word_count_df = word_count_df[word_count_df['Count'] >= threshold]

    # Exclude words that compare in the list 
    word_count_df = filter_rows_by_values(word_count_df, 'Word', exclude_word)

    
    return word_count_df

def count_pair(df, threshold, exclude_pair):
    '''Takes a pandas df already preprocessed and with a col All_Text(Title + Content),
    and creates a pandas df with words and their counting, filtering out words that
    are under the threshold or appearing in the exclude_words list'''
    
    review_text = df['text_preprocessed'].dropna().values.tolist()

    vectorizer = CountVectorizer(ngram_range=(2,2))
    X = vectorizer.fit_transform(review_text)
    bigram_counts = X.sum(axis=0)
    
    bigram_count_dict = {
    'Word': vectorizer.get_feature_names_out(),
    'Count': bigram_counts.tolist()[0]
    }
    bigram_count_df = pd.DataFrame(bigram_count_dict)
    bigram_count_df = bigram_count_df.sort_values('Count', ascending=False)
    bigram_count_df = bigram_count_df[bigram_count_df['Count'] >= threshold]
    bigram_count_df = filter_rows_by_values(bigram_count_df, 'Word', exclude_pair)
    
    return bigram_count_df


def get_positive(df):

    '''Take a df with col "Rating" and return a subset with rating greater
    or equal to 4'''
    df =  df[df['rating'] >= 4]
    return df

def get_negative(df):
    
    '''Take a df with col "Rating" and return a subset with rating smaller
    or equal to 2'''
    df = df[df['rating'] <= 2]
    return df

def get_counts_english(my_dict, exclude_word, exclude_pair):

    '''Take as input 
    - a dictionnary with reviews divided by platform,
    - and two lists of words and pairs to exclude,
    and return a dictionnary with counting of words and pairs divided by positive(5-4) and negative(1-2) reviews'''

    df_preprocess_split = {}

    for platform in my_dict.keys():
        df_preprocess_split[platform] = {}
        df_preprocess_split[platform]['positive'] = get_positive(my_dict[platform])
        df_preprocess_split[platform]['negative'] = get_negative(my_dict[platform])

    polar = ['positive', 'negative']
    forms = ['word', 'pair']

    platform_count = {}

    for f in forms:
        
        platform_count[f] = {}

        for pol in polar:

            platform_count[f][pol] = {}

            for platform in my_dict.keys():
                
                if f == 'word':

                    platform_count[f][pol][platform] = count_words(df_preprocess_split[platform][pol], 5, exclude_word)
                    platform_count[f][pol][platform]['Total Review'] = my_dict[platform].shape[0]
                    platform_count[f][pol][platform][f'Total {pol.capitalize()}'] = df_preprocess_split[platform][pol].shape[0]


                    word_counts = {}
                    # define the list of words to count
                    for word in platform_count[f][pol][platform]['Word']:

                        # create a regular expression pattern
                        pattern = r'\b' + word + r'\b'

                        

                        # count the occurrences of the words in each block of text
                        word_counts[word] = df_preprocess_split[platform][pol]['text_preprocessed'].str.contains(pattern).sum()

                    word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Unique Count'])

                    platform_count[f][pol][platform] = platform_count[f][pol][platform].merge(word_counts_df, on='Word')

                    platform_count[f][pol][platform]['% Unique on Total Review'] = round(platform_count[f][pol][platform]['Unique Count']/platform_count[f][pol][platform]['Total Review']*100, 2)

                    platform_count[f][pol][platform][f'% Unique on Total {pol.capitalize()}'] = round(platform_count[f][pol][platform]['Unique Count']/platform_count[f][pol][platform][f'Total {pol.capitalize()}']*100, 2)

                    platform_count[f][pol][platform] = platform_count[f][pol][platform].sort_values(by=f'% Unique on Total Review', ascending=False).reset_index().drop(columns='index')

                else:

                    platform_count[f][pol][platform] = count_pair(df_preprocess_split[platform][pol], 2, exclude_pair)

                    #get total review in the dataset with both positive and negative
                    platform_count[f][pol][platform]['Total Review'] = my_dict[platform].shape[0]
                    platform_count[f][pol][platform][f'Total {pol.capitalize()}'] = df_preprocess_split[platform][pol].shape[0]

                    pairs_count = {}
                    # define the list of words to count
                    for word in platform_count[f][pol][platform]['Word']:

                        # create a regular expression pattern
                        pattern = r'\b' + word + r'\b'

                        # count the occurrences of the words in each block of text in the relative df (positive or negative)
                        pairs_count[word] = df_preprocess_split[platform][pol]['text_preprocessed'].str.contains(pattern).sum()

                    pairs_count_df = pd.DataFrame(list(pairs_count.items()), columns=['Word', 'Unique Count'])

                    platform_count[f][pol][platform] = platform_count[f][pol][platform].merge(pairs_count_df, on='Word')

                    platform_count[f][pol][platform]['% Unique on Total Review'] = round(platform_count[f][pol][platform]['Unique Count']/platform_count[f][pol][platform]['Total Review']*100, 2)

                    platform_count[f][pol][platform][f'% Unique on Total {pol.capitalize()}'] = round(platform_count[f][pol][platform]['Unique Count']/platform_count[f][pol][platform][f'Total {pol.capitalize()}']*100, 2)

                    platform_count[f][pol][platform] = platform_count[f][pol][platform].sort_values(by=f'% Unique on Total {pol.capitalize()}', ascending=False).reset_index().drop(columns='index')


    return platform_count



#### KEYWORDS VISUALIZATION

def num_fields_based_height(num_fields: int):
    padding = 150 # arbitrary value depending on legends
    row_size = 50 # arbitrary value
    return padding + row_size * num_fields

def num_entries_based_width(num_entries: int):
    padding = 150 # arbitrary value depending on legends
    entry_size = 110 # arbitrary value
    return padding + entry_size * num_entries

def hbar_words(my_dict, form, polar, platform, n, date):
    '''Take a dicitonnary with word extracted from topic_extraction:
    - form: word or pair (str)
    - polar: positive or negative (str)
    - platform: ios or android (str)
    - positve: if true change color in green, otherwise stays red
    and return a horizontal bar chart with top n words of the dataset
    showing percentage of words in total review'''

    assert n >= 5, 'You need to enter a number greater or equal to 5'    

    if polar == 'positive':
        color_code = '#008000'        
    else:
        color_code = '#ff0000'
    

    max_positive = my_dict[form]['positive'][platform]['Unique Count'].max()
    max_negative = my_dict[form]['negative'][platform]['Unique Count'].max()
    max_range = max(max_positive, max_negative)
    df = my_dict[form][polar][platform]
   
    fig = go.Figure(go.Bar(
                x=df['Unique Count'][:n],
                y=df['Word'][:n],
                orientation='h',
                marker=dict(color=color_code)))
              
    if platform == 'ios':
                                 
        platform = 'iOS'
                    
    else:

        platform = platform.capitalize()
                                                 
    fig.update_layout(yaxis=dict(autorange="reversed",
                                 tickmode= 'array',
                                 tickvals=np.arange(0, len(df), 1), 
                                 ticktext = df['Word'] +  '<br>(' + df[f'% Unique on Total Review'].astype(str) + '%' + ')'), 
                                 title_text=f"<b>{platform} - Top {n} {polar.capitalize()} {form.capitalize()} from Reviews ({date})</b>",
                                 font=dict(size = 15),
                    autosize=True,
                    width = num_entries_based_width(df[:n].shape[0]),
                    height = num_fields_based_height(df[:n].shape[0]))
    
    
    fig.update_xaxes(range=[0, max_range + 20])

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

    return graphJSON
