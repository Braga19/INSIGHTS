import requests
import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()

from deep_translator import GoogleTranslator
from langdetect import detect
from datetime import datetime, timedelta

api_key = os.environ['APIKEY']
parent_dir = os.getcwd()
dataset_dir = os.path.abspath(os.path.join(parent_dir, 'dataset'))
reviews_dir = os.path.abspath(os.path.join(parent_dir, 'reviews_extractions'))
ratings_dir = os.path.abspath(os.path.join(parent_dir, 'cumulative_ratings_extractions'))


my_translator = GoogleTranslator(source='auto', target='english') 

#filtering countries according to user %
countries_df = pd.read_csv(os.path.join(dataset_dir,'User_Composition_withISOcountry.csv'), keep_default_na=False)
countries_df_subset = countries_df[countries_df['% of All Users'] >= 0.01]
country_code = countries_df_subset.Code.tolist()

def translate_google(text):
    try:
        # Only translate if the language is not English
        if detect(text) != 'en':
            result = my_translator.translate(text=text)
            return text if result is None else result
    except:
        pass # Add handling for undetectable language
    return text

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


def get_ios_and_google_reviews(start_date='', end_date=''):
     
    ios_reviews = get_reviews(start_date, end_date, 'ios')

    android_reviews = get_reviews(start_date, end_date, 'google-play')

    all_reviews = pd.concat([ios_reviews, android_reviews], ignore_index=True)
     
    file_name = f'app_reviews_extraction_{start_date}_{end_date}.xlsx'
    file_path = os.path.abspath(os.path.join(reviews_dir, file_name))

    all_reviews.to_excel(file_path, index=False)

    return file_path

def get_ios_and_google_ratings(start_date='', end_date=''):
     
     
    ios_rating = get_cumulative_rating(start_date, end_date, 'ios')
    android_rating = get_cumulative_rating(start_date, end_date, 'google-play')

    ios_rating['platform'] = 'ios'
    android_rating['platform'] = 'android'

    all_ratings = pd.concat([ios_rating, android_rating], ignore_index=True)
    file_name = f'daily_cumulative_average_{start_date}_{end_date}.xlsx'
    file_path = os.path.abspath(os.path.join(ratings_dir, file_name))

    all_ratings.to_excel(file_path, index=False)

    return file_path

def post_process_cumulative(df, platform):
      
      df['date'] = pd.to_datetime(df['date'])
      
      #drop countries with total count 0 to avoid affecting average
      df = filter_rows_by_values(df, 'total_count', [0]).reset_index(drop=True)
      
      #get the distribution per grouped by date
      distribution_per_date = pd.DataFrame(df.groupby('date')['average'].mean().round(2)).reset_index()
      
      if platform == 'ios':
            total_count_per_date = pd.DataFrame(df.groupby('date')['total_count'].sum()).reset_index()
      
      else:
            total_count_per_date = df[df['country_code'] == 'US']
            total_count_per_date = total_count_per_date.loc[:, ['date', 'total_count']]


      final_df = pd.merge(distribution_per_date, total_count_per_date, on='date')
      final_df = final_df.rename(columns={'average': 'cumulative_avg_general'})

      return final_df


def post_process_reviews(df, platform):

    '''Take a df of reviews from data.ai API, sorting date in ascending order, 
    dropping useless columns and reordering cols and creating a comment section'''

    df = df.sort_values(by='date', ascending=True).reset_index()

    if platform == 'google-play':
        df = df.drop(columns=['title', 'name', 'country', 'reviewer', 'index', 'review_id', 'language'])
        df = df.fillna('unknown')
        df['comments'] = ""
        df = df.rename(columns= {'text': 'content', 'language_name':'language'})
        new_order = ['date','version','rating','comments','content','english_translation','language','device', 'market_code']
        df = df[new_order]
        df['date'] = pd.to_datetime(df['date'])       

    else:
        df = df.drop(columns=['language', 'reviewer', 'name', 'index', 'review_id'])
        df['comments'] = ""
        df = df.rename(columns= {'text': 'content'})
        new_order = ['date','version','rating','comments', 'title', 'content','english_translation','country','device', 'market_code']
        df = df[new_order]    
        df['date'] = pd.to_datetime(df['date'])    
    
    return df

def get_reviews(start_date='', end_date='', platform=''):

    '''Request reviews from data.ai, taking as input:
    - start and end date in this format %YYYY-%MM-%DD
    - app store platform (ios or google-play)
    
    return a pandas df with reviews and relevant info
    - if excel True return a file excel'''
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')



    # setting correspondet app store id
    if platform == 'google-play':
        product_id = os.environ.get("GOOGLEPLAYID")  
    else:
        product_id = os.environ.get("APPLESTOREID")

    # Initialize an empty DataFrame to store all reviews
    all_reviews = pd.DataFrame()


    domain_url = 'https://api.data.ai'
    headers = {"Authorization": "Bearer " + api_key}

    while start_date < end_date:
        # Calculate the end date for this batch (start date + 30 days)
        batch_end_date = start_date + timedelta(days=30)

        # Make sure we don't go beyond the overall end date
        if batch_end_date > end_date:
            batch_end_date = end_date

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = batch_end_date.strftime('%Y-%m-%d')

        url = f'https://api.data.ai/v1.3/apps/{platform}/app/{product_id}/reviews?start_date={start_str}&end_date={end_str}'
        
        while url:
            try:
                r = requests.get(url, headers=headers)
                
                # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                r.raise_for_status()  
                
                json_file = r.json()
                
                df = pd.json_normalize(json_file, record_path='reviews')
                all_reviews = pd.concat([all_reviews, df], ignore_index=True)
                
                next_page = json_file.get('next_page')

                if next_page:
                    url = domain_url + next_page
                           
                else:
                    url = None

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                break

        # Adjust the date range for the next iteration
        start_date = batch_end_date + timedelta(days=1)
        
    # Translation in english
    if platform == 'google-play':

        all_reviews['english_translation'] = all_reviews['text'].apply(translate_google)

    else:

        all_reviews['english_translation'] = (all_reviews['title'] + ' - ' + all_reviews['text']).apply(translate_google)
         
    # Post-processing depending on the platform
    all_reviews_processed = post_process_reviews(all_reviews, platform)

    return all_reviews_processed



def get_cumulative_rating(start_date='', end_date = '', platform=''):
    
    '''Request cumulative ratings from data.ai, taking as input:
    - start and end date in this format %YYYY-%MM-%DD
    - app store platform (ios or google-play)
    
    return a pandas df with reviews and relevant info
    - if csv True return a file csv'''
    
    headers = {"Authorization": "Bearer " + api_key}

    urls = []
    
    query = 'https://api.data.ai/v1.3/intelligence/apps/{}/app/ratings_history?device={}-all&countries={}&start_date={}&end_date={}&granularity=daily&product_ids={}&feeds=cumulative_ratings'

    if platform == 'google-play':

        product_id = os.environ.get("GOOGLEPLAYID")
        device = 'android'

    else:
        product_id = os.environ.get("APPLESTOREID")
        device = platform
    
    for i in range(0, len(country_code), 10):
                    # Get the next 10 countries or less if there are less than 10 left
                    countries_chunk = country_code[i:i+10]
                    countries_str = '+'.join(countries_chunk)
                    new_query = query.format(platform, device, countries_str, start_date, end_date, product_id)
                    urls.append(new_query)

    cumulative_rating = pd.DataFrame()

    for url in urls:
            try:
                r = requests.get(url, headers=headers)

                r.raise_for_status()

                json_file = r.json()

                df = pd.json_normalize(json_file, record_path='list')
                cumulative_rating = pd.concat([cumulative_rating, df], ignore_index=True)
            
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                break
    
             
    cumulative_rating = post_process_cumulative(cumulative_rating, platform)

    return cumulative_rating
          
    