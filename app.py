from flask import Flask, render_template, send_file, session, redirect, url_for, flash
from src.forms import KeywordsForm, DistributionForm, SearchPatternForm, ReviewForm
from src import api_dataAI, timing, text, topic_extraction
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from datetime import datetime, timedelta, date

parent_dir = os.getcwd()
reviews_dir = os.path.abspath(os.path.join(parent_dir, 'reviews_extractions'))
ratings_dir = os.path.abspath(os.path.join(parent_dir, 'cumulative_ratings_extractions'))
subset_dir = os.path.abspath(os.path.join(parent_dir, 'subset_reviews'))
reviews_preprocessed_dir = os.path.abspath(os.path.join(parent_dir, 'reviews_preprocessed'))

app = Flask(__name__)
app.secret_key = os.environ['SECRETKEY']



@app.route('/')
def home():
    title = "This is the title"
    return render_template('home.html', title=title)


@app.route('/get_reviews', methods=['POST', 'GET'])
def get_reviews_route():

    today = datetime.today()
    two_days_ago = today - timedelta(days=2)
    android_limit = datetime(2022, 3, 15)
    
    reviews_form = ReviewForm()
    graph_JSON_daily = None
    graph_JSON_weekly = None
    graph_JSON_monthly = None

    if reviews_form.validate_on_submit():

        start_date = reviews_form.start_date.data
        end_date = reviews_form.end_date.data
        output_type = reviews_form.output_type.data

        if start_date > end_date:
            flash('Start date cannot be greater than end date, unless you are Doctor Strange.', category='error')
            return render_template('get_reviews.html', reviews_form=reviews_form)
        elif start_date > today.date():
            flash('Start date cannot be greater than today. Do not rush, tomorow is coming :)', category='error')
            return render_template('get_revuews.html', reviews_form=reviews_form)
        elif start_date < android_limit.date():
            flash('Data.AI does not store android ratings older than March 15th 2022. For more info www.data.ai.')
        elif end_date > today.date():
            flash('End date cannot be greater than today. Do not rush, tomorow is coming :)', category='error')
            return render_template('get_reviews.html', reviews_form=reviews_form)
        elif end_date >= two_days_ago.date():
            flash('End date cannot be greater than or equal to two days ago. Data.AI takes usually two days for updating the reviews.', category='error')
            return render_template('get_reviews.html', reviews_form=reviews_form)
        
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        # Define your file paths
        reviews_filepath = os.path.abspath(os.path.join(reviews_dir, f'app_reviews_extraction_{start_date}_{end_date}.xlsx'))
        ratings_filepath = os.path.abspath(os.path.join(ratings_dir, f'daily_cumulative_average_{start_date}_{end_date}.xlsx'))

        # Check if the reviews file already exists
        if not os.path.isfile(reviews_filepath):
            # check if you can reconstruct the file
            general_reviews_name = 'app_reviews_extraction_'
            data_range_exist, file_path = timing.check_date_range(reviews_dir, start_date, end_date, general_reviews_name)
            if data_range_exist:
                df_for_timeframe = pd.read_excel(file_path)
                reviews_filepath = timing.timeframe_reviews_to_excel(df_for_timeframe, start_date, end_date)
                df_reviews = pd.read_excel(reviews_filepath)
            else:
                reviews_filepath = api_dataAI.get_ios_and_google_reviews(start_date, end_date)
                df_reviews = pd.read_excel(reviews_filepath) 
        else:
            df_reviews = pd.read_excel(reviews_filepath)
        
        # Do the same for ratings
        if not os.path.isfile(ratings_filepath):
            # check if you can reconstruct the file
            general_ratings_name = 'daily_cumulative_average_'
            data_range_exist, file_path = timing.check_date_range(ratings_dir, start_date, end_date, general_ratings_name)
            if data_range_exist:
                df_for_timeframe = pd.read_excel(file_path)
                ratings_filepath = timing.timeframe_ratings_to_excel(df_for_timeframe, start_date, end_date)
                df_ratings = pd.read_excel(ratings_filepath)
            else:
                ratings_filepath = api_dataAI.get_ios_and_google_ratings(start_date, end_date)
                df_ratings = pd.read_excel(ratings_filepath)
        else:
            df_ratings = pd.read_excel(ratings_filepath)
                         
 
        session['start_date'] = start_date
        session['end_date'] = end_date

        if output_type == 'daily':
            daily_distribution = timing.get_daily_reviews_distribution(df_reviews, df_ratings)
            graph_JSON_daily = timing.daily_comparison_rating(daily_distribution['ios'], daily_distribution['android'])
            return render_template('get_reviews.html', graph_JSON=graph_JSON_daily, reviews_form=reviews_form)
        elif output_type == 'weekly':
            weekly_distribution = timing.get_weekly_reviews_distribution(df_reviews, df_ratings)
            graph_JSON_weekly = timing.weekly_comparison_rating(weekly_distribution['ios'], weekly_distribution['android'])
            return render_template('get_reviews.html', graph_JSON=graph_JSON_weekly, reviews_form=reviews_form)
        elif output_type == 'monthly':
            monthly_distribution = timing.get_monthly_reviews_distribution(df_reviews, df_ratings)
            graph_JSON_monthly = timing.monthly_comparison_rating(monthly_distribution['ios'], monthly_distribution['android'])
            return render_template('get_reviews.html', graph_JSON=graph_JSON_monthly, reviews_form=reviews_form)

    else:
        return render_template('get_reviews.html', reviews_form=reviews_form)
    
@app.route('/download_reviews', methods=['GET'])
def download_reviews_route():

    start_date = session.get('start_date')
    end_date = session.get('end_date')

    if not start_date or not end_date:
            flash('Please fill out all fields before downloading the reviews', category='error')
            return redirect(url_for('get_reviews_route'))
    
    # Define your file paths
    reviews_filepath = os.path.abspath(os.path.join(reviews_dir, f'app_reviews_extraction_{start_date}_{end_date}.xlsx'))

    return send_file(reviews_filepath, as_attachment=True)

@app.route('/clear_session', methods=['GET', 'POST'])
def clear_session():
    # Clear the session
    session.clear()
    return redirect(url_for('get_reviews_route'))

@app.route('/review_volume', methods=['GET', 'POST'])
def review_volume():
    
    distribution_form = DistributionForm()

    weekly_reviews_volume_plot = None
    monthly_reviews_volume_plot = None
         
    if distribution_form.validate_on_submit():
        start_date_distribution = distribution_form.start_date.data
        end_date_distribution = distribution_form.end_date.data

        reviews_filepath = os.path.abspath(os.path.join(reviews_dir, f'app_reviews_extraction_{start_date_distribution}_{end_date_distribution}.xlsx'))
        ratings_filepath = os.path.abspath(os.path.join(ratings_dir, f'daily_cumulative_average_{start_date_distribution}_{end_date_distribution}.xlsx'))


        if not os.path.isfile(reviews_filepath):
            # check if you can reconstruct the file
            general_reviews_name = 'app_reviews_extraction_'
            data_range_exist, file_path = timing.check_date_range(reviews_dir, start_date_distribution, end_date_distribution, general_reviews_name)
            if data_range_exist:
                df_for_timeframe = pd.read_excel(file_path)
                df_reviews = timing.get_timeframe(df_for_timeframe, start_date_distribution, end_date_distribution)
            else:
                flash('Please download the dataset in the Reviews section before creating the timeseries', category='error')
                return redirect(url_for('review_volume'))  
        else:
            df_reviews = pd.read_excel(reviews_filepath)
    
        
         # Do the same for ratings
        if not os.path.isfile(ratings_filepath):
            general_ratings_name = 'daily_cumulative_average_'
            data_range_exist, file_path = timing.check_date_range(ratings_dir, start_date_distribution, end_date_distribution, general_ratings_name)
            if data_range_exist:
                df_for_timeframe = pd.read_excel(file_path)
                df_ratings = timing.get_timeframe(df_for_timeframe, start_date_distribution, end_date_distribution)
            else:
                flash('Please download the dataset in the Reviews section before creating the timeseries', category='error')
                return redirect(url_for('review_volume'))  
        else:
            df_ratings = pd.read_excel(ratings_filepath)


        # daily_distribution_dict = timing.get_daily_reviews_distribution(df_reviews, df_ratings)
        weekly_distribution_dict = timing.get_weekly_reviews_distribution(df_reviews, df_ratings)
        monthly_distribution_dict = timing.get_monthly_reviews_distribution(df_reviews, df_ratings)

        monthly_reviews_volume_plot = timing.monthly_reviews_distribution(monthly_distribution_dict['ios'], monthly_distribution_dict['android'])
        weekly_reviews_volume_plot = timing.weekly_reviews_distribution(weekly_distribution_dict['ios'], weekly_distribution_dict['android'])

    return render_template('review_volume.html', distribution_form=distribution_form, monthly_reviews_volume_plot=monthly_reviews_volume_plot, weekly_reviews_volume_plot=weekly_reviews_volume_plot)


@app.route('/keywords_extraction', methods=['GET', 'POST'])
def keywords_extraction():

    
    keywords_form = KeywordsForm()
    plot_keywords = None 
    
    if keywords_form.validate_on_submit():
        
        start_date = session.get('start_date')
        end_date = session.get('end_date')

        if not start_date or not end_date:
            flash('Please set a date range in the R&R Distribution section before getting the keywords', category='error')
            return redirect(url_for('keywords_extraction'))
        
        reviews_filepath = os.path.abspath(os.path.join(reviews_dir, f'app_reviews_extraction_{start_date}_{end_date}.xlsx'))
        df_reviews = pd.read_excel(reviews_filepath)  

        file_ios = f'reviews_preprocessed_ios_{start_date}_{end_date}.xlsx'
        file_android =  f'reviews_preprocessed_android_{start_date}_{end_date}.xlsx'

        filepath_ios = os.path.abspath(os.path.join(reviews_preprocessed_dir, file_ios))
        filepath_android = os.path.abspath(os.path.join(reviews_preprocessed_dir, file_android))        

        if not os.path.isfile(filepath_android) or not os.path.isfile(filepath_ios):
            #text preprocessing 
            filepath_preprocessed_ios, filepath_preprocessed_android = text.preprocess_to_excel(df_reviews, start_date, end_date)
            reviews_preprocessed_dict = {'ios': pd.read_excel(filepath_preprocessed_ios),'android': pd.read_excel(filepath_preprocessed_android)}  

        else:
            reviews_preprocessed_dict = {'ios': pd.read_excel(filepath_ios),'android': pd.read_excel(filepath_android)} 
        
        exclude_word =  ['very'] # add words to exclude like the example in the list
        exclude_pair = ['do not'] # add pair of words to exclude like the example in the list
        
        keywords_dict = topic_extraction.get_counts_english(reviews_preprocessed_dict, exclude_word, exclude_pair)
        
        plot_keywords = topic_extraction.hbar_words(keywords_dict, keywords_form.form.data, keywords_form.polar.data, keywords_form.platform.data,
                                                    keywords_form.n.data, keywords_form.date.data)
        

       
    return render_template('keywords_extraction.html', keywords_form=keywords_form, plot_keywords=plot_keywords)
    
@app.route('/search_keywords', methods=['GET', 'POST'])
def search_keywords():

    search_pattern_form = SearchPatternForm()
    subset_df = None
    
    if search_pattern_form.validate_on_submit():
            start_date = session.get('start_date')
            end_date = session.get('end_date')

            if not start_date or not end_date:
                flash('Please set a date range in the R&R Distribution section before getting the Reviews', category='error')
                return redirect(url_for('search_keywords'))    

            file_ios = f'reviews_preprocessed_ios_{start_date}_{end_date}.xlsx'
            file_android =  f'reviews_preprocessed_android_{start_date}_{end_date}.xlsx'

            filepath_ios = os.path.abspath(os.path.join(reviews_preprocessed_dir, file_ios))
            filepath_android = os.path.abspath(os.path.join(reviews_preprocessed_dir, file_android))   

            if not os.path.isfile(filepath_android) or not os.path.isfile(filepath_ios):
                flash('Please go back to the section Keywords Extraction to pre-processed the text', category='error')
                return redirect(url_for('search_keywords'))
            else:
                reviews_preprocessed_dict = {'ios': pd.read_excel(filepath_ios),'android': pd.read_excel(filepath_android)}         

            pattern_for_keywords = r'\b' + search_pattern_form.pattern.data + r'\b'
            pattern_for_excel = search_pattern_form.pattern.data
            platform = search_pattern_form.platform.data
            
            #drop NaN value in text_preprocessed column
            reviews_preprocessed_dict = text.drop_nan(reviews_preprocessed_dict)

            #get the subset
            mask = reviews_preprocessed_dict[platform]['text_preprocessed'].str.contains(pattern_for_keywords)
            subset_df = reviews_preprocessed_dict[platform][mask]
            subset_df = subset_df.loc[:, ['english_translation', 'rating', 'date']].reset_index(drop=True)

            #save subset file in the directory subset_reviews
            pattern_for_excel = pattern_for_excel.replace(' ','_')
            subset_filename = f'subset_reviews_containing_{pattern_for_excel}_{platform}_{start_date}_{end_date}.xlsx'
            subset_path = os.path.abspath(os.path.join(subset_dir, subset_filename))
            subset_df.to_excel(subset_path, index=False)
            subset_df = subset_df.to_html(classes='table table-striped')
            

            session['pattern'] = pattern_for_excel
            session['platform'] = platform


    return render_template('search_keywords.html', search_pattern_form=search_pattern_form, subset_df=subset_df)

@app.route('/download_subset', methods=['GET'])
def download_subset():

    start_date = session.get('start_date')
    end_date = session.get('end_date')
    pattern = session.get('pattern')
    platform = session.get('platform')
       
    

    if not start_date or not end_date or not pattern or not platform:
            flash('Please fill out all fields before downloading the file', category='error')
            return redirect(url_for('search_keywords'))
    
    # Define your file paths
    subset_filepath = os.path.abspath(os.path.join(subset_dir, f'subset_reviews_containing_{pattern}_{platform}_{start_date}_{end_date}.xlsx'))

    return send_file(subset_filepath, as_attachment=True)


if __name__ == '__main__':
    app.run()

  