import pandas as pd 
import numpy as np 
import os 
import re
import json

from datetime import timedelta
from pandas.tseries.offsets import Week
import pandas.tseries.offsets as offsets

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

parent_dir = os.getcwd()
reviews_dir = os.path.abspath(os.path.join(parent_dir, 'reviews_extractions'))
ratings_dir = os.path.abspath(os.path.join(parent_dir, 'cumulative_ratings_extractions'))
plots_dir = os.path.abspath(os.path.join(parent_dir, 'plots'))

def sort_date(df, colname):

    '''Sort df col already transformed in datetime64 in ascending order'''

    return df.sort_values(by=colname, ascending=True)

def get_timeframe(df, start_date, end_date):

    '''Take a df with a col with date and a timeframe in the format YYYY-MM-DD:
    Convert the col to datetime, sort it in ascending order and subset time equal or greater to date''' 
    
    df['date'] = pd.to_datetime(df['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    return df.reset_index(drop=True)

def check_date_range(directory, start_date, end_date, pattern):
    files = os.listdir(directory)
    date_files = [file for file in files if re.match(fr'{pattern}.*\.xlsx', file)]
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    for file in date_files:
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', file)
        file_start_date, file_end_date = map(pd.to_datetime, dates)
        
        if file_start_date <= start_date <= file_end_date and file_start_date <= end_date <= file_end_date:
            


            return True, os.path.abspath(os.path.join(directory, file))
            
    return False, None

def timeframe_reviews_to_excel(df_to_subset, start_date, end_date):

    df_timeframe = get_timeframe(df_to_subset, start_date, end_date)

    file_name = f'app_reviews_extraction_{start_date}_{end_date}.xlsx'
    file_path = os.path.abspath(os.path.join(reviews_dir, file_name))
    df_timeframe.to_excel(file_path, index=False)

    return file_path

def timeframe_ratings_to_excel(df_to_subset, start_date, end_date):

    df_timeframe = get_timeframe(df_to_subset, start_date, end_date)

    file_name = f'daily_cumulative_average_{start_date}_{end_date}.xlsx'
    file_path = os.path.abspath(os.path.join(ratings_dir, file_name))
    df_timeframe.to_excel(file_path, index=False)

    return file_path






###### DAILY DISTIBUTIONS #########

def get_daily_reviews_distribution(df_reviews, rating_df):

    '''Input datasets from data.ai API: 
    - df_reviewss: dataset with reviews and ratings (cumulative average)
    - general_df: daily distribution of rating
    - platform: ios or android
    - date_col: name of the datetime column
    return a dataset with daily statistics:
    - reviews: volume of reviews
    - avg_rating_reviews: average rating of reviews on volume
    - cumulative_avg_general: cumulative average rating over time on total_count
    - total_count: lifetime ratings app (reviews&ratings)'''

    all_reviews = {'android':df_reviews[df_reviews['device'] == 'android-all'].reset_index(drop=True),
                   'ios': df_reviews[df_reviews['device'] == 'ios-all'].reset_index(drop=True)}
    
    all_ratings = {'android': rating_df[rating_df['platform'] == 'android'].reset_index(drop=True),
                   'ios': rating_df[rating_df['platform'] == 'ios'].reset_index(drop=True)}

    distribution_dict = {}

    for platform, df in all_reviews.items():
   

        daily_rating_mean = pd.DataFrame(df.groupby(pd.Grouper(key='date', freq='D'))['rating'].mean().round(2)).reset_index()

        distribution_df = pd.DataFrame(df.groupby(pd.Grouper(key='date', freq='D'))['rating'].size()).reset_index()

        distribution_df = distribution_df.rename(columns={'rating': 'reviews'})

        df_review_processed = pd.merge(distribution_df, daily_rating_mean, how='right', on=['date'])
        
        final_df =  pd.merge(all_ratings[platform], df_review_processed, on='date', how='right')
        
        final_df = final_df.rename(columns={'rating': 'avg_rating_reviews'})


        if platform == 'android':

            final_df['weighted_avg'] = (final_df['cumulative_avg_general']*0.9 + final_df['avg_rating_reviews']*0.1).round(2)

            new_order = ['date', 'avg_rating_reviews','reviews', 'cumulative_avg_general', 'weighted_avg', 'total_count', 'platform']

            final_df = final_df[new_order]

            distribution_dict[platform] = final_df

        else:

            new_order = ['date', 'avg_rating_reviews','reviews', 'cumulative_avg_general', 'total_count', 'platform']

            final_df = final_df[new_order]

            final_df['cumulative_avg_general'] = final_df['cumulative_avg_general'].round(2)

            distribution_dict[platform] = final_df

    return distribution_dict

###### WEEKLY DISTIBUTIONS #########

def get_weekly_reviews_distribution(df_reviews, rating_df):

    '''Input datasets from data.ai API: 
    - general_df: daily distribution of rating (cumulative average)
    - df_reviews: dataset with reviews and ratings
    - platform: ios or android
    - date_col: name of the datetime column
    return a dataset with daily statistics:
    - reviews: volume of reviews
    - avg_rating_reviews: average rating of reviews on volume
    - cumulative_avg_general: cumulative average rating over time on total_count
    - total_count: lifetime ratings app (reviews&ratings)
    - comments: incomplete week or None'''

    all_reviews = {'android':df_reviews[df_reviews['device'] == 'android-all'].reset_index(drop=True),
                   'ios': df_reviews[df_reviews['device'] == 'ios-all'].reset_index(drop=True)}
    
    all_ratings = {'android': rating_df[rating_df['platform'] == 'android'].reset_index(drop=True),
                   'ios': rating_df[rating_df['platform'] == 'ios'].reset_index(drop=True)}
    
    distribution_dict = {}

    for platform, df in all_reviews.items():


        week_distribution_rating_reviews_df = df.groupby(pd.Grouper(key='date', freq='W')).agg({'rating': 'mean',
                                                                                                        'device': 'size'}).reset_index()    
        
        week_distribution_cumulative_rating = all_ratings[platform].groupby(pd.Grouper(key='date', freq='W')).agg({'cumulative_avg_general': 'mean',
                                                                                                                'total_count': 'last'}).reset_index()

        week_distribution_df = pd.merge(week_distribution_rating_reviews_df, week_distribution_cumulative_rating, on='date')

        week_distribution_df = week_distribution_df.rename(columns={'rating':'avg_rating_reviews', 'device': 'reviews'})

        week_distribution_df['date'] = week_distribution_df['date'].dt.to_period('W')
        week_distribution_df[['start_date', 'end_date']] = week_distribution_df['date'].apply(lambda x: pd.Series((x.start_time, x.end_time)))
    
        # format the end date column to remove the seconds
        week_distribution_df['end_date'] = week_distribution_df['end_date'].dt.strftime('%Y-%m-%d')
        week_distribution_df['end_date'] = pd.to_datetime(week_distribution_df['end_date'])
        
        week_distribution_df = week_distribution_df.drop(columns='date')
        week_distribution_df['platform'] = platform

        week_distribution_df['avg_rating_reviews'] = week_distribution_df['avg_rating_reviews'].round(2)
        week_distribution_df['cumulative_avg_general'] = week_distribution_df['cumulative_avg_general'].round(2)

        week_distribution_df['comments'] = None
        last_week = week_distribution_df['start_date'].max()
        first_week = week_distribution_df['start_date'].min()


        if last_week + Week(weekday=6) > df['date'].max():

            week_distribution_df.loc[week_distribution_df.index[-1], 'comments'] = 'incomplete week'
        
        if first_week < df['date'].min():

            week_distribution_df.loc[week_distribution_df.index[0], 'comments'] = 'incomplete week'


        if platform == 'android':

            week_distribution_df['weighted_avg'] = (week_distribution_df['cumulative_avg_general']*0.9 + week_distribution_df['avg_rating_reviews']*0.1).round(2)

            new_order = ['start_date', 'end_date', 'avg_rating_reviews','reviews', 'cumulative_avg_general', 'weighted_avg', 'total_count', 'platform', 'comments']

            week_distribution_df = week_distribution_df[new_order]

            distribution_dict[platform] = week_distribution_df
    
        else:

            new_order = ['start_date', 'end_date','avg_rating_reviews', 'reviews', 'cumulative_avg_general', 'total_count', 'platform', 'comments']

            week_distribution_df = week_distribution_df[new_order]

            distribution_dict[platform] = week_distribution_df
    
    return distribution_dict


###### MONTHLY DISTIBUTIONS #########

def get_monthly_reviews_distribution(df_reviews, rating_df):

    '''Input datasets from data.ai API: 
    - df_reviews: dataset with reviews and rating
    - general_df: daily distribution of rating (cumulative average)  
    - platform: ios or android
    return a dataset with daily statistics:
    - reviews: volume of reviews
    - avg_rating_reviews: average rating of reviews on volume
    - cumulative_avg_general: cumulative average rating over time on total_count
    - total_count: lifetime ratings app (reviews&ratings)'''

    all_reviews = {'android':df_reviews[df_reviews['device'] == 'android-all'].reset_index(drop=True),
                   'ios': df_reviews[df_reviews['device'] == 'ios-all'].reset_index(drop=True)}
    
    all_ratings = {'android': rating_df[rating_df['platform'] == 'android'].reset_index(drop=True),
                   'ios': rating_df[rating_df['platform'] == 'ios'].reset_index(drop=True)}
    
    distribution_dict = {}

    for platform, df in all_reviews.items():    

        monthly_rating_reviews = df.groupby(pd.Grouper(key='date', freq='M')).agg({'rating': 'mean',
                                                                                        'device': 'size'}).reset_index()
        monthly_rating_reviews['platform'] = platform
    
        monthly_distribution_cumulative_rating = all_ratings[platform].groupby(pd.Grouper(key='date', freq='M')).agg({'cumulative_avg_general':'mean',
                                                                                                        'total_count': 'last'}).reset_index()

        final_df =  pd.merge(monthly_distribution_cumulative_rating, monthly_rating_reviews, on='date', how='right')
        final_df = final_df.rename(columns={'rating': 'avg_rating_reviews', 'device': 'reviews'})

        final_df['avg_rating_reviews'] = final_df['avg_rating_reviews'].round(2)
        final_df['cumulative_avg_general'] = final_df['cumulative_avg_general'].round(2)
        final_df['date'] = final_df['date'].apply(lambda x: x.replace(day=1))


        final_df['comments'] = None
        last_day_df = df['date'].max()
        last_day_current_month = offsets.MonthEnd().rollforward(last_day_df)

        if last_day_current_month > last_day_df:

            final_df.loc[final_df.index[-1], 'comments'] = 'incomplete month'
    

        if platform == 'android':

            final_df['weighted_avg'] = (final_df['cumulative_avg_general']*0.9 + final_df['avg_rating_reviews']*0.1).round(2)

            new_order = ['date', 'avg_rating_reviews','reviews', 'cumulative_avg_general', 'weighted_avg', 'total_count', 'platform', 'comments']

            final_df = final_df[new_order]

            distribution_dict[platform] = final_df

        else:

            new_order = ['date','avg_rating_reviews','reviews','cumulative_avg_general', 'total_count', 'platform', 'comments']

            final_df = final_df[new_order]    

            distribution_dict[platform] = final_df
    
    return distribution_dict

##### VISUALIZATION from distribution datasets ########

def daily_comparison_rating(df_ios, df_android):

    '''Take two dataset already grouped by the daily distribution of reviews and ratings
    and return a scatterplot with 4 lines iOS vs Android, Cumulative vs Daily'''
    
    x = df_ios['date']

    cumulative_ios = df_ios['cumulative_avg_general']
    cumulative_android = df_android['weighted_avg']
    rating_ios = df_ios['avg_rating_reviews']
    rating_android = df_android['avg_rating_reviews']

    fig = go.Figure(data=[
        go.Scatter(name='iOS Rating<br>with written review', x=x, y=rating_ios,line=dict(color='blue'), customdata=df_ios['reviews'].apply(lambda x: '{:,}'.format(x)), legendrank=1, 
                   hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>'),

        go.Scatter(name='Android Rating<br>with written review',x=x, y=rating_android, line=dict(color='red'), customdata=df_android['reviews'].apply(lambda x: '{:,}'.format(x)), legendrank=2, 
                   hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>'),

        go.Scatter(name='Lifetime rating', x=x, y=cumulative_ios, mode='lines', line=dict(dash='dash', color='blue'), customdata=df_ios['total_count'].apply(lambda x: '{:,}'.format(x)), legendrank=1, 
                   hovertemplate='Avg on App Store: <b>%{y}</b><extra></extra><br>Total Rating: <b>%{customdata}</b>'),

        go.Scatter(name='Lifetime rating', x=x, y=cumulative_android, mode='lines', line=dict(dash='dash', color='red'),customdata=df_android['total_count'].apply(lambda x: '{:,}'.format(x)), legendrank=2, 
                   hovertemplate='Avg on Google Play: <b>%{y}</b><extra></extra><br>Total Rating: <b>%{customdata}</b>')]
                    )

    fig.update_xaxes(dtick='D1')
    fig.update_yaxes(range=[1,5], dtick=1)
    #fig.update_traces(hovertemplate="Avg: <b>%{y}</b><extra></extra>")

    fig.update_layout(title_text=f"Daily Distribution of Ratings - iOS vs Android")

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
   
    return graphJSON

def weekly_comparison_rating(df_ios, df_android):
    
    '''Take two dfs from iOS and Android with items group by week with a col "start_date" and a col "end_date" 
    and return a time series with weekly distribution'''

    fig = go.Figure()

    for df, color, name, platform, rank in zip([df_ios, df_android], ['blue', 'red'], ['iOS Rating<br>with written review', 'Android Rating<br>with written review'], ['iOS', 'Android'], [1, 2]):
    # Create a mask for the rows where 'comment' is not 'incomplete week'
        mask_complete_weeks = df['comments'] != 'incomplete week'
        
        # Plot the data for complete weeks
        fig.add_trace(go.Scatter(name=name, x=df.loc[mask_complete_weeks, 'start_date'], y=df.loc[mask_complete_weeks, 'avg_rating_reviews'], legendrank=rank,
                                    line=dict(color=color), legendgroup=platform, customdata=df.loc[mask_complete_weeks, 'reviews'].apply(lambda x: '{:,}'.format(x)),
                                    hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>'))

        # Plot the data for the first and last incomplete weeks
        if df.loc[df.index[0], 'comments'] == 'incomplete week':
            fig.add_trace(go.Scatter(name='incomplete week', x=df.loc[df.index[:2], 'start_date'], y=df.loc[df.index[:2], 'avg_rating_reviews'],
                                        line=dict(color=color, dash='dot'), customdata=df.loc[df.index[:2], 'reviews'].apply(lambda x: '{:,}'.format(x)),
                                     hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>', legendgroup=platform, showlegend=False))
            
        if df.loc[df.index[-1], 'comments'] == 'incomplete week':
            fig.add_trace(go.Scatter(name='incomplete week', x=df.loc[df.index[-2:], 'start_date'], y=df.loc[df.index[-2:], 'avg_rating_reviews'],
                                        line=dict(color=color, dash='dot'),customdata=df.loc[df.index[-2:], 'reviews'].apply(lambda x: '{:,}'.format(x)),
                                     hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>', legendgroup=platform, showlegend=False))
    


    # Add the cumulative average rating traces
    fig.add_trace(go.Scatter(name='Lifetime Rating', x=df_ios['start_date'], y=df_ios['cumulative_avg_general'], mode='lines', line=dict(dash='dash', color='blue'), legendrank=1,
                             customdata=df_ios['total_count'].apply(lambda x: '{:,}'.format(x)), hovertemplate='Avg on App Store: <b>%{y}</b><extra></extra><br>Total Rating: <b>%{customdata}</b>'))
    fig.add_trace(go.Scatter(name='Lifetime Rating', x=df_android['start_date'], y=df_android['weighted_avg'], mode='lines', line=dict(dash='dash', color='red'), legendrank=2,
                             customdata=df_android['total_count'].apply(lambda x: '{:,}'.format(x)), hovertemplate='Avg on Google Play: <b>%{y}</b><extra></extra><br>Total Rating: <b>%{customdata}</b>'))

    week_ranges = [f"{day.strftime('%b %d')} - {(day + timedelta(days=6)).strftime('%b %d')}" for day in df_ios['start_date']]

    fig.update_xaxes(dtick=86400000.0*7,
                     tickvals = df_ios['start_date'],  # Set tick values to the start of each week
                     ticktext = week_ranges,  # Set tick text to the corresponding week range
                     autorange=True,
                     fixedrange=False
    )

    fig.update_layout(title_text=f"Weekly Distribution of Ratings - iOS vs Android")
    

    fig.update_yaxes(range=[1,5], dtick=1)

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

    return graphJSON

def monthly_comparison_rating(df_ios, df_android):

    '''Take two dataset already grouped by the daily distribution of reviews and ratings
    and return a scatterplot with 4 lines iOS vs Android, Cumulative vs Daily'''
    
    fig = go.Figure()

    for df, color, name, platform, rank in zip([df_ios, df_android], ['blue', 'red'], ['iOS Rating<br>with written review', 'Android Rating<br>with written review'], ['iOS', 'Android'], [1, 2]):
    # Create a mask for the rows where 'comment' is not 'incomplete week'
        mask_complete_monthss = df['comments'] != 'incomplete month'


        fig.add_trace(go.Scatter(name=name, x=df.loc[mask_complete_monthss, 'date'], y=df.loc[mask_complete_monthss, 'avg_rating_reviews'], legendrank=rank,
                                    line=dict(color=color), legendgroup=platform, customdata=df.loc[mask_complete_monthss, 'reviews'].apply(lambda x: '{:,}'.format(x)),
                                    hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>'))
            
        if df.loc[df.index[-1], 'comments'] == 'incomplete month':
            fig.add_trace(go.Scatter(name='incomplete week', x=df.loc[df.index[-2:], 'date'], y=df.loc[df.index[-2:], 'avg_rating_reviews'],
                                        line=dict(color=color, dash='dot'),customdata=df.loc[df.index[-2:], 'reviews'].apply(lambda x: '{:,}'.format(x)),
                                     hovertemplate='Avg: <b>%{y}</b><extra></extra><br>Reviews:<b>%{customdata}</b>', legendgroup=platform, showlegend=False))
            
        # Add the cumulative average rating traces
    fig.add_trace(go.Scatter(name='Lifetime Rating', x=df_ios['date'], y=df_ios['cumulative_avg_general'], mode='lines', line=dict(dash='dash', color='blue'), legendrank=1,
                             customdata=df_ios['total_count'].apply(lambda x: '{:,}'.format(x)), hovertemplate='Avg on App Store: <b>%{y}</b><extra></extra><br>Total Rating: <b>%{customdata}</b>'))
    fig.add_trace(go.Scatter(name='Lifetime Rating', x=df_android['date'], y=df_android['weighted_avg'], mode='lines', line=dict(dash='dash', color='red'), legendrank=2,
                             customdata=df_android['total_count'].apply(lambda x: '{:,}'.format(x)), hovertemplate='Avg on Google Play: <b>%{y}</b><extra></extra><br>Total Rating: <b>%{customdata}</b>'))

                    
    fig.update_xaxes(dtick='M1')
    fig.update_yaxes(range=[1,5], dtick=1)

    fig.update_layout(title_text=f"Monthly Distribution of Ratings - iOS vs Android")

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    return graphJSON

######## REVIEWS DISTRIBUTION ##########

def daily_reviews_distribution(df_ios, df_android):

    '''Take two dfs from iOS and Android with items group by day with a col "End Date" 
    and return a time series with daily distribution'''
        
    x = df_ios['date']
    y_ios = df_ios['reviews']
    y_android = df_android['reviews']

    fig = go.Figure(data=[
        go.Scatter(name='iOS', x=x, y=y_ios),
        go.Scatter(name='Android',x=x, y=y_android)]
                    )

    fig.update_xaxes(dtick='D1')
    fig.update_traces(hovertemplate="Reviews: <b>%{y}</b><extra></extra>")

    fig.update_layout(title_text=f"Daily Distribution of Reviews - iOS vs Android")

    return fig.show(renderer='notebook')


def weekly_reviews_distribution(df_ios, df_android):

    '''Take two dfs from iOS and Android with items group by week with a col "Start Date" and a col "End Date" 
    and return a time series with weekly distribution'''

    fig = go.Figure()

    for df, color, name in zip([df_ios, df_android], ['blue', 'red'], ['iOS', 'Android']):
    # Create a mask for the rows where 'comment' is not 'incomplete week'
        mask_complete_weeks = df['comments'] != 'incomplete week'
        
        # Plot the data for complete weeks
        fig.add_trace(go.Scatter(name=name, x=df.loc[mask_complete_weeks, 'start_date'], y=df.loc[mask_complete_weeks, 'reviews'], legendgroup=name,
                                    line=dict(color=color), hovertemplate='Reviews:<b>%{y}<extra></extra></b>'))

        # Plot the data for the first and last incomplete weeks
        if df.loc[df.index[0], 'comments'] == 'incomplete week':
            fig.add_trace(go.Scatter(name='incomplete week', x=df.loc[df.index[:2], 'start_date'], y=df.loc[df.index[:2], 'reviews'],
                                        line=dict(color=color, dash='dot'),hovertemplate='Reviews:<b>%{y}<extra></extra></b>',legendgroup=name, showlegend=False))
            
        if df.loc[df.index[-1], 'comments'] == 'incomplete week':
            fig.add_trace(go.Scatter(name='incomplete week', x=df.loc[df.index[:2:-1], 'start_date'], y=df.loc[df.index[:2:-1], 'reviews'],
                                        line=dict(color=color, dash='dot'), hovertemplate='Reviews:<b>%{y}<extra></extra></b>', legendgroup=name, showlegend=False))

    #week_data = pd.merge(df_ios, df_android, on=start_date, how='outer').sort_values(by=start_date)
    week_ranges = [f"{day.strftime('%b %d')} - {(day + timedelta(days=6)).strftime('%b %d')}" for day in df_ios['start_date']]

    

    fig.update_xaxes(dtick=86400000.0*7,
                    tickvals = df_ios['start_date'],  # Set tick values to the start of each week
                    ticktext = week_ranges,  # Set tick text to the corresponding week range
                    autorange=True,
                    fixedrange=False
    )
    

    fig.update_layout(title_text=f"Weekly Distribution of Reviews - iOS vs Android")

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    return graphJSON

def monthly_reviews_distribution(df_ios, df_android):

    '''Take two dfs from iOS and Android with items group by day with a col "End Date" 
    and return a time series with daily distribution'''
        
    x = df_ios['date']
    y_ios = df_ios['reviews']
    y_android = df_android['reviews']

    fig = go.Figure(data=[
        go.Scatter(name='iOS', x=x, y=y_ios),
        go.Scatter(name='Android',x=x, y=y_android)]
                    )

    fig.update_xaxes(dtick='M1')
    fig.update_traces(hovertemplate="Reviews: <b>%{y}</b><extra></extra>")

    fig.update_layout(title_text=f"Monthly Distribution of Reviews - iOS vs Android")

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

    return graphJSON

