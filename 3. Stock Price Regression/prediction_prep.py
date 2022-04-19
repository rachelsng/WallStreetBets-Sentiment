import pandas as pd
import numpy as np
import datetime

def get_regression_df(ticker_name, comments, fin_data, sentiment):
    
    # Drop NA from comments
    comments = comments[~comments['created_utc'].isna()]
    comments = comments[comments['body']!='removed']
    #print(len(comments))
    comments = comments.drop_duplicates(subset = ['id'])
    #print(len(comments))
    ##### JOIN SENTIMENT DATA ######
    sentiment = sentiment[sentiment['body']!='removed']
    sentiment = sentiment[['id', 'predicted sentiment']]
    comments = comments.merge(sentiment, how = 'left', on = 'id' )
    comments = comments.dropna(subset = ['predicted sentiment'])
    #print(len(comments))

    # Grab only the ticker assignment and the date
    comments['created_utc'] = pd.to_numeric(comments['created_utc'], errors = 'coerce')
    comments = comments[~comments['created_utc'].isna()]
    comments['created_utc'] = comments['created_utc'].astype(int)
    comments['created_utc'] = pd.to_datetime(comments['created_utc'],unit= 's')
    comments['date'] = comments['created_utc'].dt.date
    comments['date'] = pd.DatetimeIndex(comments['date'])
    #print(len(comments))

    # Only one line of weird values, drop
    comments=comments[~(comments['date'] < pd.Timestamp('2021-01-01'))]
    #print(len(comments))

    # Create grouped DF
    grouped = comments.groupby(['date', 'ticker', 'predicted sentiment'])[['index']].count().reset_index()
    grouped = grouped[grouped['ticker'] == ticker_name]
    grouped['predicted sentiment'] = grouped['predicted sentiment'].astype(int)
    grouped = grouped.pivot(index='date', columns='predicted sentiment', values='index').reset_index()
    grouped = grouped.fillna(0)
    grouped.rename(columns={0: 'comments_neutral',
                        1:'comments_positive'}, inplace = True)
    if 'comments_positive' not in grouped.columns:
        grouped['comments_positive'] = 0
    if 'comments_neutral' not in grouped.columns:
        grouped['comments_neutral'] = 0
    grouped['comment_count'] = grouped['comments_neutral'] +grouped['comments_positive']
    grouped['pct_pos_comments'] = (grouped['comments_positive'] / (grouped['comment_count']+0.0001)) *100

    ##### BUILD EMPTY DATE DF ######
    # Build an empty df with date only
    start_date = '2021-01-01'
    end_date = '2022-01-01'

    # Generate a list of month-end closes
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = pd.date_range(start, end, freq='D')

    # Format date columns 
    df = pd.DataFrame(date_generated)
    df.columns = ['date'] # rename column
    df = df.merge(grouped, how = 'left', on = 'date')
    df = df.fillna(0)
    
    ##### GET MEME PERIOD ######
    df['7_day_avg'] = df['comment_count'].rolling(7).mean() 
    df['30_day_avg'] = df['comment_count'].rolling(30).mean() 
    df['is_meme'] = np.where((df['comment_count'] > df['30_day_avg']*2)
                                 & (df['comment_count']>df['comment_count'].quantile(0.90))
                                 , 1 ,0 )
    
    # Formatting
    df = df.drop(['7_day_avg', '30_day_avg'], axis = 1)
    
    #### JOIN POST DATA ######
    # rolling comment count or percentage change is not useful
    #for i in [3, 7]:
    for i in range(2,8):
        for metric in ['comments_neutral','comments_positive','comment_count', 'pct_pos_comments']:
            if i > 1:
                roll_avg = df[metric].rolling(i).sum()
                df= df.join(roll_avg, rsuffix = '_rollsum_'+str(i)+'D')
                mean = df[metric].rolling(i).mean()
                df = df.join(mean, rsuffix = '_rollmean_'+str(i)+'D')
            pct_change = df[metric].pct_change(periods=i)*100
            df = df.join(pct_change, rsuffix= '_'+str(i)+'D_pctchg')
    df.replace([np.inf, -np.inf],100, inplace=True)
    
    # #EWM IS NOT USEFUL
    # df['comment_count_ewm'] = df['comment_count'].ewm(span=3).sum()
    # df['comment_count_mean_ewm'] = df['comment_count'].ewm(span=3).mean()
    # df['is_meme_ewm'] = df['is_meme'].ewm(span=3).mean()
    
    
    ##### JOIN FINANCIAL DATA ######
    fin_data['date'] = pd.DatetimeIndex(fin_data['date'])
    fin_data = fin_data.drop(['Unnamed: 0', 'TICKER'], axis = 1)
    df = df.merge(fin_data, how = 'left', on = 'date')
    df['day_of_week'] = df['date'].dt.dayofweek # create day-of-week 0 = monday
    
    df.rename(columns={'Close_x': 'Close',
                       'Open_x': 'Open',
                       'High_x': 'High',
                       'Low_x': 'Low',
                      'Volume_x': 'Volume'}, inplace = True)
    
    ##### ADDITIONAL FEATUERS ######
    df['SD_log'] = np.log(df['SD'])
    #df['Volume_log'] = np.log(df['Volume'])
    
    return df

def lag_variables(data, vars_to_lag, shift_by):
    df = data.copy()
    for variable in vars_to_lag:
        df['{}(t-{})'.format(variable,shift_by)] = df[variable].shift(periods = shift_by)
        # Drop the original column once done
        df = df.drop(variable, axis = 1)
    return df

def lag_pred_df(df, dep_var, other_var, fin_vars, sentiment_vars, shift):
    pred_df = df[dep_var+other_var+fin_vars+sentiment_vars]

    # Remove saturday and sunday --> assume that comment changes over the weekend will affect Monday
    # Lag sentiment first, in place
    pred_lag = lag_variables(data = pred_df, 
                             vars_to_lag = sentiment_vars, 
                             shift_by = shift)

    # Drop weekends, dependent variable is only present for trading days 
    print('Length of dataframe before dropping weekends: {}'.format(len(pred_lag)))
    pred_lag = pred_lag.dropna(subset=dep_var).reset_index(drop=True)
    print('Length of dataframe after dropping weekends: {}'.format(len(pred_lag)))

    # Lag finance variables in place
    pred_lag = lag_variables(data = pred_lag, 
                             vars_to_lag = fin_vars, 
                             shift_by = shift)

    # Lag dependent variable
    dep_keep = pred_lag[dep_var]
    pred_lag = lag_variables(data = pred_lag, 
                             vars_to_lag = dep_var, 
                             shift_by = shift)
    pred_lag = pd.concat([dep_keep, pred_lag], axis = 1)
    
    return pred_lag