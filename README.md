# WallStreetBets-Sentiment

<b> Data Directory </b>

Code Files

All related data referenced is available in the *Data* folder.

0. Generate Raw Data
    - *Pulling_Reddit_Posts.ipynb*: Pulling posts associated to 9 tickers to create *combined_posts.csv*
    - *Pull_and_Combine_Reddit_Comments.ipynb*: Pulling comments associated with posts pulled to create *combined_comments_untagged.csv*. After manual tagging, NB is used to merge tagged data to form *comments_and_tags.csv*.

1. Cleanup Text
    - *Assign Comments to Tickers.ipynb*: Input *comments_preprocessed.csv* to output *combined_comments_assigned.csv* with ticker assignment for each comment.
    - *TextPreprocessing.ipynb*: Input *combined_comments_untagged.csv *to output *comments_preprocessed.csv*. NB contains preliminary EDA as well for labelled comments. 
2. Sentiment Prediction

    All files below use manually tagged data of 5500 comments (*comments_and_tags.csv*).
    - *BERT.ipynb*: BERT sentiment analysis.
    - *VADER.ipynb*: VADAR sentiment analysis.
    - *Classical models.ipynb*: Various classifiers for sentiment analysis.
    - *Classical models with PCA.ipynb*: Various classifiers for sentiment analysis with PCA.
3. Stock Price Regression

    - *LSTM By Stock*: ipynb notebooks for each of the 9 selected tickers
    - *LSTM CSV Results by Stock*: MAPE, MDAPE and MAE results for each LSTM over 1, 4 and 7 day lags
    - *prediction_prep.py*: Script to combine comments data (*comments_preprocessed_assigned.csv*), yfinance data (*TICKER_2021.csv*) and sentiment data (*df_pred_id_body.csv*)
    - *LSTM Final Weights*: File is sizeable and cannot be uploaded to Git. Access via this link: https://drive.google.com/drive/folders/1FudwuUJBcG4BGKAqV30uYE4hr8caS1nN?usp=sharing.
