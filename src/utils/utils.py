import os
import pandas as pd
import numpy as np
import math
import sys
import time
import datetime
import pickle
import json
import random
import sklearn



def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']

    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    #  Note: -1 are missing values for real world data 
    df_final=df_final.replace([-1],0)
    
    return df_final


def get_train_test_set(df, start_date_training, delta_train = 7, delta_delay = 7, delta_test = 7):

    # Get training set
    train_df = df[(df['TX_DATETIME'] >= start_date_training) & \
        (df['TX_DATETIME'] < start_date_training + datetime.timedelta(days=delta_train))]

    # Get test set
    test_df = []
    # Remove known compromised accounts from test set
    known_compromised_accounts = set(train_df[train_df['TX_FRAUD'] == 1]['CUSTOMER_ID'])
    start_tx_time_days_training = train_df['TX_TIME_DAYS'].min()

    for day in range(delta_test):

        test_df_day = df[df['TX_TIME_DAYS'] == start_tx_time_days_training + delta_train + delta_delay + day]

        # Remove known compromised accounts from test day minus the delay period are added to known compromised accounts
        test_df_delay = df[df['TX_TIME_DAYS'] == start_tx_time_days_training + delta_train + day - 1]

        new_compromised_accounts = set(test_df_delay[test_df_delay['TX_FRAUD'] == 1]['CUSTOMER_ID'])
        known_compromised_accounts = known_compromised_accounts.union(new_compromised_accounts)

        test_df_day = test_df_day[~test_df_day['CUSTOMER_ID'].isin(known_compromised_accounts)]
        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')

    return train_df, test_df

def fit_model_and_predict(classifier, train_df, test_df, features, target, scale=True):

    # Scale features
    if scale:
        train_scaler = sklearn.preprocessing.StandardScaler()
        train_scaler.fit(train_df[features])
        train_df[features] = train_scaler.transform(train_df[features])
        test_df[features] = train_scaler.transform(test_df[features])

    # Fit model and predict
    start_time =  time.time()
    classifier.fit(train_df[features], train_df[target])
    training_time = time.time() - start_time

    start_time = time.time()
    predictions_test = classifier.predict_proba(test_df[features])[:, 1]
    prediction_time = time.time() - start_time

    predictions_train = classifier.predict_proba(train_df[features])[:, 1]

    results = {
        "classifier": classifier,
        "predictions_train": predictions_train,
        "predictions_test": predictions_test,
        "training_time": training_time,
        "prediction_time": prediction_time
    }

    return results

def card_precision_top_k_day(df_day, top_k):

    # Group by CUSTOMER_ID and keep the maximum prediction for each customer
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values('predictions', ascending=False).reset_index(drop=False)

    # Get the top k customers
    top_k_customers = df_day.head(top_k)['CUSTOMER_ID'].tolist()
    list_compromised_customers = list(top_k_customers[top_k_customers['TX_FRAUD'] == 1]['CUSTOMER_ID'])

    # Calculate the card precision
    card_precision = len(list_compromised_customers) / top_k

    return list_compromised_customers, card_precision

def card_precision_top_k(df, top_k, remove_detected_customers=True):

    list_days = list(df['TX_TIME_DAYS'].unique())
    list_days.sort(ascending=True)

    list_compromised_customers = []
    list_card_precision_per_day = []
    num_compromised_customers_per_day = []

    # Compute card precision for each day
    for day in list_days:

        df_day = df[df['TX_TIME_DAYS'] == day]
        
        df_day = df_day[df_day['CUSTOMER_ID'].isin(list_compromised_customers) == False]

        num_compromised_customers_per_day.append(len(df_day[df_day['TX_FRAUD'] == 1]['CUSTOMER_ID'].unique()))

        detected_customers, card_precision = card_precision_top_k_day(df_day, top_k)

        list_card_precision_per_day.append(card_precision)

        if remove_detected_customers:
            list_compromised_customers.extend(detected_customers)

    # Compute mean card precision
    mean_card_precision = np.mean(list_card_precision_per_day)

    return num_compromised_customers_per_day, list_card_precision_per_day, mean_card_precision

def performance_assessment(predictions_df, output_feature='TX_FRAUD', 
                           prediction_feature='predictions', top_k_list=[100],
                           rounded=True):
    
    AUC_ROC = sklearn.metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = sklearn.metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
    
    performances = pd.DataFrame([[AUC_ROC, AP]], 
                           columns=['AUC ROC','Average precision'])
    
    for top_k in top_k_list:
    
        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
        performances['Card Precision@'+str(top_k)]=mean_card_precision_top_k
        
    if rounded:
        performances = performances.round(3)
    
    return performances

# def prequential_split(df, start_date_training, delta_train, delta_delay, delta_test, n_folds = 4):

#     prequential_splits_indices = []

#     for i in range(n_folds):
    
#     train_df, test_df = get_train_test_set(df, start_date_training, delta_train, delta_delay, delta_test)
#     prequential_splits_indices.append((train_df, test_df))

#     return prequential_splits_indices