#!/usr/bin/env ipython

import pandas as pd
import numpy as np
import seaborn as sns
import ipdb
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def ez_dog(classifier='LR'):
    df = pd.read_csv('Hutson Jump 2019 - Sheet1.csv')
    df = clean_data(df)
    features, feature_names, labels = encode_data(df)

    X_train, X_test, y_train, y_test = build_training_data(features, labels)
    clf = fit_model(X_train, y_train, classifier)
    evaluate(clf, X_test, y_test)
    if classifier == 'LR':
        nice_thingy = list(zip(feature_names, clf.coef_[0]))
        for yoke in nice_thingy:
            print(yoke)
    return clf, feature_names

def clean_data(df):
    """
    """
    # delete everything after row 66 or whatever
    df = df.iloc[:65, :]
    # only keep first 10 columns
    df = df.iloc[:, :10]

    # strip whitespace at ends
    for column in ['Player1', 'Player2', 'Stage', 'char1', 'char2']:
        df[column] = [x.strip(' ') for x in df[column].astype('str')]
    
    return df

def encode_data(df):
    """
    Everything that is categorical must be made one-hot
    """
    # pit players and characters against each other
    list_of_players = pd.concat([df['Player1'], df['Player2']]).unique()
    list_of_characters = pd.concat([df['char1'], df['char2']]).unique()
    
    categorical_columns = ['Player1', 'Player2', 'Stage', 'char1', 'char2']
    df = pd.get_dummies(df, columns=categorical_columns)
    
    for player in list_of_players:
        df['Player_' + player] = df['Player1_' + player].astype(int) - df['Player2_' + player].astype(int)
        del df['Player1_' + player]
        del df['Player2_' + player]

    for char in list_of_characters:
        if not 'char1_' + char in df.columns:
            df['char1_' + char] = 0
        if not 'char2_' + char in df.columns:
            df['char2_' + char] = 0
        df['char_' + char] = df['char1_' + char].astype(int) - df['char2_' + char].astype(int)

        del df['char1_' + char]
        del df['char2_' + char]

    labels = df['player1 won'].astype(int).values
    forbidden_columns = ['player1 won', 'Player1 Stock', 'Player2 stock']
    features = df.drop(columns=forbidden_columns)
    feature_names = features.columns
    features = features.values

    return features, feature_names, labels

def build_training_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    return X_train, X_test, y_train, y_test

def fit_model(X_train, y_train, classifier='LR'):
    if classifier == 'RF':
        clf = RandomForestClassifier()
    else:
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test):
    print(clf.score(X_test, y_test))
    return True

def predict_new_game(clf, player1, player2, player1char, player2char, stage, feature_names, player1last=0, player2last=0):
    featurevec = np.zeros([1,len(feature_names)])
    # set the player bits
    featurevec[0,feature_names.index('Player_' + player1)] +=  1
    featurevec[0,feature_names.index('Player_' + player2)] +=  -1
    # set the character bits
    featurevec[0,feature_names.index('char_' + player1char)] +=  1
    featurevec[0,feature_names.index('char_' + player2char)] +=  -1
    # set the stage
    featurevec[0,feature_names.index('Stage_' + stage)] =  1
    # set the previous player losses
    featurevec[0,feature_names.index('Player1 lost last game')] = player1last
    featurevec[0,feature_names.index('player2 lost last game')] = player2last

    print(clf.predict_proba(featurevec)) 
    return featurevec
    
