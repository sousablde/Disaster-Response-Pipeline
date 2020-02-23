# import libraries
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download(['punkt', 'wordnet'])

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler


def load_data(database_filepath):
    """
    Load the data
    
    Inputs:
    database_filepath: String. Filepath for the db file containing the cleaned data.
    
    Output:
    X: dataframe. Contains the feature data.
    y: dataframe. Contains the labels (categories) data.
    category_names: List of strings. Contains the labels names.
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = y.columns.tolist()
    
    return X, y, category_names

def tokenize(text):
    """
    Normalize, tokenize and stems texts.
    
    Input:
    text: string. Sentence containing a message.
    
    Output:
    stemmed_tokens: list of strings. A list of strings containing normalized and stemmed tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text) 
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Creates feature based on verb presence in text
    """
    def start_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        X_tag = pd.Series(X).apply(self.start_verb)
        return pd.DataFrame(X_tag)
    
def get_text_len(data):
    """ 
    Gets the text length.
    """
    
    return np.array([len(text) for text in data]).reshape(-1, 1)


def build_model():
    """
    Builds a ML pipeline and performs gridsearch.
    Args:
    None
    Returns:
    cv: gridsearchcv object.
    """
    
    # Creates pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('best', TruncatedSVD()),
                ('tfidf', TfidfTransformer())])), 
        ('start_verb', StartVerbExtractor())])), 
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Parameters
    parameters = {'features__text_pipeline__tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [100, 200, 300], 
              'clf__estimator__random_state': [42],
             'clf__estimator__learning_rate': [0.05]} 

    
    # Creates gridsearch
    cv = GridSearchCV(pipeline, param_grid = parameters, cv = 10, refit = True, verbose = 1, return_train_score = True, n_jobs = -1)
    
    return cv
    

def evaluate_model(model, X_test, y_test, category_names):
    """
    Returns test accuracy, number of 1s and 0s, recall, precision and F1 Score.
    
    Inputs:
    model: model object. Instanciated model.
    X_test: pandas dataframe containing test features.
    y_test: pandas dataframe containing test labels.
    category_names: list of strings containing category names.
    
    Returns:
    None
    """
    
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = category_names)
    
    
    print(classification_report(y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        
        mlss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)

        for train_index, test_index in mlss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.values[train_index], y.values[test_index]
        y_train = pd.DataFrame(y_train,columns=category_names)
        y_test = pd.DataFrame(y_test,columns=category_names)
                
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()