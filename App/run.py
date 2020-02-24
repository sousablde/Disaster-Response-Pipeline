import json
import plotly
import pandas as pd
import sys
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

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

# counted words feature
def get_counted_words():
    counted_words = np.load('data/counts.npz')
    return list(counted_words['top_words']), list(counted_words['top_counts'])


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', con=engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #category frequencies prep
    labels=df.iloc[:,4:].sum().sort_values(ascending=False).reset_index()
    labels.columns=['category','count']
    label_values=labels['count'].values.tolist()
    label_names=labels['category'].values.tolist()
    
    #category top 10 prep
    category_counts = df.iloc[:,4:].sum(axis = 0).sort_values(ascending = False)
    category_top = category_counts.head(10)
    category_names = list(category_top.index)
    
    #top words
    word_srs = pd.Series(' '.join(df['message']).lower().split())
    top_words = word_srs[~word_srs.isin(stopwords.words("english"))].value_counts()[:10]
    top_words_names = list(top_words.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_values,
                )
            ],

            'layout': {
                'title': "Messages categories frequency",
                'yaxis': {
                    'title':"Message Category Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_top
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words_names,
                    y=top_words
                )
            ],

            'layout': {
                'title': 'Most Frequent Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
        
        
        
        
    ]
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()