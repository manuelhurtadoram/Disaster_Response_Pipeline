import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter

app = Flask(__name__)

def tokenize(text):
    '''
    Method to tokenize each of the messages (text).
    
    Args:
        - text
    
    Returns:
        - tokens
    '''

    # Convert to lowercase, strip, and replace punctuation with spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()   
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    tokens = [lemmatizer.lemmatize(tok, pos='v') for tok in tokens]
    
    # Stem
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(tok) for tok in tokens]
    
    return tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponseData.db')
df = pd.read_sql_table('CleanMessageData', engine)

# load model
model = joblib.load("../models/trained_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # First Graph:
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Second Graph:
    labels = df.columns[4:-1].values
    label_totals = []
    for label in labels:
        label_totals.append(np.sum(df[col]))
    label_percentages = 100.0 * label_totals / label_totals.sum()
    
    # Third Graph:
    num_categories = []
    for i, row in df.iterrows():
        num_categories.append(np.sum(row[4:-1]))
    counter = Counter(num_categories)
    keys = counter.keys()
    frequencies = counter.values()
    
    # create visuals
    graphs = [
        { # First Graph
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
        
        { # Second Graph
            'data': [
                Pie(
                    labels=labels,
                    values=label_percentages,
                    textinfo='label',
                    hoverinfo='percent',
                    hole=0.3,
                )   
            ],
            
            'layout': {
                'title': 'Proportion of Message Categories'
            }
        },
        
        { # Third Graph
            'data': [
                Histogram(
                    x=num_categories,
                    hoverinfo='y',
                    cumulative
                    
                )
            ],
            
            'layout': {
                'title': 'Distribution of Category Matches',
                'xaxis': {
                    'title': 'Number of Category Matches',
                },
            },
                
            'marker': {
                'color':'rgb(107, 107, 107)',
                
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