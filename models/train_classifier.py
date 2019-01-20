import sys

# Data handling
import pandas as pd
import numpy as np

# String processing
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Database handling
from sqlalchemy import create_engine

# Model design, training, and testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Saving model
import joblib

def load_data(database_filepath):
    '''
    Loads clean message data from SQLite database,
    and breaks it down into feature (X) and target (y)
    variables.
    
    Args:
        - database_filepath: Path to local SQLite database 
        
    Returns:
        - X: Message data from samples
        - y: Category data from samples
        - category_names: Labels for each of the categories
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CleanMessageData', con=engine)
    
    # Break data into feature and target sets
    X = df[['Message']].values.flatten()
    y = df.drop(['ID', 'Message', 'Genre', 'Original'], axis=1).values
    
    # Category names
    category_names = df.drop(['ID', 'Message', 'Genre', 'Original'], axis=1).columns.values
    
    return X, y, category_names


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


def build_model():
    '''
    Builds MultiOutputClassifier with RandomForestClassifier as estimator.
    A pipeline is used to streamline two feature extraction processes:
        
        1. Bag Of Words (CountVectorizer)
        2. Tfidf Transformation
        
    The method creates a GridSearchCV object to train multiple parameters of
    the pipeline, and returns this object.
        
    Args: (None)
        
    Returns:
        - cv: GridSearchCV object of the pipeline
    '''
    
    # Make pipeline with feature transformations and estimator
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    # Set parameters to tune
    params = {
        'clf__estimator__criterion':['gini', 'entropy'],
        'clf__estimator__warm_start':[True, False],
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [0.2, 0.3],
        'vect__ngram_range':[(1, 1), (1, 2)],
    }

    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=params, verbose=2)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluates model performance by computing classification reports
    for each of the category labels.

    Args:
        - model: Trained pipeline with MultiOutputClassifier of 
                 RandomForestClassifier estimators
                 (fine-tuned through GridSearchCV)
        - X_test: Test sample features
        - y_test: Test sample targets
        - category_names: Labels of categories
           
    Returns: (None)
    '''
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Iterate over each label to print classification report
    for i, category in enumerate(category_names):
        print('\n\nClassification Report:', category.upper())
        print(classification_report(y_test[:, i], y_pred[:, i],
                                    target_names=['Is', 'Is Not']))


def save_model(model, model_filepath):
    '''
    Saves model to file using JobLib (preferred instead of pickle
    because:
    
        "In the specific case of scikit-learn, it may be better 
        to use joblib’s replacement of pickle (dump & load), which 
        is more efficient on objects that carry large numpy arrays 
        internally as is often the case for fitted scikit-learn estimators." 
        
        (Sklearn Documentation - for more information, refer to:
         https://scikit-learn.org/stable/modules/model_persistence.html)
    
    Args:
        - model
        - model_filepath
    
    Returns: (None)
    '''
    
    # Save model using joblib
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
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