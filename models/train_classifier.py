import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    # creat an sqlite connection and loads the database
    engine = create_engine('sqlite:///'+database_filepath)
    
    # convert the table in the database to the dataframe
    df = pd.read_sql_table(con=engine, table_name='df')
    
    # split the data into dependent and independent variables
    X = df['message']
    Y = df.drop(['id', 'message','original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    # tokenize and clean the text messages
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build a pipeline model that consists of text processing and multiple output classifier
    pipeline = Pipeline([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    # specify the paramaters for grid search
    parameters = {
    'text_pipeline__vect__max_df': (0.5, 1.0),
    'text_pipeline__tfidf__use_idf': (True, False),

    }
    
    # create grid search model
    cv = GridSearchCV(pipeline, parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    # predict the categories
    Y_pred = model.predict(X_test)
    
    # print the classification report for each category
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[category], Y_pred[:,i]))

def save_model(model, model_filepath):
    # save the model into a pickle model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
