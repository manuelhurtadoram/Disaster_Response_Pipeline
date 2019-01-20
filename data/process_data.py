#Import libraries
import sys

# Data handling
import numpy as np
import pandas as pd

# SQL database
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from .csv files for both messages (text) and categories to which
    they belong (labels).
    
    Args:
        - messages_filepath: .csv file pointing to message data
        - categories_filepath: .csv file pointing to categories data
        
    Returns:
        - df: Pandas DataFrame of merged (raw) message and category data
    '''
    
    # Load messages
    messages = pd.read_csv(messages_filepath, sep=',',
                       names=['ID', 'Message', 'Original', 'Genre'],
                       skiprows=1)
    
    # Load categories
    categories = pd.read_csv(categories_filepath, sep=',',
                         names=['ID', 'Categories'],
                         skiprows=1)
    
    # Merge
    df = messages.merge(categories, how='inner', on='ID', sort=False)
    
    return df


def clean_data(df):
    '''
    Cleans raw data held in DataFrame by:
    
        1. Extracting category names (labels)
        2. Extracting category values for each sample
        3. Renaming category columns for category data
        4. Converting category values from string to 0 or 1 for each category
        5. Creating new DataFrame with processed data
        6. Removing duplicates and NaNs
        
    Args:
        - df: DataFrame with raw message and category data
    
    Returns:
        - df: Clean, processed DataFrame with samples and category values as 0 or 1
    '''
    
    # Create DataFrame of the individual category columns
    categories = df['Categories'].str.split(pat=';', n=-1, expand=True)
    
    # Select first row of DataFrame
    row = categories.iloc[0, :]

    # Slice each string to remove the last dash and number
    category_colnames = [element[:-2] for element in row]
    
    print('The category column names are:\n', category_colnames)
    
    # Rename columns of 'categories' DF
    categories.columns = category_colnames
    
    # Iterate through the category columns to extract values
    # of 0 or 1 at the end of each string
    for col in categories:
        categories[col] = categories[col].apply(lambda x: str(x[-1])).astype('int')
            
    # 'related' category currently has values of 1.0 for messages that are relevant,
    # and 2.0 for messages that are irrelevant (no related categories), so
    # we need to map values of 2.0 for 'related' to 0.0 to keep the matrix binary
    categories['related'] = categories['related'].map({0:0, 1:1, 2:0})
        
    # Add new column reflecting unrelated messages
    categories['unrelated'] = categories['related'].apply(lambda x: 1 if x is 0 else 0)
    
    # Drop original categories column
    df = df.drop(['Categories'], axis=1, inplace=False)
    
    # Concatenate original dataframe with new 'categories' DF
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop NaNs from all columns except 'Original', because 
    # it has 16k NaNs for messages that were originally in
    # English, and therefore did not have to be traduced
    to_drop = np.delete(df.columns.values, 
                        np.argwhere(df.columns.values == 'Original'))
    df = df.dropna(how='any', subset=to_drop, axis=0)
    
    return df
    
    
def save_data(df, database_filename):
    '''
    Saves data to local SQLite database using a connection through an
    SQLAlchemy engine.
    
    Args:
        - df: DataFrame containing clean data.
        - database_filename: .db path to local database
    
    Returns:
        - None
    '''
    
    # Create engine to access database
    engine = create_engine('sqlite:///' + database_filename)
    
    # Connect to database and upload df data
    df.to_sql(name='CleanMessageData', con=engine, if_exists='replace', index=False)
        

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'\
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()