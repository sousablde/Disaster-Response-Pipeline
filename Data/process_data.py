#########################################################################

######################## ETL Pipeline ##################################

#########################################################################

# Importing Libraries
import sys
import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(messages_filepath, categories_filepath):
    '''
    Loads and merges datasets
    Args:
    messages_filepath: String. Filepath for the csv file containing the messages.
    categories_filepath: String. Filepath for the csv file containing the categories.
    Returns:
    df: pandas dataframe. Dataframe containing messages and respective categories.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    print('Messages dataframe')
    messages.head()
    print('Categories dataframe')
    categories.head()
    
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df

def clean_data(df):  
    '''
    Clean dataframes from uneeded columns, duplicates and text artifacts
    Args:
    df: pandas dataframe. Dataframe containing messages and categories.
    Returns:
    df: pandas dataframe. Dataframe containing cleaned version of messages and categories.
    '''
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[[1]]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Based on the figure 8 documentation original mapping is the following: 1 - yes, 2 - no, so I will convert all the 2's to 0's
    categories['related'] = categories['related'].replace(2, 0)
    
    # The child alone column has a single label so that won't be helpful to train my model I will drop that column
    categories.drop("child_alone", axis=1, inplace=True)
    
    # Put processed columns back into the main df
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    #Checking for imbalances
    df1 = df.copy()
    df1.drop(['id', 'message', 'original'], axis=1, inplace=True)

    df1.shape

    rowCnt = 6
    colCnt = 6     # cols
    subCnt = 1     # initialize plot number

    fig = plt.figure(figsize=(20,30))

    for i in df1.columns:
        fig.add_subplot(rowCnt, colCnt, subCnt)

        plt.xlabel(i, fontsize=12)
        sns.countplot(df1[i])
        subCnt = subCnt + 1
    fig.subplots_adjust(top = 1, wspace = 1, hspace = 1)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned data.
    
    Input:
    df: pandas dataframe. Dataframe containing cleaned version of messages and respective categories.
    database_filename: String. Filename for the output database.
    
    Output:
    None.
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index = False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
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

