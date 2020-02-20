#########################################################################

######################## ETL Pipeline ##################################

#########################################################################

# Importing Libraries

import os
import pandas as pd
from sqlalchemy import create_engine

# Load datasets
messages = pd.read_csv('data/messages.csv')
categories = pd.read_csv('data/categories.csv')

# Viz datasets
print('Messages dataframe')
messages.head()
print('Categories dataframe')
categories.head()

#merge datasets
df = pd.merge(messages, categories, on='id', how='outer')

# create a dataframe of the 36 individual category columns
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
df.head()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save the clean dataset into an sqlite database
engine = create_engine('sqlite:///messages.db')
df.to_sql('messages', engine, index=False)