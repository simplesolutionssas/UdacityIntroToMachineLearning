#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
"""

import pickle
import numpy as np
import pandas as pd

# load the final_project_dataset.pkl file
file_path = '../final_project/final_project_dataset.pkl'
file = open(file_path, 'r')
enron_data = pickle.load(file)
file.close()

# construct and explore the enron_df DataFrame
enron_df = pd.DataFrame.from_dict(enron_data, orient='index')
# convert the 'Nan' strings into real NaN values
enron_df.replace('NaN', np.nan, regex=True, inplace=True)

# describe the DataFrame
print(enron_df.describe())

# see a sample of the DataFrame data
print(enron_df.head(10))

# ¿how many data points (i.e. people) are in the Enron dataset?
data_points = float(len(enron_df))
print('data points: {}'.format(data_points))

# ¿for each person, how many features are available?
print('number of features: {}'.format(len(enron_df.keys())))

# ¿how many POIs are there in the E+F dataset?
pois = enron_df.loc[enron_df.poi == True]
poi_count = float(len(pois))
print('pois in dataset: {}'.format(len(pois)))

# load and explore the poi_names.txt file
file_path = '../final_project/poi_names.txt'
file = open(file_path, 'r')
poi_names = 0
for line in file:
    line = line.strip('\n')
    if (len(line) > 0) and (line[0] == '('):
        poi_names += 1

file.close()

# ¿how many POI’s were there total? 
print('pois in poi_names: {}'.format(poi_names))

# ¿what is the total value of the stock belonging to James Prentice?
print('James Prentice\'s stock value: {}'
      .format(enron_df.loc['PRENTICE JAMES'].total_stock_value))

# ¿how many email messages do we have from Wesley Colwell to pois?
print('Wesley Colwell\'s emails to pois: {}'
      .format(enron_df.loc['COLWELL WESLEY'].from_this_person_to_poi))

# ¿what’s the value of stock options exercised by Jeffrey K Skilling?
print('Jeffrey K Skilling\'s stock options exercised: {}'
      .format(enron_df.loc['SKILLING JEFFREY K'].exercised_stock_options))

# ¿of these three individuals (Lay, Skilling and Fastow), who took home the 
# most money (largest value of “total_payments” feature)?
# ¿how much money did that person get?
print('Skilling\'s total payments: {}'
      .format(enron_df.filter(like='SKILLING', axis=0).total_payments[0]))
print('Lay\'s total payments: {}'
      .format(enron_df.filter(like='LAY', axis=0).total_payments[0]))
print('Fastow\'s total payments: {}'
      .format(enron_df.filter(like='FASTOW', axis=0).total_payments[0]))

# ¿how many folks in this dataset have a quantified salary? 
print('People with quantified salaries: {}'
      .format(enron_df.salary.count()))

# ¿how many folks in this dataset have a known email address?
print('People with know email address: {}'
      .format(enron_df.email_address.count()))

# ¿what percentage of people in the dataset have NaN for their total payments?
print('% of people with NaN for total payments: {:.2%}'
      .format((enron_df.total_payments.isnull().sum())/(data_points)))

# ¿what percentage of pois in the dataset have NaN for their total payments?
print('% of pois with NaN for total payments: {:.2%}'
      .format(pois.total_payments.isnull().sum()/poi_count))

# ¿what if we add 10 more pois?
print('% of people with NaN for total payments, with 10 more pois: {}'
      .format((enron_df.total_payments.isnull().sum() + 10)))
print('new poi count: {}'.format(poi_count + 10))
print('new poi with NaN for total payments: {}'
      .format(pois.total_payments.isnull().sum() + 10))
