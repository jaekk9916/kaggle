# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:25:58 2024

@author: Jaekyeong
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('nyc-rolling-sales.csv')

print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.dtypes)
print(df.nunique())

# Convert 'SALE PRICE' and 'YEAR BUILT' to numeric
df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['YEAR BUILT'] = pd.to_numeric(df['YEAR BUILT'], errors='coerce')
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')

df_filtered = df[(df['SALE PRICE'] >= 1000) & 
                 (df['YEAR BUILT'] >= 1875) &
                 (df['GROSS SQUARE FEET'] > 0) &
                 (df['LAND SQUARE FEET'] > 0)]


df_filtered = df_filtered.copy()
print('df filtered: ', len(df_filtered))
# Mapping borough numbers to names
borough_mapping = {1:'Manhattan', 2:'Bronx', 3:'Brooklyn', 4:'Queens', 5:'Saten Island'}
df_filtered['BOROUGH'] = df_filtered['BOROUGH'].map(borough_mapping)

sns.set(style='whitegrid')

plt.figure(figsize=(10,6))
borough_counts = df_filtered['BOROUGH'].value_counts()
sns.barplot(x=borough_counts.index, y=borough_counts.values, palette='viridis')
plt.title('Number of Data Points by BOROUGH')
plt.xlabel('BOROUGH')
plt.ylabel('Number of Data points')
plt.show()

# Revert the BOROUGH column back to numeric values
borough_reverse_mapping = {'Manhattan': 1, 'Bronx': 2, 'Brooklyn': 3, 'Queens': 4, 'Saten Island': 5}
df_filtered['BOROUGH'] = df_filtered['BOROUGH'].replace(borough_reverse_mapping)



df_filtered['SALE DATE'] = pd.to_datetime(df_filtered['SALE DATE'], errors='coerce')


year_built_trend = df_filtered['YEAR BUILT'].value_counts().reset_index(name='counts')
plt.figure(figsize=(12,6))
sns.lineplot(data=year_built_trend, x='YEAR BUILT', y='counts')
plt.title('Trend in Building Constructed Over Time')
plt.xlabel('Year Built')
plt.ylabel('Number of buildings Constructed')
plt.grid(True)
plt.show()

label_encoder = LabelEncoder()

# Applying Label Encoding to categorical columns
label_encoder_neighborhood = LabelEncoder()
df_filtered['NEIGHBORHOOD'] = label_encoder_neighborhood.fit_transform(df_filtered['NEIGHBORHOOD'])
label_encoder_building_class = LabelEncoder()
df_filtered['BUILDING CLASS CATEGORY'] = label_encoder_building_class.fit_transform(df_filtered['BUILDING CLASS CATEGORY'])
label_encoder_tax_class = LabelEncoder()
df_filtered['TAX CLASS AT PRESENT'] = label_encoder_tax_class.fit_transform(df_filtered['TAX CLASS AT PRESENT'])
label_encoder_building_at_present = LabelEncoder()
df_filtered['BUILDING CLASS AT PRESENT'] = label_encoder_building_at_present.fit_transform(df_filtered['BUILDING CLASS AT PRESENT'])
label_encoder_at_time_sale = LabelEncoder()
df_filtered['BUILDING CLASS AT TIME OF SALE'] = label_encoder_at_time_sale.fit_transform(df_filtered['BUILDING CLASS AT TIME OF SALE'])

df_filtered['SALE PRICE'] = pd.to_numeric(df_filtered['SALE PRICE'], errors='coerce')
df_filtered['LAND SQUARE FEET'] = pd.to_numeric(df_filtered['LAND SQUARE FEET'], errors='coerce')
df_filtered['GROSS SQUARE FEET'] = pd.to_numeric(df_filtered['GROSS SQUARE FEET'], errors='coerce')

df_filtered = df_filtered.drop(columns=['Unnamed: 0','EASE-MENT','ADDRESS','APARTMENT NUMBER'])

print(df_filtered.dtypes)

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_filtered.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Create a new column for property age
df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])
df['PROPERTY AGE'] = df['SALE DATE'].dt.year - df['YEAR BUILT']


# Drop rows with missing or nonsensical values
df = df.dropna(subset=['PROPERTY AGE', 'SALE PRICE'])
df = df[df['PROPERTY AGE'] > 0]

plt.figure(figsize=(10,8))
sns.heatmap(df_filtered.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.show()

# Focusing on the correlation with the target variable (e.g., 'SALE PRICE')
correlation_with_target = df_filtered.corr()['SALE PRICE'].sort_values(ascending=False)
print(correlation_with_target)

plt.figure(figsize=(8,6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette='viridis')
plt.title('Correlationship of Feature with SALE PRICE')
plt.xlabel('Features')
plt.ylabel('Correlationship with SALE PRICE')
plt.xticks(rotation=90)
plt.show()

# Capture the original mapping of encoded values
tax_class_mapping = dict(zip(label_encoder_tax_class.transform(label_encoder_tax_class.classes_), label_encoder_tax_class.classes_))
print("Encoded to Origianl Mapping: ", tax_class_mapping)

# Filter the dataset to include only Tax Class 1 properties and its variations (e.g., '1','1A','1B','1C')
df_residential = df_filtered[df_filtered['TAX CLASS AT PRESENT'].astype(str).str.startswith('1')]

# Check the unique encoded values for TAX CLASS AT PRESENT after filtering and encoding
print(df_residential['TAX CLASS AT PRESENT'].unique())
df_residential = df_residential.copy()

# Analyze the price trende of residential homes over time
df_residential['SALE DATE'] = pd.to_datetime(df_residential['SALE DATE'], errors='coerce')
df_residential['YEAR'] = df_residential['SALE DATE'].dt.year

# Calculate the age of the property at the time of sale
df_residential['PROPERTY AGE AT SALE'] = df_residential['YEAR'] - df_residential['YEAR BUILT']

# Create a scatter plot with thresholds and log transformation applied to the sale price
df_residential = df_residential[df_residential['SALE PRICE'] <= 100000000]
df_residential['LOG SALE PRICE'] = np.log(df_residential['SALE PRICE'])

#Recreate the scatter plot with log-transformed sale price
plt.figure(figsize=(12,8))
sns.scatterplot(x='PROPERTY AGE AT SALE', y='LOG SALE PRICE', hue='YEAR', data=df_residential, palette='Set2', alpha=0.7)

plt.title('Log-Transformed Relationship Between Property Age At Sale, Sale Year, and Sale Price')
plt.xlabel('Property Age At Sale')
plt.ylabel('Log of Sale Price')
plt.grid(True)
plt.show()

# sns.lineplot(x='YEAR', y='SALE PRICE', data=df_residential, ci=None)
# plt.title('Price Trend of Residential Homes Over Time')
# plt.xlabel('Year')
# plt.ylabel('Sale Price')
# plt.show()


borough_mapping = {1:'Manhattan', 2:'Bronx', 3:'Brooklyn', 4:'Queens', 5:'Saten Island'}
df_residential['BOROUGH'] = df_residential['BOROUGH'].map(borough_mapping)

# Revert the BOROUGH column back to numeric values
borough_reverse_mapping = {'Manhattan': 1, 'Bronx': 2, 'Brooklyn': 3, 'Queens': 4, 'Saten Island': 5}
df_filtered['BOROUGH'] = df_filtered['BOROUGH'].replace(borough_reverse_mapping)

# Analyze the average sale price per borough
borough_avg_price = df_residential.groupby('BOROUGH')['SALE PRICE'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=borough_avg_price.index, y=borough_avg_price.values, palette='icefire')
plt.title('Average Sale Price per Borough')
plt.xlabel('Borough')
plt.ylabel('Average Sale Price')
plt.show()

# Calculate the mean sale price for each YEAR BUILT per TAX CLASS AT PRESENT
sale_price_trend = df_filtered.groupby(['YEAR BUILT','TAX CLASS AT PRESENT'])['SALE PRICE'].mean().reset_index()


# Plotting the trend
plt.figure(figsize=(12,8))
sns.lineplot(data=sale_price_trend, x='YEAR BUILT', y='SALE PRICE', hue='TAX CLASS AT PRESENT', marker='o', palette='tab10')

plt.yscale('log')
plt.title('Sale Price Trends by Year Built per TAX CLASS AT PRESENT')
plt.xlabel('Yar Built')
plt.ylabel('Average Sale Price')
plt.grid(True)
plt.legend(title='TAX CLASS AT PRESENT', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adding labels to each line similar to the first image
tax_classes = sale_price_trend['TAX CLASS AT PRESENT'].unique()
palette = sns.color_palette('tab10', len(tax_classes))  
for i, tax_class in enumerate(tax_classes):
    subset = sale_price_trend[sale_price_trend['TAX CLASS AT PRESENT'] == tax_class]
    original_label = tax_class_mapping.get(tax_class, str(tax_class))

# Adding a legend similar to a label box with original labels
handles, labels = plt.gca().get_legend_handles_labels()
original_labels = [tax_class_mapping[int(label)] for label in labels]
plt.legend(handles, original_labels, title='TAX CLASS AT PRESENT', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.show()
