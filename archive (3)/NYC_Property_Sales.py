# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:25:58 2024

@author: Jaekyeong
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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

plt.figure(figsize=(10,6))
sns.histplot(df_filtered['SALE PRICE'], bins=50, kde=True, color='blue')
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

label_encoder = LabelEncoder()

# Applying Label Encoding to categorical columns
df_filtered['NEIGHBORHOOD'] = label_encoder.fit_transform(df_filtered['NEIGHBORHOOD'])
df_filtered['BUILDING CLASS CATEGORY'] = label_encoder.fit_transform(df_filtered['BUILDING CLASS CATEGORY'])
df_filtered['TAX CLASS AT PRESENT'] = label_encoder.fit_transform(df_filtered['TAX CLASS AT PRESENT'])
df_filtered['BUILDING CLASS AT PRESENT'] = label_encoder.fit_transform(df_filtered['BUILDING CLASS AT PRESENT'])
df_filtered['BUILDING CLASS AT TIME OF SALE'] = label_encoder.fit_transform(df_filtered['BUILDING CLASS AT TIME OF SALE'])

df_filtered['SALE PRICE'] = pd.to_numeric(df_filtered['SALE PRICE'], errors='coerce')
df_filtered['LAND SQUARE FEET'] = pd.to_numeric(df_filtered['LAND SQUARE FEET'], errors='coerce')
df_filtered['GROSS SQUARE FEET'] = pd.to_numeric(df_filtered['GROSS SQUARE FEET'], errors='coerce')

df_filtered = df_filtered.drop(columns=['Unnamed: 0','EASE-MENT','ADDRESS','APARTMENT NUMBER'])

print(df.dtypes)

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

# Plot the distribution of propery ages
plt.figure(figsize=(10, 6))
sns.histplot(df['PROPERTY AGE'], bins=30, kde=True)
plt.title('Distribution of Property Ages')
plt.xlabel('Property Age (years)')
plt.ylabel('Frequency')
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

# Filter the dataset to include only Tax Class 1 properties and its variations (e.g., '1','1A','1B','1C')
df_residential = df_filtered[df_filtered['TAX CLASS AT PRESENT'].astype(str).str.startswith('1')]
df_residential = df_residential.copy()

plt.figure(figsize=(10,8))
sns.heatmap(df_residential.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.show()

# Analyze the price trende of residential homes over time
df_residential['SALE DATE'] = pd.to_datetime(df_residential['SALE DATE'], errors='coerce')
df_residential['YEAR'] = df_residential['SALE DATE'].dt.year

# Calculate the age of the property at the time of sale
df_residential['PROPERTY AGE AT SALE'] = df_residential['YEAR'] - df_residential['YEAR BUILT']

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(12,8))
sns.scatterplot(x='PROPERTY AGE AT SALE', y='SALE PRICE', hue='YEAR', data=df_residential, palette='Set2', alpha=0.7)
# plt.figure(figsize=(12,6))

plt.title('Relationship Between Property Age at Sales, Sale Year, and Sale Price')
plt.xlabel('Property Age at Sale (Year)')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()
# sns.lineplot(x='YEAR', y='SALE PRICE', data=df_residential, ci=None)
# plt.title('Price Trend of Residential Homes Over Time')
# plt.xlabel('Year')
# plt.ylabel('Sale Price')
# plt.show()


borough_mapping = {1:'Manhattan', 2:'Bronx', 3:'Brooklyn', 4:'Queens', 5:'Saten Island'}
df_residential['BOROUGH'] = df_residential['BOROUGH'].map(borough_mapping)

# Analyze the number of residentail homes per neiborhood
borough_counts = df_residential['BOROUGH'].value_counts()

plt.figure(figsize=(12,8))
sns.barplot(x=borough_counts.index, y=borough_counts.values, palette='viridis')
plt.title('Number of Residential Homes per Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Residentail Homes')
plt.xticks(rotation=90)
plt.show()

# Revert the BOROUGH column back to numeric values
borough_reverse_mapping = {'Manhattan': 1, 'Bronx': 2, 'Brooklyn': 3, 'Queens': 4, 'Saten Island': 5}
df_filtered['BOROUGH'] = df_filtered['BOROUGH'].replace(borough_reverse_mapping)

# Analyze the average sale price per borough
borough_avg_price = df_residential.groupby('BOROUGH')['SALE PRICE'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=borough_avg_price.index, y=borough_avg_price.values, palette='viridis')
plt.title('Average Sale Price per Borough')
plt.xlabel('Borough')
plt.ylabel('Average Sale Price')
plt.show()

# Ensure 