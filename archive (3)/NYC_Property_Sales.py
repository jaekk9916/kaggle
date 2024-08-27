# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:25:58 2024

@author: Jaekyeong
"""

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('nyc-rolling-sales.csv')

print(df.isnull().sum())

print(df.head())

print(df.describe())

print(df.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns

# Mapping borough numbers to names
borough_mapping = {1:'Manhattan', 2:'Bronx', 3:'Brooklyn', 4:'Queens', 5:'Saten Island'}
df['BOROUGH'] = df['BOROUGH'].map(borough_mapping)

sns.set(style='whitegrid')

plt.figure(figsize=(10,6))
borough_counts = df['BOROUGH'].value_counts()
sns.barplot(x=borough_counts.index, y=borough_counts.values, palette='viridis')
plt.title('Number of Data Points by BOROUGH')
plt.xlabel('BOROUGH')
plt.ylabel('Number of Data points')
plt.show()


df_filtered = df[df['YEAR BUILT'] > 1800]
year_built_trend = df_filtered['YEAR BUILT'].value_counts().reset_index(name='counts')
plt.figure(figsize=(12,6))
sns.lineplot(data=year_built_trend, x='YEAR BUILT', y='counts')
plt.title('Trend in Building Constructed Over Time')
plt.xlabel('Year Built')
plt.ylabel('Number of buildings Constructed')
plt.grid(True)
plt.show()

df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')

df['YEAR_MONTH'] = df['SALE DATE'].dt.to_period('M')
sale_data_trend = df['YEAR_MONTH'].value_counts().sort_index()

plt.figure(figsize=(12,6))
sns.lineplot(x=sale_data_trend.index.astype(str), y=sale_data_trend.values)
plt.title('Trend in Real Estate Sales Over Time')
plt.xlabel('Year of Sale')
plt.ylabel('Number of Sales')
plt.grid(True)
plt.show()

