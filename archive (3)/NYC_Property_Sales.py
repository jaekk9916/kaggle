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

sns.set(style='whitegrid')

plt.figure(figsize=(10,6))
borough_counts = df['BOROUGH'].value_counts()
sns.barplot(x=borough_counts.index, y=borough_counts.values, palette='viridis')
plt.title('Number of Data Points by BOROUGH')
plt.xlabel('BOROUGH')
plt.ylabel('Number of Data points')
plt.show()

year_built_trend = df['YEAR BUILT'].value_counts().reset_index(name='counts')
plt.figure(figsize=(12,6))
sns.lineplot(data=year_built_trend, x='YEAR BUILT', y='counts')
plt.title('Trend in Building Constructed Over Time')
plt.xlabel('Year Built')
plt.ylabel('Number of buildings Constructed')
plt.grid(True)
plt.show()

