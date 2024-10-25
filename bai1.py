# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
# importing data
ad_prediction = pd.read_csv('C:/Users/PC/Downloads/ad_click_dataset.csv')
print(ad_prediction.head())

# Basic information about the dataset
ad_prediction.info()

# Statistical summary (mean, std, min, max, etc.)
ad_prediction.describe()

# Frequency of categorical features
ad_prediction['gender'].value_counts()
ad_prediction['device_type'].value_counts()
ad_prediction['ad_position'].value_counts()
ad_prediction['browsing_history'].value_counts()
print(ad_prediction['time_of_day'].value_counts())

# Distribution of the target variable
print(ad_prediction['click'].value_counts(normalize=True))  # Normalized percentage of clicks and non-clicks

# Distribution of age
plt.figure(figsize=(8,6))
sns.histplot(ad_prediction['age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Click distribution by age
plt.figure(figsize=(8,6))
sns.boxplot(x='click', y='age', data=ad_prediction)
plt.title('Age vs Click Behavior')
plt.xlabel('Click (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

# Click rate by device type
plt.figure(figsize=(8,6))
sns.countplot(x='device_type', hue='click', data=ad_prediction)
plt.title('Click Rate by Device Type')
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.show()

# Click rate by ad position
plt.figure(figsize=(8,6))
sns.countplot(x='ad_position', hue='click', data=ad_prediction)
plt.title('Click Rate by Ad Position')
plt.xlabel('Ad Position')
plt.ylabel('Count')
plt.show()

# Click distribution by gender
ad_prediction.groupby('gender')['click'].mean().plot(kind='bar')
plt.title('Click Rate by Gender')
plt.ylabel('Click Rate')
plt.show()

# Click distribution by time of day
ad_prediction.groupby('time_of_day')['click'].mean().plot(kind='bar')
plt.title('Click Rate by Time of Day')
plt.ylabel('Click Rate')
plt.show()


# Cross-tabulation between device type and click behavior
pd.crosstab(ad_prediction['device_type'], ad_prediction['click'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Click Rate by Device Type')
plt.show()



