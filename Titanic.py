#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gonzalobetancourt
"""

import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pandas import Series, DataFrame
from sklearn.metrics import accuracy_score



# read in the training data from folder
titanic_train = pd.read_csv('train.csv')

#### Data information ###

# 1 --> survived 
# 0 --> did not survive
# SibSp --> sinlings on ship
# Parch --> parents on board.
# cabin --> some inforamtion is NaN
# Ticket --> 
# Fate --> how much they paid
titanic_train.head()

titanic_train.info()

# A. Who were the passengers on the Titanic? (Ages, Gender, Class,..etc)
# B.	What deck were the passengers on and how does that their class?
# C.	Where did the passengers come from?
# D.	Which passengers traveled alone or with family?
# E.	What factors helped someone survive the sinking?

# pass a column argument
sns.catplot('Sex', data=titanic_train, kind='count')
# from the plot I can see that there were way more males than femals in the ship

# separate the genders by classes
sns.catplot('Sex', data=titanic_train, hue='Pclass', kind='count')

# reversing the order of Pclasa and Sex I can really see
# the difference of males and females and the class they 
# were in.

# Much more males in the 3rd class than females.
sns.catplot('Pclass', data=titanic_train, hue='Sex', kind='count')


# Males, females and children. I used an apply function to see the passangers that 
# were female, male and children

def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex

# Created a Person column that will have femal, male or child axis=1, column names
titanic_train['Person'] = titanic_train[['Age', 'Sex']].apply(male_female_child, axis=1)

# Now I graphed the female, male and children by class
sns.catplot('Pclass', data=titanic_train, hue='Person', kind='count')

# there are way more childresn in third class than 1st and 2nd class. 

# histogram allowed me to see the distribution of the ages.
titanic_train['Age'].hist(bins=70)
titanic_train['Age'].mean()

# Compare child, male, female total counts on board
titanic_train['Person'].value_counts()

# I generated mutiple plots on one figure
# aspect sets the ratio
# oldest person
oldest = titanic_train['Age'].max()

fig = sns.FacetGrid(titanic_train, hue='Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
fig.set(xlim=(0, oldest))
fig.add_legend()

# include children use Person
# KDE map goes past age 16 but that is just the bandwidth.
fig = sns.FacetGrid(titanic_train, hue='Person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
fig.set(xlim=(0, oldest))
fig.add_legend()



# Use class pclass
fig = sns.FacetGrid(titanic_train, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
fig.set(xlim=(0,oldest))
fig.add_legend()



#####################
# I have gottent a pretty good understanding of who the 
# passengers were. No I will take a look at what deck were 
# the passengers on and how does that affect their class?

# The cabin has information about the deck, however, it has
# several NaN values. I will take care of the null values first

deck = titanic_train['Cabin'].dropna()

deck.head()

# The level of the deck is classified A-G
# I used a for loop to get the first characted from Cabin

levels = []
for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.catplot('Cabin', data=cabin_df, palette='winter_d', kind='count')

# In this last graph there was a letter T I need to drop that 
# from my dataframe
cabin_df = cabin_df[cabin_df.Cabin != 'T']


sns.catplot('Cabin', data=cabin_df, palette='summer', kind='count')

# Readingabout the project on Kaggle, the port of embarkation
# C --> Cherbourg
# Q --> Queenstown
# S --> Southhampton

# Where did passengers board?
# From the graph in Q most of the passengers were from class 3
# In S most were also from class 3. 
sns.catplot('Embarked', data=titanic_train, hue='Pclass', kind='count')


#####################
# Which passengers traveled alone or with family?

# Sib --> siblings 1 Yes 0 No
# Parch --> 1 had children or parents
# if both are zero then they traveled alone

# added both columns because if the result is 0 then
# I know that the passenger traveled alone
titanic_train['Loner'] = titanic_train.SibSp + titanic_train.Parch

# locations where Loner column is grearer than zero
titanic_train['Loner'].loc[titanic_train['Loner'] > 0] = 'with Family'

titanic_train['Loner'].loc[titanic_train['Loner'] == 0] = 'Alone'

# Visualizing the travelers

sns.catplot('Loner', data=titanic_train, palette='Blues', kind='count')

# What factors helped with surviving the crash

titanic_train['Survivor'] = titanic_train.Survived.map({0:'No', 1:'Yes'})

sns.catplot('Survivor', data=titanic_train, palette='Set1', kind='count')

# The third class was lower
sns.catplot('Pclass', 'Survived', data=titanic_train, kind='point')

# Being a male or 3rd class were not favorable 
# for survival. 
sns.catplot('Pclass', 'Survived', data=titanic_train, hue='Person', kind='point')

# Is age a factor
# 0 --> did not survive
# 1 --> survived
# There seems to be a general trend in that the older
# a person was the least likely it was they survived.
sns.lmplot('Age', 'Survived', data=titanic_train)

# this shows trhe trend b/w the classes
sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_train, palette='winter')

# I used the x_bin argument to clean up the data a bit
# I created a generations and spread the ages 10 years apart 
# and 20 from 40 - 80
# Grab the points that are closes to generations
# and then create a standard deviation from there.
generations = [10, 20, 30, 40, 60, 80]

sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_train, palette='winter',x_bins=generations)
# as can be seen there is a higher standard deviation with 
# the older age

# It seems that if you were an older female you had a better
# chance of survival than an older male.
sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_train, palette='winter', x_bins=generations)

############# Follow up quesitons for additional research ###########
# Did the deck have an effect on the passengers survival rate? Higher or lower ded, male female
# Did the answer match up with my intuition
# Did having a family member increase the odds of survivinf the crash? Siblings or parent



