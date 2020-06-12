# My General Guideline: Starting on a Project
-----------

#### Example Training and Testing sets.
		
		This document will include my notes on how to get tarted on a machine 
		learning project. The main purpose of this project was to start
		putting to work some of the skills I learned about working with data.
		Using Nympy and Pandas to manipulate the data and Matplotlib and
		seaborn to visualize the data I am working with. 
		
		I was not trying to make this a very long project since my overall
		goal was to start getting comfortable with starting a project and 
		learning to ask key questions about the data. 
		
The dataset I used can be found at [Titanic: A Kaggle dataset](https://www.kaggle.com/c/titanic)

Before starting on any project make sure to understand the type of data I will be working with. What is the underlying goal, am I trying to predict values? Do I want to understand certain outcomes? What variables influence the outcome, in a positive or negative way?

Try to have some general/basic questions before analyzing the data. Now that I know what type of data I have, take a look at it. 

###### **Training set:** I will be using the Titanic data set to analyze the survival rates of the crash. My model will be based on "features" like passengers' gender and class.

###### **Testing set:** Will be used to see how good my model performs on unseen data. I will try to predict the true values for each passenger, whether or not they survived the Titanic.

<br>
-----------------

1. open file(s) with pandas __pd.read_csv('name.csv')__

		I use pandas to read in the dataset. 
	
	For example: __titanic\_train = pd.read_csv('train.csv')__

	Once the dataset is in a variable use __head.()__ to look at the first 
	few rows.
		
		
2. looking at the first few rows:
		
		- name.head()
		- understand what each column represents. Get an understanding
			of the type of data I will be working with.
		- name.info() gives additional information about the data in 
			terms of totals.
		
3. Now that I have seen the data and looked at the column names. Think 
	about some basic questions that can guide me into a deeper 
	understanding of the data.
	
		For Example:
				A.	Who were the passengers on the Titanic? (Ages, 
					Gender, Class,..etc)
				B.	What deck were the passengers on and how does that 
					relate to their class?
				C.	Where did the passengers come from?
				D.	Which passengers traveled alone or with family?
				E.	What factors helped someone survive the sinking?
				
4. 	 Also, think of ways to visualize the data to get a visual
	 understanding of what I will be working with. 
	 
	 For this project and many more to come I will be using seaborn and 
	 matplotlib to visualize the data. 
	 
	 Use seaborn:
	 
	- sns.catplot('Sex', data=titanic_train, kind='count'): passs in a column name element. The dataset, and kind='count'. This will allowed me to see the number of males vs female passangers. 
	- I also passes a __hue__="Pclass". The __hue__ separated the "Sex" by class on the Titanic; 1st, 2nd and 3rd class. 
	- name['attribute'].hist(bins=#)
	- I can also create additional columns if needed by using an apply function based on some conditions. 

5. Now that I have gotten a pretty good view of my data. See if thier are any nan value that I may need to drop or replace. With drop.na() or drop.fillna()

	- I used an lmplot() to spot any linear trends amongst the passengers that survived and didn't survive, where they female, male or children, what class were they in?


6. For datasets that require training and testing data. I will typically use 20% of the data for testing and the rest for training. 


__Training set:__ data used to train the model

__Testing set:__ data will be used to see how good my model performs. 








