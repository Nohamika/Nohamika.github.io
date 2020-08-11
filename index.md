# Intro

The idea behind this paper is to try to understand what causes corona virus to spread faster in one country than another. What features about a country makes it so much better coping with the pandemic. the results may surprise you.
## In this article I will cover the three parts of the model:

1.  Collecting and preparing the data
2.  Clustering the countries according to their covid-19 spread rate
3.  Using data about the countries to classify them into the different clusters from step 2.

I will show parts of my code within the explanation, if you want the full model code you can find it in the github repository.

# Step 1: collecting and preapring the data

I collected the data from the website "Our World in Data".
I download a csv that includes for each country the number of total dead people from COVID-19 for each date.
The csv file I used had the dates until 31/07/2020, but the site is updated every day so new data can be added.
You can find the file with the data in the Github repository.
For each country I named day1 (or t0 if you like) the first day with at least 5 dead people.
The following data frame include each country and the total deaths from Covid-10 from day1:

location | total_deaths_per_million
-------- | ------------------------
Afghanistan | 0.128
Afghanistan | 0.180
Afghanistan | 0.180
... | ...
Zimbabwe | 2.691
Zimbabwe | 2.759
Zimbabwe | 2.566


To make sure all the countries have the same number of days, I took the biggest number of days all the countries have entries for, and I got 102 days each.
This process created a list of lists, 116 lists inside the big list, each list represent a country, and in each list of a country I have 92 entries which are the number of dead people per million that day (from day 10 to day 102):

![the list of lists.png](https://raw.githubusercontent.com/Nohamika/Nohamika.github.io/master/the%20list%20of%20lists.png)

Lastly ill do log on all the data in the list of lists and our data is ready:

```
[[-1.584745299843729, -1.5005835075220182,...],...,[...,2.908266247772715, 2.908266247772715]]
```

# Step 2: Clustring the Countries
To cluster countries I need to understand the similarity between them. To do that I create a distance matrix:
``` 
from scipy.spatial import distance_matrix
countries_dis = pd.DataFrame(distance_matrix(data, data))
```

This matrix shows the distance between every 2 countries:
```
           0          1          2    ...        113        114        115
0     0.000000   6.659591   9.301262  ...  34.289718  37.705250  23.534778
1     6.659591   0.000000   4.683849  ...  29.019417  32.331640  18.004010
2     9.301262   4.683849   0.000000  ...  25.128690  28.515404  14.322602
3    13.712355   7.769635   4.931949  ...  21.422582  24.644642  10.304785
4    15.723163  10.284375   6.822073  ...  19.221853  22.401135   8.153682
..         ...        ...        ...  ...        ...        ...        ...
111  35.660148  30.138568  26.460299  ...   3.958171   3.166323  12.199858
112  50.512313  45.760384  41.620737  ...  17.451746  15.246749  28.432069
113  34.289718  29.019417  25.128690  ...   0.000000   3.770752  11.212965
114  37.705250  32.331640  28.515404  ...   3.770752   0.000000  14.356435
115  23.534778  18.004010  14.322602  ...  11.212965  14.356435   0.000000
```

I attend to cluster the countries according to those distances using the Louvain method.
The Louvain method is a community's detection method in graphs. This method partitions the nodes of the graph into different communities according to the edge's weights.
Here, Every node is a country, the weight of an edge repsrest the similarity between the 2 countries, and the communities are the clustesrs.

To use this method I will use the Networkx library and create an empty graph.
To fill the graph with the nodes, edges and weights that I need, I will create a data frame with all the edges (there is one edge between every 2 countries) and the weight of every edge.
The more similar 2 countries are I want the edge that connects them to have a higher weight, in other words, The smaller the distance the bigger the weight. Therefore, I will calculate the weight as follow:

                      Weight = Divider/ Distance

The divider can be any number greater than 0, I choose 100.


The data frame for the graph is as follow:
```
 source        target     weight
Kenya         Niger  15.015936
Kenya          Mali  10.751229
Kenya       Somalia   7.292694
Kenya       Liberia   6.360043
Kenya  Burkina Faso  10.940958
...       ...           ...        ...
Italy       Austria   6.558775
Italy        Greece   3.517155
Finland       Austria  26.519906
Finland        Greece   8.918247
Austria        Greece   6.965518
```

Our graph ready! Time to use the Louvain method to cluster the countries. For that we will use the community package:
```
import community
partition = community.best_partition(G)
```
the function best_partition gives us the different clusters. 'G' is the graph I made before.
the result is saved in the variable 'partition' which is a dictionary that each key is a country (node) and the value is the cluster:
```
{'Kenya': 0, 'Niger': 0, 'Mali': 0, 'Somalia': 0, 'Liberia': 0,...,'Austria': 2, 'Greece': 1}
```
The clusters are ready, lets take a look at them:

lets look clusters of the countries:
* cluster 0 (51 countries): 'Kenya', 'Niger', 'Mali', 'Somalia', 'Liberia', 'Burkina Faso', 'Tanzania’, 'Democratic Republic of Congo’,…
* cluster 1 (22 countries): 'South Africa', 'Guatemala', 'Honduras', 'Guyana', 'Bolivia', 'El Salvador’,…
* cluster 2 (43 countries): 'Dominican Republic', 'Peru', 'Brazil', 'Panama', 'Ecuador’,…

We can also draw the graph:

![graph_partition](https://raw.githubusercontent.com/Nohamika/Nohamika.github.io/master/graph_partition.png)

each node is a country colored according to its cluter. the closter 2 countries are to each other is means they are more similer.

lastly I will plot the time seires of the COVID-19 spread by the days of each country and color it by the cluster:

![/master/time_series_plot.png](https://raw.githubusercontent.com/Nohamika/Nohamika.github.io/master/time_series_plot.png)


# Step 3: predicting the country's cluster by its data

We are finally at the fun part, finding the characteristic of a country that makes it good (or bad) dealing with the pandemic.

The model that I am building is a supervised classification model. The labels are the clusters from the previous step, and the data is the features about the country as you can see in the following data frame:

```
         is lockdown  num_days  ...  Civilliberties  Obesity rate
country                         ...                              
Kenya              0       100  ...            4.41           7.1
Niger              0       100  ...            4.71           5.5
Mali               0       100  ...            5.59           8.6
Somalia            0       100  ...            2.94           8.3
Liberia            1        20  ...            5.59           9.9
```
In total there are 23 features:
```
Index(['is lockdown', 'num_days', 'Literacy(%)', 'population',
       'population_density', 'median_age', 'aged_65_older', 'gdp_per_capita',
       'hospital_beds_per_100k', 'latitude', 'longitude',
       'death_by_lack_of_hygiene', 'stringency index', 'cvd_death_rate',
       'diabetes_prevalence', 'life_expectancy', 'democracy_score',
       'Electoral processand pluralism', 'functioning of government',
       'Political participation', 'Political culture', 'Civilliberties',
       'Obesity rate'],
      dtype='object')
```

For my model, I used the XGBClassifier from the xgboost package:
```
# Split the data into train test
X_train, X_test, y_train, y_test = train_test_split(countries_df, labels, random_state=42, test_size=0.25,
                                                            shuffle=True)
# The xgb model
clf = xgboost.XGBClassifier()

# Fitting the data
clf.fit(X_train, y_train)

# Predicting the train data and checking the accuracy to see if the model learns
y_pred_train = clf.predict(X_train)
acc = accuracy_score(y_train, y_pred_train)
print("Accuracy of train %s is %s" % (clf, acc))

# Predicting the test data (a data the model havent seen) to evaluate it
y_pred_test = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
print("Accuracy of test %s is %s" % (clf, acc))

# a confusion matrix on the test data
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix of %s is" % (clf))
print(cm)
```
running the simplest model, lets see the results:
```
Accuracy of train XGBClassifier(objective='multi:softprob') is 1.0
Accuracy of test XGBClassifier(objective='multi:softprob') is 0.7241379310344828
Confusion Matrix of XGBClassifier(objective='multi:softprob') is
[[14  1  1]
 [ 2  2  1]
 [ 2  1  5]]
 ```
The train accuracy is 1.0 so we know that the model is learning but our test accuracy is 0.72 which is good but still could use improvement.

First thing we can do is feature selection. The features I have right now (e.g.: population density, median age) are just my guesses of things I think can affect the spread of corona-virus, but not all of them are good predictors. 
To choose the best features I used a method called forward selection. you can read about it here: [forward selection in wikipidia](https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches)

The forward selection results are: **gdp per capita, Obesity rate, longitude and Political culture.**

let see the results after feature selection:
 ```
Accuracy of train XGBClassifier(objective='multi:softprob') is 1.0
Accuracy of test XGBClassifier(objective='multi:softprob') is 0.896551724137931
Confusion Matrix of XGBClassifier(objective='multi:softprob') is
[[16  0  0]
 [ 1  5  0]
 [ 1  1  5]]
 ```
 The model has improved, yay!
 
The features in the model are very intreting, let look how much each of them contributes to the predition of the model by plotting a feature importance bar plot:
![Feature importance after](https://raw.githubusercontent.com/Nohamika/Nohamika.github.io/master/Feature_importance_after.png)

It appears that GDP per capita is the most important feature. This makes sense to me, the wealter the country the more resurces it has to do tests and give medical treatment to those in need.
Obesity rate also makes sense beacuse over weight people are in a danger group, which mean an over weight person who catches COVID-19 is more likey to die from it than a person with a standard BMI.
Political culture is hard to define but it is masured by the trust the people give to their goverment, if the people listen to the govrement laws and desicions, and how much influence does the people have on the goverment, or in other words, if the govrement really represents the people. This means the the functioning of the goveremnt and the peoples trust in it, matters when it comes to deal with a pandamic, this may not be suprising, yes it is very intresting.
Lastly we have the longitude feature. this feature suprised me beacuse I can not think of a direct influence on the COVID-19 situation. but to see how longitude effect, I colored a world map by the clusters:


In fact lets take a look at how all of the 4 features affect. To do that I used a UMAP.
UMAP is a demision reducation tool that helps present high dimensional data on a 2D scatter plot.
I plotted the data about the COVID-19 deaths using UMAP and colored it by each of the features. Each point represent a country:


 
 
 
 Another thing we can do is model tuning. Every machine learning model, XGBoost included, has different paramaters that can be changed to improve model performance.
 Sadly, the only way to find those much wanted paramters is to try A LOT of options and hope we will get the best ones. I used GridSearchCV to tune my model if you want to read more about it, [click here](https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost).
 
 The paramaters I got(the paramaters that does not appear means the deafult option is the best): ```learning_rate=0.5, max_depth=4, n_estimators=400```
 
 lets look at out final results:
 ```
 Accuracy of train XGBClassifier(learning_rate=0.5, max_depth=4, n_estimators=400,
              objective='multi:softprob') is 1.0
Accuracy of test XGBClassifier(learning_rate=0.5, max_depth=4, n_estimators=400,
              objective='multi:softprob') is 0.9310344827586207
Confusion Matrix of XGBClassifier(learning_rate=0.5, max_depth=4, n_estimators=400,
              objective='multi:softprob') is
[[15  0  0]
 [ 0  7  0]
 [ 1  1  5]]
```
The accuracy has gone up to 93%! 93% might not seem as exciting as 99.8% models we see on kaggle but if we look at the confusion matrix we can see there are only 2 countries that are miss predict.

# conclusion


## this is a title
# this is more of a title


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/Nohamika/Nohamika.github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
```markdown
import pandas as pd
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Nohamika/Nohamika.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
