# Intro

The idea behind this paper is to try to understand what causes corona virus to spread faster in one country than another. What features about a country makes it so much better coping with the pandemic. the results may surprise you.
## In this article I will cover the three parts of the model:

1.  Collecting and preparing the data
2.  Clustering the countries according to their covid-19 spread rate
3.  Using data about the countries to classify them into the different clusters from step 2.

I will show parts of my code within the explanation, if you want the full model code you can find it in the github repository.

# Step 1: collecting and preapring the data

I collected the data from the website "Our World in Data".
I download a csv that include for each country the number of total dead people from COVID-19 for each date. The csv file I used had the dates until 31/07/2020, but the site is updated every day so new data can be added.
You can find the file with the data you can find it in the Github repository.
Lets look at our data:

location | total_deaths_per_million
-------- | ------------------------
Afghanistan | 0.128
Afghanistan | 0.180
Afghanistan | 0.180
... | ...
Zimbabwe | 2.691
Zimbabwe | 2.759
Zimbabwe | 2.566

For each country I named day1 (or t0 if you like) the first day with at least 5 dead people.
Then, to make sure all the countries have the same number of days, I took the biggest number of days all the countries have entries for, and I got 102 days each.
So now I have a list of lists, I have 116 lists inside the big list, each list represent a country, and in each list of a country I have 92 entries which are the number of dead people per million that day (from day 10 to day 102):

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

Great, now we will cluster the countries according to those distances using the Louvain method.
The Louvain method is a community's detection method in graphs. This method partitions the nodes of the graph into different communities according to the edge's weights. Here, Every node is a country, the weight of an edge repsrest the similarity between the 2 countries, and the communities are the clustesrs.

To use this method I will use the Networkx library and create an empty graph.
To fill the graph with the nodes, edges and weights that I need, I will create a data frame with all the edges (there is one edge between every 2 countries) and the weight of that edge.
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

Our graph ready, time to use the Louvain method to cluster the countries. For that we will use the community package:
```
import community
partition = community.best_partition(G)
```
the function best_partition gives us the different clusters. 'G' is the graph I made before.
the result is saved in the variable 'partition' which is a dictionary that each key is a country (node) and the value is the cluster:
```
{'Kenya': 0, 'Niger': 0, 'Mali': 0, 'Somalia': 0, 'Liberia': 0,...,'Austria': 2, 'Greece': 1}
```

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
