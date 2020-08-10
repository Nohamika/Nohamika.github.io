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
Afghanistan | 0.283
... | ...
Zimbabwe | 2.422
Zimbabwe | 2.691
Zimbabwe | 2.759
Zimbabwe | 2.566

For each country I named day1 (or t0 if you like) the first day with at least 5 dead people.
Then, to make sure all the countries have the same number of days, I took the biggest number of days all the countries have entries for, and I got 102 days each.
So now I have a list of lists, I have 116 lists inside the big list, each list represent a country, and in each list of a country I have 92 entries which are the number of dead people per million that day (from day 10 to day 102)

**there will be an image here**

Lastly ill do log on all the data in the list of lists and our data is ready.

```
[[-1.584745299843729, -1.5005835075220182,...],...,[...,2.908266247772715, 2.908266247772715]]
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
