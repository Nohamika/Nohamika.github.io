import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.cm as cm

import community as community
import networkx as nx
from scipy.spatial import distance_matrix


class Corona_Virus_Model():

    def __init__(self, death_data_file = 'covid_19_data.csv', countries_data_file='countries_data.csv'):
        """
        getting the files with the data
        :param death_data_file: the file name with the data about the COVID-19 related deaths in different countries over time
        :param countries_data_file: the file name with the data about the contries (e.g: population size, gdp, etc)
        """
        self.death_data_file = death_data_file
        self.countries_data_file = countries_data_file

        # names of the countries
        countries_data = pd.DataFrame(pd.read_csv(self.countries_data_file))
        self.names = countries_data['country']

    # arrange a list of the death data
    def get_data(self,to_day=102):
        """
        create a list that include a list of the number of dead people every day from t0 for each country
        :param to_day: how many days the countries have (must be the same length for all of them)
        :return: list of lists
        """
        # covid data
        df = pd.DataFrame(pd.read_csv(self.death_data_file))
        deaths_data = df[['location', 'new_deaths_per_million', 'total_deaths_per_million']]

        returned_death_data = []
        for country in self.names:
            country_data = deaths_data.loc[deaths_data['location'] == country]
            # day1 = reaching 5 total deaths. the clustering is from day10 to day 102.
            # if data is added this needs to be changes
            returned_death_data.append(list(np.log(country_data['total_deaths_per_million']))[10:to_day])
        return returned_death_data



    # divide the countries into clusters according to their corona virus mortality rate
    def fit(self, data, visualize=False,plot=False, res=1, divider=100,modularity=False,print_clusters=True):
        """

        :param data(list of lists): the list of lists with the death data of every country
        :param visualize(bool): should the graph showing the partition appear
        :param plot(bool): should the plot with the graph of the total death by day of each country appear
        :param res(0,inf): the louvain partition resolution attribute (to change the number of clusters)
        :param divider(0,inf): a number to calculate the similarity between the countries.
        the bigger the number the bigger the similarities are.
        In the code its: similarity = divider/distance
        :param modularity(bool): should this louvain partition attribute appear
        :param print_clusters(bool): should the clusters with the countries that belong to each cluster be printed
        :return:labels(list): return a list with the cluster
        (first number in the list is the cluster of the first number in the data frame and etc)
        """
        def print_cluster(prediction,names,num_of_clusters):
            # Each loop prints a different cluster
            for i in range(num_of_clusters):
                # Going trough all the countries, if the country prediction is the same as i, add it to the list
                countries_names = [names[j] for j in range(len(names)) if prediction[j] == i]
                print("number of countries in cluster", i, "is:", prediction.count(i), "and the countries are:",
                      countries_names)


        countries_dis = pd.DataFrame(distance_matrix(data, data))
        print(countries_dis)
        weight_data = []

        for c1 in range(len(self.names)):
            for c2 in range(c1 + 1, len(self.names)):
                weight_data.append([self.names[c1], self.names[c2], countries_dis[c1][c2]])
        weight_data = pd.DataFrame(weight_data)
        weight_data.columns = ['source', 'target', 'weight']
        #
        weight_data['weight'] = divider / weight_data['weight']

        G = nx.Graph()
        G.add_weighted_edges_from(weight_data[['source', 'target', 'weight']].values)

        partition = community.best_partition(G, resolution=res)
        if modularity:
            print('modularity:', community.modularity(partition, G))

        if visualize == True:
            # draw the graph
            pos = nx.spring_layout(G)
            # color the nodes according to their partition
            cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
            nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                                   cmap=cmap, node_color=list(partition.values()))
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='w')
            plt.title('Graph of countries clustered by death rate',fontsize = 'xx-large')
            plt.show()
        # list of the clusters:
        results = list(partition.values())
        # printing the clusters with the countries the belong to
        if print_clusters:
            print_cluster(results,self.names,np.max(results)+1)
        # return a list with the cluster
        # (first number in the list is the cluster of the first number in the data frame and etc)
        return results


    def predict(self,labels,features=['gdp_per_capita', 'Obesity rate', 'longitude', 'Political culture'],feature_importance = False):
        # DataFrame with the data about the countries
        countries_df = pd.read_csv(self.countries_data_file)
        # Setting the country column to the index because I don't need it for the prediction part
        countries_df = countries_df.set_index('country')

        # feature selection
        countries_df = countries_df[features]

        # Split the data into train test
        X_train, X_test, y_train, y_test = train_test_split(countries_df, labels, random_state=42, test_size=0.25,
                                                            shuffle=True)
        # The xgb model
        clf = xgboost.XGBClassifier(learning_rate=0.5, max_depth=4, n_estimators=400)

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

        if feature_importance == True:
            # plot feature importance
            from xgboost import plot_importance
            plot_importance(clf,importance_type="weight")
            plt.title('Feature Importance',fontsize='xx-large')
            plt.xlabel('F-score',fontsize='xx-large')
            plt.ylabel('Features',fontsize='xx-large')
            plt.yticks(fontsize='x-large')
            plt.xticks(fontsize='x-large')
            plt.show()



# how to use the model
cvm = Corona_Virus_Model()
death_data = cvm.get_data()
labels = cvm.fit(death_data,visualize=False)
# the labels looka like [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 2, 2, 0, 2, 0, 0, 1, 2, 0, 2, 0, 0, 2, 0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 0, 1, 2, 2, 2, 0, 1, 1, 1, 2, 1, 0, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1]
cvm.predict(labels = labels ,feature_importance=False)
