import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import nltk
import warnings
import itertools
from pathlib import Path
from sklearn import preprocessing, cluster, model_selection, metrics, svm, ensemble, decomposition, linear_model, tree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from wordcloud import WordCloud, STOPWORDS
from IPython.display import display, HTML
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import datetime

warnings.filterwarnings("ignore") #sets up a filter to suppress certain types of warnings in the notebook.
plt.rcParams["patch.force_edgecolor"] = True #configuration for Matplotlib. It sets the edge color for patches (e.g., bars in a bar chart) to be more visible.
plt.style.use('fivethirtyeight') #sets the style for Matplotlib plots. In this case, it's using the 'fivethirtyeight' style, which emulates the style of graphics used by the website FiveThirtyEight
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1) #configuration for Matplotlib, appearance of elements like bars in a plot

#__________________

## 1. Data preparation

# read the datafile
df_initial = pd.read_csv('data.csv',encoding="ISO-8859-1",
                         dtype={'CustomerID': str,'InvoiceID': str})
print('Dataframe dimensions:', df_initial.shape)

#Step-1
# show first 10 lines and data frame info
display(df_initial.head(10))
display(df_initial.info())

#Step-2
# Drop null values (For ex : There are rows with unassigned customers).
df_dropna = df_initial.dropna()
# Show no.of rows before and after the action.
print(f'Number of rows before dropping null values: {df_initial.shape[0]}')
print(f'Number of rows after dropping null values: {df_dropna.shape[0]}')

#Step-3
# Drop Duplicate Rows.
df_drop_duplicates = df_dropna.drop_duplicates()
# Show no.of rows before and after the action.
print(f'Number of rows before dropping duplicate rows: {df_dropna.shape[0]}')
print(f'Number of rows after dropping duplicate rows: {df_drop_duplicates.shape[0]}')

#Step-4
# Countries where orders are made
# df_cleaned[['CustomerID', 'InvoiceNo', 'Country']] is a new DataFrame with only three columns
# The following counts the number of rows with unique combinations of CustomerID, InvoiceNo, and Country and the result is stored in a new
temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()  #The result is a Series where the index represents unique countries and the values represent the count of orders made from each country.
# Plot Choropleth World Map displaying the number of orders per country.
# It specifies the data, layout, and color scale for the map using Plotly's API.
# Finally, it uses the Plotly offline module to display the map.
# defining a dictionary called data which contains various parameters needed to create the choropleth map.
data = dict(
            type='choropleth', #specifies that we're creating a choropleth map.
            locations = countries.index, # .index is used to access the index labels of countries Series, i.e. the country names
            locationmode = 'country names', # tells Plotly to interpret the values in locations as country names
            z = countries, #sets the values associated with each country, which will determine the color on the map. In this case, it's the number of orders.
            text = countries.index, #provides the text that will be displayed when one hovers over each country.
            colorbar = {'title':'Order nb.'}, #sets the title for the colorbar.
            colorscale=[
                        [0, 'rgb(224,255,255)'],
                        [0.01, 'rgb(166,206,227)'],
                        [0.02, 'rgb(31,120,180)'],
                        [0.03, 'rgb(178,223,138)'],
                        [0.05, 'rgb(51,160,44)'],
                        [0.10, 'rgb(251,154,153)'],
                        [0.20, 'rgb(255,255,0)'],
                        [1, 'rgb(227,26,28)']
                       ],
            reversescale = False #specifies that the colors should not be reversed.
        )
# defining a dictionary called layout which contains parameters related to the layout and appearance of the map.
layout = dict(
                title='Number of orders per country',
                geo = dict(showframe = True, projection={'type':'mercator'})
            )
# creating a Figure object using the provided data and layout.
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)

#Step-5
# Likewise explore the Cancelling Orders, Stock Code and price.
# Exploring the Cancelled Orders
df_cancelled_orders = df_cleaned[
                            df_cleaned['InvoiceNo'].str.startswith('C')
                        ]
temp = df_cancelled_orders[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()
data = dict(
            type='choropleth',
            locations = countries.index,
            locationmode = 'country names',
            z = countries,
            text = countries.index,
            colorbar = {'title':'No. of Cancelled Orders'},
            colorscale=[
                        [0, 'rgb(224,255,255)'],
                        [0.01, 'rgb(166,206,227)'],
                        [0.02, 'rgb(31,120,180)'],
                        [0.03, 'rgb(178,223,138)'],
                        [0.05, 'rgb(51,160,44)'],
                        [0.10, 'rgb(251,154,153)'],
                        [0.20, 'rgb(255,255,0)'],
                        [1, 'rgb(227,26,28)']
                       ],
            reversescale = False
        )
layout = dict(
                title='Number of cancelled orders per country',
                geo = dict(showframe = True, projection={'type':'mercator'})
            )
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
#Exploring how many unique stock numbers are there
unique_stock_numbers = df_cleaned['StockCode'].nunique()
print(f'Number of unique stock numbers: {unique_stock_numbers}')
#Exploring the number of unique products sold by exploring number of unique Stock Codes
# Filter out cancelled orders
valid_orders = df_cleaned[~df_cleaned['InvoiceNo'].str.startswith('C')]
# Finding unique stock numbers in valid orders
unique_stock_numbers = valid_orders['StockCode'].nunique()
print(f'Number of unique stock numbers (excluding cancelled orders): {unique_stock_numbers}')
# Exploring prices
# Summary Statistics
price_summary = valid_orders['UnitPrice'].describe()
print("Price Summary Statistics:")
print(price_summary)

#-------------------------------- End of Milestone 1 --------------------------------

# Categorize the products using nltk, data encoding and clustering
# This function takes as input the dataframe and analyzes the content of the Description column by
# performing the following operations:
    # extract the names (proper, common) appearing in the products description
    # for each name, I extract the root of the word and aggregate the set of names associated with this particular root
    # count the number of times each root appears in the dataframe
    # when several words are listed for the same root, I consider that the keyword associated with this root is the shortest name (this systematically selects the singular when there are singular/plural variants)
is_noun = lambda string: string[:2] == 'NN'
def keywords_inventory(dataframe, column='Description'):
    stemmer = nltk.stem.SnowballStemmer(
        "english")  # creating an instance of the Snowball stemmer specifically configured for the English language, will be used to find the root form of words.
    keywords_roots = dict()  # store the root forms of words along with their associated words.
    keywords_select = dict()  # establish association: root <-> selected keyword
    category_keys = []  # list used to store selected keywords.
    count_keywords = {keyword: 0 for keyword in
                      category_keys}  # # dictionary used to keep track of the count of each keyword. It will store the number of times each root form appears in the data.Initialising the count as 0 for each keyword
    for s in dataframe[column]:  # for each element in the column of the dataframe
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(
            lines)  # This line tokenizes (splits) the text in lines into individual words, creating a list of tokens. This step is essential for further processing.
        # This following line uses nltk.pos_tag() to tag each token with its associated part of speech. It then filters the tokens to include only those identified as nouns by the is_noun function.
        nouns = [word for (word, tag) in nltk.pos_tag(tokenized) if is_noun(tag)]
        # This following code is building a dictionary (keywords_roots) where each key is a stemmed root form of a word, and the value associated with each key is a set of words that share that same root.
        # Additionally, count_keywords keeps track of how many times each root form appears in the data.
        for t in nouns:
            t = t.lower();  # ensuring consistency in word comparisons, as Python is case-sensitive.
            root = stemmer.stem(
                t)  # stems t using the Snowball stemmer initialized earlier. The stemmed form is assigned to root.
            # If root is already in keywords_roots, it means that this root has been encountered before. In that case:
            # keywords_roots[root].add(t): The original word t is added to the set of words associated with this root.
            # count_keywords[root] += 1: The count of this root is incremented to keep track of how many times it appears.
            if root in keywords_roots:
                keywords_roots[root].add(t)
                count_keywords[root] += 1
            # If root is not in keywords_roots, it means this is the first time this root is encountered. In that case:
            # keywords_roots[root] = {t}: A new set containing only t is created and associated with this root.
            # count_keywords[root] = 1: The count of this root is initialized to 1.
            else:
                keywords_roots[root] = {t}
                count_keywords[root] = 1
    # This following code is selecting the "representative" word for each root form, favoring the shortest word when there are multiple options.
    # The selected words are stored in category_keys, and the association between roots and selected words is stored in keywords_select
    for root in keywords_roots.keys():
        # if there is more than one associated word for the current root form (s)
        if len(keywords_roots[root]) > 1:
            min_length = 1000  # used to keep track of the shortest word associated with the root.
            for word in keywords_roots[root]:  # iterates over all the associated words for the current root.
                if len(word) < min_length:
                    shortest_word = word;
                    min_length = len(word)
            category_keys.append(
                shortest_word)  # at the end of iteration, the shortest word is added to the category_keys list
            keywords_select[root] = shortest_word
        else:
            category_keys.append(list(keywords_roots[root])[
                                     0])  # The first (and only) word associated with this root is added to the category_keys list.
            keywords_select[root] = list(keywords_roots[root])[0]
    print("Number of keywords in variable '{}': {}".format(column,
                                                           len(category_keys)))  # len(category_keys) gives the total number of keywords found in that column.
    return category_keys, keywords_roots, keywords_select, count_keywords

#STEP 1
# Plot the keywords vs number of occurences
category_keys, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_cleaned)
# Extracting keywords and occurrences
keywords = list(count_keywords.keys())
occurrences = list(count_keywords.values())
# Plotting the keywords vs. occurrences
plt.figure(figsize=(12, 6))
sns.barplot(x=keywords, y=occurrences, palette='viridis')
# Adding labels and title
plt.title('Keywords vs. Number of Occurrences')
plt.xlabel('Keywords')
plt.ylabel('Number of Occurrences')
# Rotating x-axis labels for better visibility (optional)
plt.xticks(rotation=90)
# Display the plot
plt.show()

#STEP 2
# Eliminate occurances below a certain threshold
# Performing one-hot encoding,
# One-hot encoding should not be used if variables take more than 15 different values, here the variable keyword takes 1483 values
# Hence eliminating some of the keywords below a certain threshold, i.e. ones with 10,000 occurences
threshold = 10000
filtered_keywords = {k: v for k, v in count_keywords.items() if v >= threshold}
print(f'Number of items in the filtered_keywords dictionary: {len(filtered_keywords)}')
# Encode product description with keywords and other relevant info, if any, in a matrix
for keyword in filtered_keywords.keys():
    df_cleaned[keyword] = df_cleaned['Description'].str.lower().str.contains(keyword).astype(int)
# df_cleaned.head(10)

#STEP 3
# Use kmeans to cluster products
# Normalize Price
# StandardScaler is a preprocessing step that scales the data so that it has a mean of 0 and a standard deviation of 1. This is important because it can make some algorithms (like K-means) work more effectively.
scaler = StandardScaler()
encoded_df['NormalizedPrice'] = scaler.fit_transform(encoded_df[['UnitPrice']]) #applies the scaling to the 'UnitPrice' column in encoded_df, and the scaled values are stored in a new column called 'NormalizedPrice' in encoded_df.
# Now 'encoded_df' contains the original data with normalized price
# encoded_df.head(10)
# for column_name, data_type in encoded_df.dtypes.items():
#     print(f"Column '{column_name}' has data type: {data_type}")
to_cluster_df = encoded_df.loc[:, 'heart':'NormalizedPrice']
# Deciding the best number of clusters using silhouette score as a criterion
# Defining a range of cluster numbers to test
cluster_range = range(2, 11)
best_silhouette_score = -1
best_num_clusters = 1
for num_clusters in cluster_range:
    # Applying K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(to_cluster_df)
    # Calculate silhouette score
    silhouette_avg = silhouette_score(to_cluster_df, clusters)
    # Update best silhouette score and number of clusters
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_num_clusters = num_clusters
print(f"The best number of clusters is: {best_num_clusters} with a silhouette score of {best_silhouette_score}")
# Applying K-means Clustering
kmeans = KMeans(n_clusters=best_num_clusters, random_state=0) #random_state=0 ensures that the results are reproducible.
clusters = kmeans.fit_predict(to_cluster_df) #fits the K-means model to the data and assigns each data point to a cluster. The resulting cluster labels are stored in the clusters variable.
to_cluster_df['ClusterLabel'] = clusters #cluster labels are added as a new column 'ClusterLabel' to the encoded_df DataFrame.

#STEP 4
#Plot results in a Word Cloud
# Converting dictionary to list of tuples
list_products = list(filtered_keywords.items())
print(list_products)