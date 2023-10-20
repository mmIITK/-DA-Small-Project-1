

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

## 1. Data preparation

#__________________
# read the datafile ( Sample code below )
df_initial = pd.read_csv('data.csv',encoding="ISO-8859-1",
                         dtype={'CustomerID': str,'InvoiceID': str})
print('Dataframe dimensions:', df_initial.shape)

# This dataframe contains 8 variables that correspond to:

# **InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.  <br>
# **StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. <br>
# **Description**: Product (item) name. Nominal. <br>
# **Quantity**: The quantities of each product (item) per transaction. Numeric.	<br>
# **InvoiceDate**: Invoice Date and time. Numeric, the day and time when each transaction was generated. <br>
# **UnitPrice**: Unit price. Numeric, Product price per unit in sterling. <br>
# **CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. <br>
# **Country**: Country name. Nominal, the name of the country where each customer resides.<br>"""



#_______STEP 1___________
# show first 10 lines and data frame info
# Help - Dataframe info method - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html
# Help - Print first n records - https://www.geeksforgeeks.org/get-first-n-records-of-a-pandas-dataframe/


#________STEP 2__________
# Drop null values (For ex : There are rows with unassigned customers).
# Show no.of rows before and after the action.
# Help - dropna method - https://www.w3schools.com/python/pandas/ref_df_dropna.asp

#________STEP 3__________
# Drop Duplicate Rows.
# Show no.of rows before and after the action.
# Help - drop_duplicates method - https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/

#________STEP 4__________
# Next, we explore each column to the dataframe
# To analyze the country column, we create a choropleth map for Countries
### Countries

# Countries where orders are made ( Sample code below )
temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()

# Plot Choropleth World Map

data = dict(type='choropleth',
locations = countries.index,
locationmode = 'country names', z = countries,
text = countries.index, colorbar = {'title':'Order nb.'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],
reversescale = False)
#_______________________
layout = dict(title='Number of orders per country',
geo = dict(showframe = True, projection={'type':'mercator'}))
#______________
choromap = go.Figure(data = [data], layout = layout)
py.plot(choromap, validate=False)

#____________STEP 5__________________________
# Likewise explore the Cancelling Orders, Stock Code and price.
# Try to understand the patterns in each column
# Make any dataframe edits necessary
# Give short comments/supporting visualizations/graphs wherever relevant

#-------------------------------- End of Milestone 1 --------------------------------

# Categorize the products using nltk, data encoding and clustering
# Sample code for keyword extraction below
#This function takes as input the dataframe and analyzes the content of the Description column by
# performing the following operations:
    #extract the names (proper, common) appearing in the products description
    #for each name, I extract the root of the word and aggregate the set of names associated with this particular root
    #count the number of times each root appears in the dataframe
    #when several words are listed for the same root, I consider that the keyword associated with this root is the shortest name (this systematically selects the singular when there are singular/plural variants)

is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, colonne='Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    count_keywords = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

        for t in nouns:
            t = t.lower();
            racine = stemmer.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k;
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("Nb of keywords in variable '{}': {}".format(colonne, len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords

#____________STEP 1__________________________
# Plot the keywords vs number of occurences
# Help - matplotlib function : https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm

#____________STEP 2_________________________
# Eliminate occurances below a certain threshold
# Encode product description with keywords and other relevant info, if any, in a matrix
# Help - One-hot encoding : https://www.kaggle.com/code/dansbecker/using-categorical-data-with-one-hot-encoding

#____________STEP 3_________________________
# Use kmeans to cluster products
# Help - K-means with one-hot : https://medium.com/analytics-vidhya/clustering-on-mixed-data-types-in-python-7c22b3898086#:~:text=Method%202%3A%20K%2DMeans%20with%20One%20Hot%20Encoding&text=One%20hot%20encoding%20involves%20creating,the%20pandas%20%E2%80%9Cget_dummies%E2%80%9D%20function.
# Help - Calculating silhoutte score : https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c

#____________STEP 4__________________________
#Plot results in a Word Cloud ( Sample code below )

liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))

#________________________________________________________________________
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#________________________________________________________________________
def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey',
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))

fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)
#-------------------------------- End of Milestone 2 --------------------------------

