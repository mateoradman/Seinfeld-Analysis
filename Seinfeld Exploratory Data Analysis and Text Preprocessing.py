#!/usr/bin/env python
# coding: utf-8

# # Dialogues in the comedy TV show Seinfeld: A show about nothing!

# **This project analyses my favourite comedy TV show Seinfeld. I created this project to practice exploratory data analysis, visualization and natural language processing of dialogues in Seinfeld.**

# The data consists of 54616 rows and 6 columns.
# Column information:
# 1. Unnamed: 0 - index column
# 2. Character - name of the character in the TV show
# 3. Dialogue - quotes from show's dialogues
# 4. EpisodeNo - episode number
# 5. SEID - Season and episode ID (i.e. S01E01 denotes 1st episode of the 1st season)
# 6. Season - number of show's season

# ### Load packages required for the project

# In[78]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import re
import nltk
import spacy
import string
pd.options.mode.chained_assignment = None
import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
import warnings #Ignore warnings
warnings.filterwarnings('ignore')
#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[79]:


# Load data and show the first 10 rows 
df = pd.read_csv('~/Desktop/scripts.csv', index_col=0)
df.head(10)


# We can tell that our data is structured in a way that every character's line represents a row (instance) and columns are features of this instance. 

# In[80]:


print(df.shape) #There are 6 columns and 54616 rows


# In[81]:


print(df.columns) #Column names


# 
# **Let's find _missing values_ and _clean our data_**

# In[82]:


# Information about our data frame, type of variables and missing values
# We can see that column Dialogue has missing values
df.info() 


# In[83]:


#find missing values in Dialogue column
df[df.isnull().any(axis=1)]


# We will now create a new data frame df and assign the old one to it. At the same time, we will drop all values in 10 rows which contain no dialogue. Lastly, we will check the info again to make sure all rows with missing values are deleted. 

# In[84]:


df = df.dropna(axis = 0)
df.info()


# In[85]:


#basic statistical characteristics of numerical features
df.describe() 


# Seinfeld series ran for 9 seasons. Let's check whether our dataset contains all seasons of the show.

# In[86]:


df['Season'].unique()


# In[87]:


help(sns.distplot)


# We can also check the distribution of dialogue lines per season by plotting it.

# In[88]:


sns.set()
sns.distplot(df['Season'], kde=False, color = 'darkblue')


# **Season 7** has the biggest number of lines. **Season 1** has the smallest number but it is expected given the number of episodes per season. We can also check the number of episodes!

# In[89]:


episode_count = df.groupby('Season')['SEID'].aggregate(['count', 'unique'])
episode_count['episodes'] = episode_count['unique'].apply(lambda x: len(x))
episode_count[['count', 'episodes']]


# **Season 7** has the highest episode_count == 24 and **Season 1** the lowest == 4. However, if we look at Seinfeld's wikipedia page [https://en.wikipedia.org/wiki/Seinfeld#Episodes] we can notice that Season 1 has 5 episodes and not 4. By looking at the dataset again, we can notice that both Pilot episodes are merged and labeled as the same episode 'S01E01'. Now, we need to find the ending of the first pilot within S01E01 and divide them into two separate episodes. We can make our lives easier by dividing the total number of lines with the number of episodes and get an approximate line per episode measure.

# In[90]:


#Number of lines of dialogue per episode
episode_count['count']/episode_count['episodes']


# We can see that season 1 has approximately 327 lines per episode. It is obvious that we should look in the range of 250 to 350 lines in the first episode of the first season to find the ending of the first Pilot episode and the beginning of the second one which is nested in it.

# In[91]:


df[250:350] 


# Being a fan of the TV show, I notice that the second pilot has already started. We need to look in the range between 200 and 250.

# In[92]:


df[200:250]


# Jerry's monologue about not understanding women ends the first pilot at the line 210. Hence we need to label first 210 differently to separate the two pilot episodes.

# In[93]:


df[0:211]['SEID'] = 'Pilot 1' #Remember that 211 is not included in the range


# # Data Visualisation

# ## Plot number of lines per character
# 
# Define the color palette and assign it to variable col.
# 
# Define the function plot_lines which will plot the most frequent 9 characters in the TV show, by the number of lines. 

# In[94]:


col = sns.color_palette("Paired", 9)
def plot_lines(season = None, episode = None, top_n = 9):
    filtered_scripts = df
    if season:
        filtered_scripts = filtered_scripts[filtered_scripts['Season'] == season]
    if episode:
        filtered_scripts = filtered_scripts[filtered_scripts['SEID'] == episode]
    filtered_scripts['Character'].value_counts().head(top_n).plot(kind = 'bar', color=col, title = 'Number of lines per character', figsize=(16,10))


# In[95]:


sns.set()
plot_lines()


# ### We are going to zoom in on the main four characters using Bokeh interactive plotting

# In[96]:


line_count = df['Character'].value_counts().head(7)
line_count
characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN', 'MORTY', 'HELEN']
characters


# In[97]:


from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.palettes import brewer
colors = brewer["Dark2"][len(characters)]

output_notebook()

p = figure(x_range=characters, plot_height=600, plot_width=700,
           title="Number of lines in the script per main character", toolbar_location=None, tools="")

p.vbar(x=characters, top=line_count, width=0.9, color = colors)

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)


# # Text preprocessing 

# In[98]:


#Make all strings in the column Dialogue lowercase 
df["text_lower"] = df["Dialogue"].str.lower()
df.head()


# ### We can also drop all the punctuation and stopwords
# Stopwords are words in English language that carry no meaning for the computer, but are essential for human interaction. The next cell will list some stopwords.

# In[99]:


#remove punctuation
import string 

punctuation = string.punctuation
def remove_punctuation(text):
    """function to remove the punctuation"""
    return text.translate(str.maketrans('', '', punctuation))

df["no_punctuation"] = df["text_lower"].apply(lambda text: remove_punctuation(text))
df.head()


# In[100]:


#Note: you might need to download some packages from nltk, otherwise you'll get an error here
#If you get an error here, uncomment the following two lines to download it
#import nltk
#nltk.download()
from nltk.corpus import stopwords

", ".join(stopwords.words('english'))


# In[101]:


#Now we remove the stopwords
stopwords = set(stopwords.words('english'))
def remove_stopwords(text):
    """function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in stopwords])

df["no_stop_words"] = df["no_punctuation"].apply(lambda text: remove_stopwords(text))
df.head()


# ### Find the most common words in the Seinfeld script
# The cell below imports Counter and looks for the most frequent words

# In[102]:


from collections import Counter
cnt = Counter()
for text in df["no_stop_words"].values:
    for word in text.split():
        cnt[word] += 1
        
most_common_words = cnt.most_common(10)
most_common_words


# In[103]:


#Let's plot the most common words
c=dict(most_common_words)
key=list(c.keys())
value=list(c.values())

output_notebook()

p = figure(x_range=key, plot_height=550, plot_width=550,
           title="The most frequent words in the script", toolbar_location=None, tools="")

p.vbar(x=key, top=value, width=0.9, color = ['blue', 'red', 'green', 'wheat', 'yellow', 'magenta', 'black', 'navy', 'lightblue', 'silver'])

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)


# In[104]:


#Let's find the most common 10 words and remove them

frequentwords = set([w for (w, wc) in cnt.most_common(10)])
print(frequentwords)
def remove_freqwords(text):
    """remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in frequentwords])

df["no_stopwords_reduced"] = df["no_stop_words"].apply(lambda text: remove_freqwords(text))
df.head()


# In[105]:


#We now find the 10 least used words and remove them
n_rare_words = 10
rarewords = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
print(rarewords)
def remove_rarewords(text):
    """remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in rarewords])

df["no_stopwords_extra_reduced"] = df["no_stopwords_reduced"].apply(lambda text: remove_rarewords(text))
df.head()


# ### Stemming
# Stemming is a process of reducing the word to their root. For instance consulting, consultant, consultants and consulted all have the same stem 'consult'. It makes analysis much more clear and easier to keep track of many variations of the same word. 

# In[106]:


from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["text_stemmed"] = df["Dialogue"].apply(lambda text: stem_words(text))
df.head()


# ### Lemmatization
# You may notice that the process of stemming made some mistakes. Instead of try on, it returned tri. Instead of was it returned wa, and so on. We can use lemmatization instead of stemming, which reduces the words to their 'lemma'. 

# In[107]:


from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["final_lemmatized"] = df["no_stopwords_extra_reduced"].apply(lambda text: lemmatize_words(text))
df.head()


# In[108]:


# Drop all unnecessary columns
df.drop(['no_punctuation', 'text_lower', 'no_stop_words', 
         'no_stopwords_reduced', 'no_stopwords_extra_reduced', 'text_stemmed'], axis = 1)
#We are left with final_lemmatized column at the end


# **Let's find the most common words out of all lemmatized words**

# In[109]:


from collections import Counter
cnt = Counter()
for text in df["final_lemmatized"].values:
    for word in text.split():
        cnt[word] += 1
        
most_common_lemmatized_words = cnt.most_common(12)
most_common_lemmatized_words


# In[110]:


#Let's plot the most common words now
c=dict(most_common_lemmatized_words)
key_lemma=list(c.keys())
value_lemma=list(c.values())

colors = brewer["Paired"][len(key_lemma)]
output_notebook()

p = figure(x_range=key_lemma, plot_height=550, plot_width=750,
           title="The 20 most frequent lemmatized words in the script", toolbar_location=None, tools="")

p.vbar(x=key_lemma, top=value_lemma, width=0.9, color = colors)

p.xgrid.grid_line_color = 'lightgrey'
p.y_range.start = 1000


show(p)


# **We can also create a dendrogram!** 
# 
# Let's use plotly this time.

# In[111]:


import plotly as py
import plotly.figure_factory as ff
X = np.random.rand(10, 10)
fig = ff.create_dendrogram(X, orientation='bottom', 
labels=key_lemma)
fig['layout'].update({'width': 800, 'height': 800})
fig.show()


# In[ ]:




