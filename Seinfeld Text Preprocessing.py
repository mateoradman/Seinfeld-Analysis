#!/usr/bin/env python
# coding: utf-8

# # Seinfeld text preprocessing 

# In[27]:


import pandas as pd
import numpy as np
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.palettes import brewer
import plotly as py
import plotly.figure_factory as ff
import nltk
import re
import spacy
import string
import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
import warnings #Ignore warnings
warnings.filterwarnings('ignore')


# In[9]:


# Load data and show the first 10 rows 
df = pd.read_csv('~/Desktop/scripts.csv', index_col=0)
df.head(10)


# In[10]:


#Make all strings in the column Dialogue lowercase 
df["text_lower"] = df["Dialogue"].str.lower()


# ### We can also drop all the punctuation and stopwords
# Stopwords are words in English language that carry no meaning for the computer, but are essential for human interaction. The next cell will list some stopwords.

# In[12]:


#remove punctuation
punctuation = string.punctuation
def remove_punctuation(text):
    """function to remove the punctuation"""
    return text.translate(str.maketrans('', '', punctuation))

df["no_punctuation"] = df["text_lower"].astype(str).apply(lambda text: remove_punctuation(text))
df.head(3)


# In[13]:


#Note: you might need to download some packages from nltk, otherwise you'll get an error here
#If you get an error here, uncomment the following two lines to download it
#import nltk
#nltk.download()
from nltk.corpus import stopwords

", ".join(stopwords.words('english'))


# In[14]:


#Now we remove the stopwords
stopwords = set(stopwords.words('english'))
def remove_stopwords(text):
    """function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in stopwords])

df["no_stop_words"] = df["no_punctuation"].apply(lambda text: remove_stopwords(text))
df.head()


# ### Find the most common words in the Seinfeld script
# The cell below imports Counter and looks for the most frequent words

# In[15]:


from collections import Counter
cnt = Counter()
for text in df["no_stop_words"].values:
    for word in text.split():
        cnt[word] += 1
        
most_common_words = cnt.most_common(10)
most_common_words


# In[16]:


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


# In[17]:


#Let's find the most common 10 words and remove them

frequentwords = set([w for (w, wc) in cnt.most_common(10)])
print(frequentwords)
def remove_freqwords(text):
    """remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in frequentwords])

df["no_stopwords_reduced"] = df["no_stop_words"].apply(lambda text: remove_freqwords(text))
df.head()


# In[18]:


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

# In[21]:


from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["text_stemmed"] = df["Dialogue"].astype(str).apply(lambda text: stem_words(text))
df.head()


# ### Lemmatization
# You may notice that the process of stemming made some mistakes. Instead of try on, it returned tri. Instead of was it returned wa, and so on. We can use lemmatization instead of stemming, which reduces the words to their 'lemma'. 

# In[22]:


from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["final_lemmatized"] = df["no_stopwords_extra_reduced"].apply(lambda text: lemmatize_words(text))
df.head()


# In[23]:


# Drop all unnecessary columns
df.drop(['no_punctuation', 'text_lower', 'no_stop_words', 
         'no_stopwords_reduced', 'no_stopwords_extra_reduced', 
         'text_stemmed'], axis = 1)
#We are left with final_lemmatized column at the end


# **Let's find the most common words out of all lemmatized words**

# In[24]:


cnt = Counter()
for text in df["final_lemmatized"].values:
    for word in text.split():
        cnt[word] += 1
        
most_common_lemmatized_words = cnt.most_common(12)
most_common_lemmatized_words


# In[25]:


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

# In[28]:


X = np.random.rand(10, 10)
fig = ff.create_dendrogram(X, orientation='bottom', 
labels=key_lemma)
fig['layout'].update({'width': 800, 'height': 800})
fig.show()


# In[ ]:




