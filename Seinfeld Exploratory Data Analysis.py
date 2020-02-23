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

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
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


# In[21]:


# Load data and show the first 10 rows 
df = pd.read_csv('~/Desktop/scripts.csv', index_col=0)
df.head(10)


# We can tell that our data is structured in a way that every character's line represents a row (instance) and columns are features of this instance. 

# In[22]:


print(df.shape) #There are 6 columns and 54616 rows


# In[23]:


print(df.columns) #Column names


# 
# **Let's find _missing values_ and _clean our data_**

# In[24]:


# Information about our data frame, type of variables and missing values
# We can see that column Dialogue has missing values
df.info() 


# In[25]:


#find missing values in Dialogue column
df[df.isnull().any(axis=1)]


# We will now create a new data frame df and assign the old one to it. At the same time, we will drop all values in 10 rows which contain no dialogue. Lastly, we will check the info again to make sure all rows with missing values are deleted. 

# In[26]:


df = df.dropna(axis = 0)
df.info()


# In[27]:


#basic statistical characteristics of numerical features
df.describe() 


# Seinfeld series ran for 9 seasons. Let's check whether our dataset contains all seasons of the show.

# In[28]:


df['Season'].unique()


# We can also check the distribution of dialogue lines per season by plotting it.

# In[29]:


sns.set()
sns.distplot(df['Season'], kde=False, color = 'darkblue')


# **Season 7** has the biggest number of lines. **Season 1** has the smallest number but it is expected given the number of episodes per season. We can also check the number of episodes!

# In[30]:


episode_count = df.groupby('Season')['SEID'].aggregate(['count', 'unique'])
episode_count['episodes'] = episode_count['unique'].apply(lambda x: len(x))
episode_count[['count', 'episodes']]


# **Season 7** has the highest episode_count == 24 and **Season 1** the lowest == 4. However, if we look at Seinfeld's wikipedia page [https://en.wikipedia.org/wiki/Seinfeld#Episodes] we can notice that Season 1 has 5 episodes and not 4. By looking at the dataset again, we can notice that both Pilot episodes are merged and labeled as the same episode 'S01E01'. Now, we need to find the ending of the first pilot within S01E01 and divide them into two separate episodes. We can make our lives easier by dividing the total number of lines with the number of episodes and get an approximate line per episode measure.

# In[31]:


#Number of lines of dialogue per episode
episode_count['count']/episode_count['episodes']


# We can see that season 1 has approximately 327 lines per episode. It is obvious that we should look in the range of 250 to 350 lines in the first episode of the first season to find the ending of the first Pilot episode and the beginning of the second one which is nested in it.

# In[32]:


df[250:350] 


# Being a fan of the TV show, I notice that the second pilot has already started. We need to look in the range between 200 and 250.

# In[33]:


df[200:250]


# Jerry's monologue about not understanding women ends the first pilot at the line 210. Hence we need to label first 210 differently to separate the two pilot episodes.

# In[34]:


df[0:211]['SEID'] = 'Pilot 1' #Remember that 211 is not included in the range


# # Data Visualisation

# ## Plot number of lines per character
# 
# Define the color palette and assign it to variable col.
# 
# Define the function plot_lines which will plot the most frequent 9 characters in the TV show, by the number of lines. 

# In[35]:


col = sns.color_palette("Paired", 9)
def plot_lines(season = None, episode = None, top_n = 9):
    filtered_scripts = df
    if season:
        filtered_scripts = filtered_scripts[filtered_scripts['Season'] == season]
    if episode:
        filtered_scripts = filtered_scripts[filtered_scripts['SEID'] == episode]
    filtered_scripts['Character'].value_counts().head(top_n).plot(kind = 'bar', color=col, title = 'Number of lines per character', figsize=(16,10))


# In[36]:


sns.set()
plot_lines()


# ### We are going to zoom in on the main four characters using Bokeh interactive plotting

# In[37]:


line_count = df['Character'].value_counts().head(7)
line_count
characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN', 'MORTY', 'HELEN']
characters


# In[38]:


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

