#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ## Loading Dataset

# In[2]:


df=pd.read_csv(r"C:\Users\User\Documents\Projects\Recommendation Projects\Restaurant Recommendation\zomato.csv")
df


# ## Data Cleaning and Feature Engineering

# ### Deleting unnecessary columns 

# #### Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"

# In[3]:


zomato=df.drop(['url','dish_liked','phone'],axis=1)


# #### Removing the Duplicates

# In[4]:


zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)


# ### Finding Null Values

# In[5]:


print(zomato.isna().sum())


# ### Treating Null Values

# #### Remove the NaN values from the dataset

# In[6]:


zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)


# ## Checking Null Values

# In[7]:


print(zomato.isna().sum())


# ### Changing the column names

# In[8]:


zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})


# ### Transformations

# #### Changing the cost to string

# In[9]:


zomato['cost'] = zomato['cost'].astype(str)


# #### Using lambda function to replace ',' from cost

# In[10]:


zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) 


# In[11]:


zomato['cost'] = zomato['cost'].astype(float)


# #### Removing '/5' from Rates

# In[12]:


zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')


# #### Adjust the column names

# In[13]:


zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)


# #### Computing Mean Rating

# In[14]:


restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)


# ## Text Preprocessing

# #### Lower Casing

# In[15]:


zomato["reviews_list"] = zomato["reviews_list"].str.lower()


# #### Removal of Puctuations

# In[16]:


import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))


# #### Removal of Stopwords

# In[17]:


import nltk
nltk.download('stopwords')


# In[18]:


from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))


# #### Removal of URLS 

# In[19]:


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)


# #### RESTAURANT NAMES:

# In[20]:


restaurant_names = list(zomato['name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]
    
zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)
import pandas


# #### Randomly sample 60% of your dataframe

# In[21]:


df_percent = zomato.sample(frac=0.5)


# ## TF-IDF Vectorization

# In[22]:


df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)


# #### Creating tf-idf matrix

# In[23]:


tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[24]:


def recommend(name, cosine_similarities = cosine_similarities):
    
    # Creating a list to put top restaurants
    recommend_restaurant = []
    
    # Finding the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Finding the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extracring top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
        
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Creating the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Dropingg the same named restaurants and sorted only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new
recommend('Pai Vihar')    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




