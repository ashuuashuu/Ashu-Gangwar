#!/usr/bin/env python
# coding: utf-8

# ***Logistic Regression With words and char n-grams***

# In[34]:


import os
os.getcwd()


# In[35]:


os.chdir(r'C:\Users\Ashu Gangwar\Desktop\NLP Pyhton')


# In[36]:


import os
os.getcwd()


# In[37]:


train_text


# In[38]:


test_text


# In[39]:


all_text


# In[40]:


type(all_text)


# In[41]:


len(all_text)


# In[42]:


word_vectorizer


# In[43]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv('C:/Users/Ashu Gangwar/Desktop/NLP Pyhton/train.csv').fillna(' ')
test = pd.read_csv('C:/Users/Ashu Gangwar/Desktop/NLP Pyhton/test.csv', encoding='latin1').fillna(' ')
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)

# sublinear_tf: bool, default=False
# Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)

# strip_accents{‘ascii’, ‘unicode’}, default=None
# Remove accents and perform other character normalization during the preprocessing step. ‘ascii’ is a fast method
# that only works on characters that have an direct ASCII mapping. ‘unicode’ is a slightly slower method that works on
# any characters. None (default) does nothing.

# analyzer: {‘word’, ‘char’, ‘char_wb’} or callable, default=’word’
# Whether the feature should be made of word or character n-grams. Option ‘char_wb’ creates character n-grams only from 
# text inside word boundaries; n-grams at the edges of words are padded with space.

# token_pattern: str
# Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'. The default regexp selects tokens
# of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).

# ngram_range: tuple (min_n, max_n), default=(1, 1)
# The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that
# min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and
# bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.


# In[88]:


print(train_word_features[110009])


# In[54]:


print(train_word_features[0])


# In[51]:


train_word_features.shape


# In[108]:


print(train_word_features[:15999])


# In[101]:


type(train_word_features)


# In[44]:


word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[128]:


print(train_word_features[10,111:333])


# In[134]:


# train_text
# print(train_word_features)
print(train_word_features[10:333,333:444])


# In[135]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)


# In[139]:


# print(train_char_features)
print(train_char_features.shape)


# In[144]:


print(train_char_features[2:45, 3:44])


# In[137]:


type(train_char_features)


# In[136]:


train_char_features


# In[26]:


char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# In[28]:


print(train_char_features)
print(test_char_features)


# In[29]:


train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


# In[30]:


scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})


# In[31]:


submission


# In[33]:


for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]


# In[32]:


class_names


# In[155]:


print('Total CV score is {}'.format(np.mean(scores)))


# In[156]:


submission.to_csv('submission.csv', index=False)


# ***NB-SVM strong linear baseline***

# In[5]:


import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#library TfidefVectorizer is used for 


# In[ ]:




