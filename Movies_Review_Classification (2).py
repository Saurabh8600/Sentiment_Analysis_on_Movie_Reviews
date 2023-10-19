#!/usr/bin/env python
# coding: utf-8

# # SENTIMENT ANALYSIS ON MOVIE REVIEWS USING NLP

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
nltk.download()
import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # LOADING THE DATA

# In[2]:


df = pd.read_csv("IMDB Dataset.csv")
df


# # TOP 5 ROWS

# In[3]:


df.head()


# # EDA

# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


df.info()


# In[7]:


df.nunique()


# In[8]:


df["sentiment"].value_counts()


# In[9]:


sns.countplot(x="sentiment", data = df)
plt.title("Sentiment Distribution")


# In[10]:


for i in range(5):
    print("Review:", [i])
    print(df["review"].iloc[i], "\n")
    print("Sentiment:", df["sentiment"].iloc[i], "\n\n\n")


# In[11]:


def no_of_words(text):
    words = text.split()
    word_count = len(words)
    return word_count


# In[12]:


df["word count"] = df["review"].apply(no_of_words)


# In[13]:


df.head()


# In[14]:


fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(df[df["sentiment"] == "positive"]["word count"], label="Positive" , color= "red", rwidth=0.9);
ax[0].legend(loc="upper right");
ax[1].hist(df[df["sentiment"] == "negative"]["word count"], label="Negative" , color= "blue", rwidth=0.9);
ax[1].legend(loc="upper right");
fig.suptitle("Number of Words in Review")
plt.show()


# In[15]:


fig, ax = plt.subplots(1,2, figsize=(11,5))
ax[0].hist(df[df["sentiment"] == "positive"]["review"].str.len(), label="Positive" , color= "red", rwidth=0.9);
ax[0].legend(loc="upper right");
ax[1].hist(df[df["sentiment"] == "negative"]["review"].str.len(), label="Negative" , color= "blue", rwidth=0.9);
ax[1].legend(loc="upper right");
fig.suptitle("Length of Reviews")
plt.show()


# In[16]:


df.sentiment.replace("positive", 1,inplace= True)
df.sentiment.replace("negative", 2,inplace= True)


# In[17]:


df.head()


# In[18]:


def data_processing(text):
    text = text.lower()
    text = re.sub("<br />", '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '',text, flags = re.MULTILINE)
    text = re.sub(r"\@w+|\#", '', text)
    text = re.sub(r"[^\w\s]", '',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[19]:


df.review = df["review"].apply(data_processing)


# In[20]:


duplicate_count = df.duplicated().sum()
print("Number of Duplicate Entries:", duplicate_count)


# In[21]:


df = df.drop_duplicates("review")


# In[22]:


stemmer = PorterStemmer()
def stemming(data):
    text = (stemmer.stem(word) for word in data)
    return data


# In[23]:


df.review = df["review"].apply(lambda x: stemming(x))


# In[24]:


df["word count"] = df["review"].apply(no_of_words)


# In[25]:


df.head()


# In[26]:


pos_reviews = df[df.sentiment == 1]
pos_reviews.head()


# In[27]:


text = " ".join([word for word in pos_reviews["review"]])
plt.figure(figsize=(20,15), facecolor="None")
wordcloud = WordCloud(max_words= 500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.title("Most Frequent Words used in Positive Review", fontsize = 19)
plt.show()


# In[28]:


from collections import Counter
count = Counter()
for text in pos_reviews["review"].values:
    for word in text.split():
        count[word]+=1
count.most_common(15)


# In[29]:


pos_words = pd.DataFrame(count.most_common(15))
pos_words.columns = ["word", "count"]
pos_words.head()


# In[30]:


px.bar(pos_words, x="count", y="word", title="Most Common Words in Positive Review", color="word")


# In[31]:


neg_reviews = df[df.sentiment == 2]
neg_reviews.head()


# In[32]:


text = " ".join([word for word in neg_reviews["review"]])
plt.figure(figsize=(20,15), facecolor="None")
wordcloud = WordCloud(max_words= 500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.title("Most Frequent Words used in Negative Review", fontsize = 19)
plt.show()


# In[33]:


from collections import Counter
count = Counter()
for text in neg_reviews["review"].values:
    for word in text.split():
        count[word]+=1
count.most_common(15)


# In[34]:


neg_words = pd.DataFrame(count.most_common(15))
neg_words.columns = ["word", "count"]
neg_words.head()


# In[35]:


px.bar(neg_words, x="count", y="word", title="Most Common Words in Negative Review", color="word")


# # ASSIGNING INDEPENDENT VARIABLE AND DEPENDENT VARIABLE

# In[36]:


X=df["review"]
y=df["sentiment"]


# In[37]:


vect = TfidfVectorizer()
X = vect.fit_transform(df["review"])


# # TRAIN-TEST SPLIT

# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=42) 


# In[39]:


print("Size of X_train:", (X_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of X_test:", (X_test.shape))
print("Size of y_test:", (y_test.shape))


# # IMPORTING REQ LIBRARIES FOR MACHINE LEARNING

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


# # LOGISTIC REGRESSION

# In[41]:


lr = LogisticRegression()
lr.fit(X_train,y_train)

#Predicting test result

lr_pred=lr.predict(X_test)
accuracy = accuracy_score(lr_pred,y_test)

# Convert the accuracy to a percentage
accuracy_percentage = accuracy * 100

print("Accuracy Score: {:.2f}%".format(accuracy_percentage))


# In[42]:


confusion_matrix(y_test,lr_pred)


# In[43]:


print(classification_report(y_test,lr_pred))


# # MULTINOMIAL NAIVE BAYES

# In[44]:


mnb = MultinomialNB()
mnb.fit(X_train,y_train)

#Predicting test result

mnb_pred=mnb.predict(X_test)
accuracy = accuracy_score(mnb_pred,y_test)

# Convert the accuracy to a percentage
accuracy_percentage = accuracy * 100

print("Accuracy Score: {:.2f}%".format(accuracy_percentage))


# In[45]:


confusion_matrix(y_test,mnb_pred)


# In[46]:


print(classification_report(y_test,mnb_pred))


# # SVM

# In[47]:


svc = LinearSVC()
svc.fit(X_train,y_train)

#Predicting test result

svc_pred=svc.predict(X_test)
accuracy = accuracy_score(svc_pred,y_test)

# Convert the accuracy to a percentage
accuracy_percentage = accuracy * 100

print("Accuracy Score: {:.2f}%".format(accuracy_percentage))


# In[48]:


confusion_matrix(y_test,svc_pred)


# In[49]:


print(classification_report(y_test,svc_pred))


# # HYPER-PARAMETER TUNING FOR SVM

# In[50]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100], 'loss':["hinge","squared_hinge"]}
grid = GridSearchCV(svc,param_grid, refit=True, verbose=3 )
grid.fit(X_train, y_train)


# In[51]:


print("Best Cross Validation Score: {:.2f}".format(grid.best_score_))
print("Best Parameters:", grid.best_params_)


# In[52]:


svc = LinearSVC(C = 1, loss= "hinge")
svc.fit(X_train,y_train)

#Predicting test result

svc_pred=svc.predict(X_test)
accuracy = accuracy_score(svc_pred,y_test)

# Convert the accuracy to a percentage
accuracy_percentage = accuracy * 100

print("Accuracy Score: {:.2f}%".format(accuracy_percentage))


# In[53]:


confusion_matrix(y_test,svc_pred)


# In[54]:


print(classification_report(y_test,svc_pred))


# # ACTUAL VS PREDICTED VALUES BY SVM

# In[55]:


pd.DataFrame(np.c_[y_test,svc_pred], columns= ["Actual", "Predicted"])


# In[56]:


pickle.dump(vect, open("count-Vectorizer.pkl", "wb"))
pickle.dump(svc, open("Movies_Review_Classification.pkl", "wb"))


# In[57]:


save_vect = pickle.load(open("count-Vectorizer.pkl", "rb"))
model =  pickle.load(open("Movies_Review_Classification.pkl", "rb"))


# # TESTING RANDOM SENTENCES

# In[58]:


def test_model(sentence):
    sen = save_vect.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        return 'Positive review'
    else:
        return 'Negative review'


# In[59]:


sen = "This is the wonderful movie of my life."
res = test_model(sen)
print(res)


# In[60]:


sen = "This is the worst movie of my life."
res = test_model(sen)
print(res)


# In[61]:


sen = "I didn't like this movie."
res = test_model(sen)
print(res)


# In[62]:


sen = "This movie was so amazing."
res = test_model(sen)
print(res)

