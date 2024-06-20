import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

spam = pd.read_csv('spam.csv', encoding='ISO-8859-1')




##    STEPS    ##
# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy

##     Cleaning     ##
spam.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

#rename columns to make it easy to understand
spam.rename(columns={'v1':'target','v2':'text'},inplace=True)


# assign ham to 0 and spam to 1
encoder = LabelEncoder()
spam['target']=encoder.fit_transform(spam['target'])


spam=spam.drop_duplicates(keep='first')

###    EDA    ###

# spam['target'].value_counts()
# plt.pie(spam['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
# plt.show() 

##88% of emails or not spam and remaining spam, hence data is imbalaced

###   Get lenght of all sms, num words and num sentences
spam['num_char']=spam['text'].apply(len)
# print(spam.head())



spam['num_words']=spam['text'].apply(lambda x:len(nltk.word_tokenize(x)))
# print(spam.head())

spam['num_sentences']=spam['text'].apply(lambda x:len(nltk.sent_tokenize(x)))



###ham
# spam[spam['target']==0][['num_char','num_words','num_sentences']].describe()

###spam
# spam[spam['target']==1][['num_char','num_words','num_sentences']].describe()

##plot the comparison
# plt.figure(figsize=(12,6))
# sns.histplot(spam[spam['target'] == 0]['num_char'])
# sns.histplot(spam[spam['target'] == 1]['num_char'],color='red')


###       TEXT PREPROSESSING     ###

#Lower case
# Tokenization [make list]
# Removing special characters
# Removing stop words and punctuation
# Stemming
ps = PorterStemmer()

def process_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


spam['transformed_text']=spam['text'].apply(process_text)
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(spam[spam['target'] == 1]['transformed_text'].str.cat(sep=" "))
ham_wc = wc.generate(spam[spam['target'] == 0]['transformed_text'].str.cat(sep=" "))

##get top used words from spam and ham

spam_words = []
for msg in spam[spam['target'] == 1]['transformed_text'].tolist():
  for word in msg.split():
     spam_words.append(word)

ham_corpus = []
for msg in spam[spam['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
# sns.barplot(pd.DataFrame(Counter(spam_words).most_common(30))[0],pd.DataFrame(Counter(spam_words).most_common(30))[1])

###we can repeat the same for ham aswell

####      MODEL BUILDING       ####

X=tfidf.fit_transform(spam['transformed_text']).toarray()
y=spam['target'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

# gnb.fit(X_train,y_train)
# y_pred1=gnb.predict(X_test)
# print(accuracy_score(y_test,y_pred1))
# print(confusion_matrix(y_test,y_pred1))
# print(precision_score(y_test,y_pred1))

##this model precision score is 0.53 which is very low

mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

## precision score is 0.83 , better but not enough

# bnb.fit(X_train,y_train)
# y_pred3=bnb.predict(X_test)
# print(accuracy_score(y_test,y_pred3))
# print(confusion_matrix(y_test,y_pred3))
# print(precision_score(y_test,y_pred3))

## this has 0.97 precision score and accuracy is 0.97 as well
### on using tfidf mnb is more precise with a score of 1.0

# import pickle
# pickle.dump(tfidf,open('vect.pkl','wb'))
# pickle.dump(mnb,open('model.pkl','wb'))
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# Check class distribution
print(spam['target'].value_counts()) 
# Evaluate model
accuracy = accuracy_score(y_test, y_pred2)
conf_matrix = confusion_matrix(y_test, y_pred2)
precision = precision_score(y_test, y_pred2)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision}")



import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))