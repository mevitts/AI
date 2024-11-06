import opendatasets as od
import pandas
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


od.download(
    "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

review_data = pd.read_csv("/content/imdb_movie_reviews_500.csv", encoding='latin-1')
print(review_data.head(5))

review_data = review_data[['Review', 'Sentiment']]
x = review_data['Review']
y = review_data['Sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.75)

class_distribution = review_data['Sentiment'].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
plt.title('Distribution of Spam and Ham Messages')
plt.show()

y_train.value_counts()

tfidf = TfidfVectorizer(stop_words='english', min_df=2, token_pattern=r'\b[A-Za-z]{2,}\b')
train_x_vec = tfidf.fit_transform(x_train)
test_x_vec = tfidf.transform(x_test)

pd.DataFrame.sparse.from_spmatrix(train_x_vec,
                                  index=x_train.index,
                                  columns=tfidf.get_feature_names_out())

mnb = MultinomialNB()
mnb.fit(train_x_vec, y_train)

y_pred = mnb.predict(test_x_vec)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='Positive')
f1_2 = f1_score(y_test, y_pred, pos_label='Negative')

print("Accuracy:", accuracy)
print("F1-score for 'positive' class:", f1)
print("F1-score for 'negative' class:", f1_2)
