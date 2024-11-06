import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

churn_df = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')

x = churn_df.drop('Churn', axis=1)
y = churn_df['Churn']

frequency_imp = SimpleImputer(strategy='most_frequent')
fill_imp = SimpleImputer(strategy='constant')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=24)

#most frequent value for each feature
x_train1 = frequency_imp.fit_transform(x_train)
x_test1 = frequency_imp.transform(x_test)
#fill in default value
x_train2 = fill_imp.fit_transform(x_train)
x_test2 = fill_imp.transform(x_test)

#one-hot encoding
x_train1 = pd.get_dummies(pd.DataFrame(x_train1, columns=x.columns))
x_test1 = pd.get_dummies(pd.DataFrame(x_test1, columns=x.columns))
x_train2 = pd.get_dummies(pd.DataFrame(x_train2, columns=x.columns))
x_test2 = pd.get_dummies(pd.DataFrame(x_test2, columns=x.columns))


x_train1, x_test1 = x_train1.align(x_test1, join='left', axis=1, fill_value=0)
x_train2, x_test2 = x_train2.align(x_test2, join='left', axis=1, fill_value=0)

#standardizing
scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)
x_train2 = scaler.fit_transform(x_train2)
x_test2 = scaler.transform(x_test2)

#NB
gnb = GaussianNB()
gnb2 = GaussianNB()

gnb.fit(x_train1, y_train)
gnb2.fit(x_train2, y_train)

y_pred1 = gnb.predict(x_test1)
y_pred2 = gnb2.predict(x_test2)

#LR
reg = LogisticRegression()
reg2 = LogisticRegression()

reg.fit(x_train1, y_train)
reg2.fit(x_train2, y_train)

reg_y_pred1 = reg.predict(x_test1)
reg_y_pred2 = reg2.predict(x_test2)

#accuracy
accuracy = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy3 = accuracy_score(y_test, reg_y_pred1)
accuracy4 = accuracy_score(y_test, reg_y_pred2)
print(accuracy, accuracy2, accuracy3, accuracy4)
#f1
f1 = f1_score(y_test, y_pred1, pos_label='Yes')
f1_2 = f1_score(y_test, y_pred2, pos_label='Yes')
f1_3 = f1_score(y_test, reg_y_pred1, pos_label='Yes')
f1_4 = f1_score(y_test, reg_y_pred2, pos_label='Yes')
print(f1, f1_2, f1_3, f1_4)
#precision
prec = precision_score(y_test, y_pred1, pos_label='Yes')
prec2 = precision_score(y_test, y_pred2, pos_label='Yes')
prec3 = precision_score(y_test, reg_y_pred1, pos_label='Yes')
prec4 = precision_score(y_test, reg_y_pred2, pos_label='Yes')
print(prec, prec2, prec3, prec4)

#recall
recall = recall_score(y_test, y_pred1, pos_label='Yes')
recall2 = recall_score(y_test, y_pred2, pos_label='Yes')
recall3 = recall_score(y_test, reg_y_pred1, pos_label='Yes')
recall4 = recall_score(y_test, reg_y_pred2, pos_label='Yes')
print(recall, recall2, recall3, recall4)

#roc_auc
y_pred1_prob = gnb.predict_proba(x_test1)[:, 1]
y_pred2_prob = gnb2.predict_proba(x_test2)[:, 1]
y_pred3_prob = reg.predict_proba(x_test1)[:, 1]
y_pred4_prob = reg2.predict_proba(x_test2)[:, 1]
ra = roc_auc_score(y_test, y_pred1_prob)
ra2 = roc_auc_score(y_test, y_pred2_prob)
ra3 = roc_auc_score(y_test, y_pred3_prob)
ra4 = roc_auc_score(y_test, y_pred4_prob)
print(ra, ra2, ra3, ra4)

