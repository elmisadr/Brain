from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
import pandas as pd
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_roc_curve
data = pd.read_csv('brain_stroke.csv', encoding = 'utf-8')
data['ever_married'] = [ 0 if i !='Yes' else 1 for i in data['ever_married'] ]
data['gender'] = [0 if i != 'Female' else 1 for i in data['gender']]
data = pd.get_dummies(data, columns = ['work_type', 'Residence_type','smoking_status'])
DATA = pd.read_csv('brain_stroke.csv', encoding = 'utf-8')

X_cat = DATA[['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'smoking_status']]
X_num = DATA.drop(['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'smoking_status', 'stroke'], axis=1)
X_cat = pd.get_dummies(X_cat)
scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)
X1 = pd.concat([X_scaled, X_cat], axis=1)
y1=DATA['stroke']
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.3, random_state = 0)
X = data.drop(['stroke'], axis = 1)
y = data['stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
len(X_train), len(y_train), len(X_test), len(y_test)
#   SVM 
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X1_train, y1_train)
# Predicting the Test set results
y_pred = classifier.predict(X1_test)
cm_test = confusion_matrix(y1_test, y_pred )
y_pred_train = classifier.predict(X1_train)
cm_train = confusion_matrix(y_pred_train, y1_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y1_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y1_test)))
print(cm_train)
print(cm_test)

import warnings
warnings.filterwarnings('ignore')

print(classification_report(y_pred, y1_test))

plt.show()
plot_confusion_matrix(classifier, X1_test, y1_test)
plot_roc_curve(classifier, X1_test, y1_test)
plt.plot(y_proba)

