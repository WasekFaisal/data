

#importing basic packages
import pandas as pd

import matplotlib.pyplot as plt
#load the dataset
df = pd.read_csv('‪https://raw.githubusercontent.com/WasekFaisal/data/dc4d70e300861f7d971ea2ca8b45538462aede8e/dataset.csv')

df.head()

df.Domain.nunique()

"""### **2. Familiarizing with Data**

"""

#Checking the shape of the dataset
df.shape

#Listing the features of the dataset
df.columns

#Information about the dataset
df.info()

"""### **3. Visualizing the data**

"""

#Plotting the data distribution
df.hist(bins = 50,figsize = (15,15))
plt.show()

"""### **4. Dataset Description**

"""

df.describe()

#Dropping the Domain column
data = df.drop(['Domain'], axis = 1).copy()

#checking the data for null or missing values
data.isnull().sum()

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)
data.head()

"""### **5. Splitting the Data**"""

# Sepratating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)
X.shape, y.shape





# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

"""### **6.0. Machine Learning Models & Training**
* KNN
* Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest
* Multilayer Perceptrons
* XGBoost
* Support Vector Machines
* ANN
Naive Bayes
"""

#importing packages
from sklearn.metrics import accuracy_score

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

#function to call for storing the results
def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))

"""### **6.1 KNN**
 
"""

from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

y_test_knn = knn.predict(X_test)
y_train_knn = knn.predict(X_train)

#computing the accuracy of the model performance
acc_train_knn = accuracy_score(y_train,y_train_knn)
acc_test_knn = accuracy_score(y_test,y_test_knn)

print("KNN: Accuracy on training Data: {:.2f}%".format(acc_train_knn*100))
print("KNN : Accuracy on test Data: {:.2f}%".format(acc_test_knn*100))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('KNN', acc_train_knn, acc_test_knn)

"""### **6.2 Naive Bayes**"""

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,y_train)

y_test_NB = NB.predict(X_test)
y_train_NB = NB.predict(X_train)

#computing the accuracy of the model performance
acc_train_NB = accuracy_score(y_train,y_train_NB)
acc_test_NB = accuracy_score(y_test,y_test_NB)

print("Naive Bayes: Accuracy on training Data: {:.2f}%".format(acc_train_NB*100))
print("Naive Bayes : Accuracy on test Data: {:.2f}%".format(acc_test_NB*100))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Naive Bayes', acc_train_NB, acc_test_NB)

"""### **6.3 Logistic Regression**

"""

#Support vector machine model
from sklearn.linear_model import LogisticRegression

# instantiate the model
LR = LogisticRegression(random_state=10)
#fit the model
LR.fit(X_train, y_train)

y_test_LR = LR.predict(X_test)
y_train_LR = LR.predict(X_train)

#computing the accuracy of the model performance
acc_train_LR = accuracy_score(y_train,y_train_LR)
acc_test_LR = accuracy_score(y_test,y_test_LR)

print("Logistic Regression: Accuracy on training Data: {:.2f}%".format(acc_train_LR*100))
print("Logistic Regression : Accuracy on test Data: {:.2f}%".format(acc_test_LR*100))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Logistic Regression', acc_train_LR, acc_test_LR)

"""### **6.4. Decision Tree Classifier**

"""

# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier

# instantiate the model 
tree = DecisionTreeClassifier(max_depth = 10)
# fit the model 
tree.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.2f}%".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.2f}%".format(acc_test_tree))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Decision Tree', acc_train_tree, acc_test_tree)

"""### **6.5. Random Forest Classifier**

"""

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
forest = RandomForestClassifier(n_estimators=1000,max_depth=10)

# fit the model 
forest.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training Data: {:.2f}%".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.2f}%".format(acc_test_forest))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Random Forest', acc_train_forest, acc_test_forest)

"""### **6.6. Multilayer Perceptrons (MLPs): Deep Learning**

"""

# Multilayer Perceptrons model
from sklearn.neural_network import MLPClassifier

# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))

# fit the model 
mlp.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)

#computing the accuracy of the model performance
acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training Data: {:.2f}%".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.2f}%".format(acc_test_mlp))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

"""### **6.7. XGBoost Classifier**

"""

#XGBoost Classification model
from xgboost import XGBClassifier

# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

#computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)

print("XGBoost: Accuracy on training Data: {:.2f}%".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.2f}%".format(acc_test_xgb))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('XGBoost', acc_train_xgb, acc_test_xgb)

"""### **6.8. Support Vector Machines**


"""

#Support vector machine model
from sklearn.svm import SVC

# instantiate the model
svm = SVC(random_state=5)
#fit the model
svm.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)

#computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train,y_train_svm)
acc_test_svm = accuracy_score(y_test,y_test_svm)

print("SVM: Accuracy on training Data: {:.2f}%".format(acc_train_svm*100))
print("SVM : Accuracy on test Data: {:.2f}%".format(acc_test_svm*100))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('SVM', acc_train_svm, acc_test_svm)

"""### **6.9. Autoencoder Neural Network**"""

#importing required packages
#import keras
from keras.layers import Input, Dense
from keras import regularizers

from keras.models import Model
from sklearn import metrics

#building autoencoder model

input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)

encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim-2), activation='relu')(code)

decoder = Dense(int(encoding_dim), activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

#compiling the model
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#Training the model
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2)

acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

print('\nAutoencoder: Accuracy on training Data: {:.3f}' .format(acc_train_auto))
print('Autoencoder: Accuracy on test Data: {:.3f}' .format(acc_test_auto))

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('AutoEncoder', acc_train_auto, acc_test_auto)

"""## **7. Comparision of Models**

"""

#creating dataframe
results = pd.DataFrame({ 'ML Model': ML_Model,    
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results

#Sorting the datafram on accuracy
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
