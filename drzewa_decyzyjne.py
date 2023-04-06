import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')
print(df['class'].value_counts())
#klasy zbalansowane

#zmienna wynikowa (class) powinna mieć charakter numeryczny
species = {
    'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2
}
df['class_value'] = df['class'].map(species)
print(df['class_value'].value_counts())

X = df[ ["petallength","petalwidth"] ]
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X.values, y.values, model)
plt.show()
