import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('/home/janek/Documents/trees/data/trees.csv')
print(data.shape)
print(data.columns)
labels, levels = pd.factorize(data['gatunek'])
data['labels'] = labels
data.fillna(0)
label_counts = pd.Series(labels).value_counts()
filt_data = data[data['labels'].isin(label_counts[label_counts > 30].index)]

filt_data['obwod'].hist(by=filt_data['labels'])
plt.scatter(data['obwod'], data['O3'], c=labels)
plt.show()



#
# plt.scatter(data['srednica_kor'], data['O3'])
# plt.show()

y = np.array(data['O3'].fillna(0))
x = np.array(data[['obwod', 'srednica_kor']])
np.any(np.isnan(x))
lr = LinearRegression()
lr.fit(x, y)
print('R^2 wynosi {}'.format(lr.score(x, y)))
a = lr.predict(x)
plt.figure()
plt.boxplot(data['srednica_kor'])
