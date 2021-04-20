
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\AI\bridge.txt", delimiter=",", names=['A','B','C','D','E','F','G','H','I','J','K','L','M'], na_values="?")
output=pd.read_csv(r"C:\AI\out.txt", names=['outputs'])
datain=pd.DataFrame(data)
cols=[1,2,3,4,5,6,7,8,9,10,11,12]
datain=datain[datain.columns[cols]]


from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()

datain=datain.astype(str).apply(number.fit_transform)
#datain.replace(3,99)
#datain[0:6]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(np.nan_to_num(datain), output, test_size=0.2, random_state=1)
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
predictions = dtree.predict(x_test)
print('Accuracy:',accuracy_score(y_test, predictions))


datain.to_csv(r"C:\Users\VaniSuresh\Desktop\convertedbridges.txt", index = None, header=['A','B','C','D','E','F','G','H','I','J','K','L'])
dataout=pd.DataFrame(output)
dataout['outputs']=number.fit_transform(dataout['outputs'].astype('str'))
dataout.to_csv(r"C:\Users\VaniSuresh\Desktop\outputbridges.txt", index = None, header=['outputs'])


