import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import joblib
data = pd.read_csv('Tsk01Data.csv')
X = np.array(data[['math']])
Y = np.array(data[['cs']])
model=LinearRegression()
reg = model.fit(X, Y)
# print(reg.coef_)
# print(reg.intercept_)
# save the model to disk
file1 = 'finalized_pickle_model.sav'
pickle.dump(model, open(file1, 'wb'))
file2 = 'finalized_joblib_model.sav'
joblib.dump(model, file2)

# load the model from disk
loaded_model = pickle.load(open(file1, 'rb'))
resultpickle = loaded_model.score(X, Y)
print(resultpickle)

# load the model from disk
loaded_model = joblib.load(file2)
resultjoblib = loaded_model.score(X, Y)
print(resultjoblib)
