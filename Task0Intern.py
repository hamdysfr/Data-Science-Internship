import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from word2number import w2n

df = pd.read_csv('Task0.csv')
#replce numeric year of experince in dataframe with number
df['experience']=df['experience'].replace(np.nan, 'zero')
df['experience']=df.experience.apply(w2n.word_to_num)
Mean1=df['experience'].mean()
df['experience']=df['experience'].replace(0,Mean1)
Mean2=df['test_score(out of 10)'].mean()
df['test_score(out of 10)']=df['test_score(out of 10)'].replace(np.nan,Mean2)
#print(df)
X=  np.asanyarray(df[['experience','test_score(out of 10)','interview_score(out of 10)']])
Y= np.asanyarray(df[['salary($)']])
reg = LinearRegression().fit(X, Y)
# print(reg.score(X, Y))
print(reg.coef_)
print(reg.intercept_)
f = open("Task0.txt", "x")
f.write("Prediction salary for 2 yr experience, 9 test score, 6 interview score is: %d\r" %reg.predict(np.array([[2,9,6]])))
f.write("And prediction salary for  12 yr experience, 10 test score, 10 interview score is: %d\r" %reg.predict(np.array([[12,10,10]])))
f.close()