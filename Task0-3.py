import pandas as pd

df = pd.read_csv('Task03Data.csv')
dummies=pd.get_dummies(df,columns=['Car Model'])
df = pd.concat([df.drop('Car Model', axis=1),dummies ], axis=1)
df.to_csv("dummy.csv")