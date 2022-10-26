import math
import pandas as pd
import numpy as np

def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, words in enumerate(units):    numwords[words] = (1, idx)
      for idx, words in enumerate(tens):     numwords[words] = (1, idx * 10)
      for idx, words in enumerate(scales):   numwords[words] = (10 ** (idx * 3 or 2), 0)

    current = result = 0

    scale, increment = numwords[textnum]
    current = current * scale + increment
    if scale > 100:
      result += current
      current = 0

    return result + current
 
  
dataset = pd.read_excel('D:\Python\Python-Course\Salary Predection\Salaryy.xlsx')
dataset = pd.DataFrame(dataset)
for i in range(2,len(dataset)):
  if pd.isnull(dataset.loc[i,"experience"]) is False:
    dataset.loc[i,"experience"]=text2int(str(dataset.loc[i,"experience"]))


dataset = dataset.dropna()
from sklearn.model_selection import train_test_split
x = dataset.drop('salary($)', axis=1)
y = dataset['salary($)'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train, y_train)
y_pred = linear.predict(x_test)

data_dict = {
    'experience': 2,
    'test_score(out of 10)': 9,
    'interview_score(out of 10)': 6
}
data_dict2 = {
    'experience': 12,
    'test_score(out of 10)': 10,
    'interview_score(out of 10)': 10
}
input_data , input_data2= pd.DataFrame([data_dict]),pd.DataFrame([data_dict2])
predicted_salary, predicted_salary2 = linear.predict(input_data),linear.predict(input_data2)
with open('D:\Python\Python-Course\Salary Predection\predicted_salary.txt','w') as f:
  f.write('experience: 2, test_score(out of 10): 9, interview_score(out of 10): 6, predicted salary is: '+str(predicted_salary))
  f.write("\n")
  f.write('experience: 12, test_score(out of 10): 10, interview_score(out of 10): 10, predicted salary is: '+str(predicted_salary2))
