from asyncio.windows_events import NULL
import math
from turtle import shape
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
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

def generateXvector(X):
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def Multivariable_Linear_Regression(X,y,learningrate, iterations):
    y_new=np.reshape(y, (len(y), 1))
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        cost_value = 1/(2*len(y))*((y_pred - y)**2) 
        #Calculate the loss for each training instance
        total = 0
        for j in range(len(y)):
            total += cost_value[j][0]
            #Calculate the cost function for each iteration
        cost_lst.append(total)
        iterr=i
        if iterr>2:
          if math.isclose(cost_lst[i],cost_lst[i-1],rel_tol=1e-20):
            break
    plt.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    plt.title('Cost function Graph')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    return theta,iterr


dataset = pd.read_excel('Salaryy.xlsx')
dataset = pd.DataFrame(dataset)
for i in range(2,len(dataset)):
  if pd.isnull(dataset.loc[i,"experience"]) is False:
    dataset.loc[i,"experience"]=text2int(str(dataset.loc[i,"experience"]))


dataset = dataset.dropna()

from sklearn.model_selection import train_test_split
x = dataset.drop('salary($)', axis=1)
y = dataset['salary($)'].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_transform=sc.fit_transform(x)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_transform, y)


print(Multivariable_Linear_Regression(X_transform,y, 0.001 , 30000))

plt.show()



