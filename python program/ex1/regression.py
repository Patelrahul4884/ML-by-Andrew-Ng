#!/usr/bin/env python
# coding: utf-8

# # Linear regression with one variable

# In[38]:


#all import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


# ## preprocessing input data

# In[61]:


data=pd.read_csv('ex1data1.txt',header=None)
data.head()


# In[62]:


X=data.iloc[:,0]
y=data.iloc[:,1]
m = len(y)
plt.scatter(X,y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# In[63]:


X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
num_iteration = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term


# ## cost function

# In[64]:


def computeCost(X,y,theta):
    predication=np.dot(X,theta)
    square_err=(predication-y)**2
    return 1/(2*m)*np.sum(square_err)
J=computeCost(X,y,theta)
print(J)


# ## gradient descent to optimize theta

# In[65]:


def gradientDescent(X,y,theta,alpha,num_iteration):
    for _ in range(num_iteration):
        temp=np.dot(X,theta)-y
        temp=np.dot(X.T,temp)
        theta=theta-(alpha/m)*temp
    return theta
theta = gradientDescent(X, y, theta, alpha, num_iteration)
print(theta)        


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

def animate():
    theta = np.zeros([2,1])
    for i in range(num_iteration):
        temp=np.dot(X,theta)-y
        temp=np.dot(X.T,temp)
        theta=theta-(alpha/m)*temp
    
        plt.plot(X[:,1], np.dot(X, theta))

ani = FuncAnimation(plt.gcf(), animate(), interval=1500)

plt.tight_layout()
plt.show()


# ### We now have the optimized value of theta . Use this value in the above cost function.

# In[17]:


J = computeCost(X, y, theta)
print(J)


# ## making predications

# In[18]:


plt.scatter(X[:,1],y,c='red',marker='x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()


# In[19]:


def predict(X,theta):
    predications=np.dot(X,theta)
    return predications[0]


# In[20]:


predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))


# In[21]:


predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))





