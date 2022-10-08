import numpy as np
import scipy.linalg
from functools import reduce

def canInf(x):
  m, n = x.shape
  S1 = np.reshape(np.sum(x, axis=1), (m, 1))
  S2 = np.zeros((m, m))
  for i in range(n):
    a = x[:, i:i + 1]
    S2 += a @ np.transpose(a)
  return (S1, S2, n)

def reduce_(x, y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2])

def result(r):
    return (r[0] / r[2], (r[1] - (r[0] @ np.transpose(r[0])) / r[2]) / (r[2] - 1))

m = 3   
n = 20  

X0 = np.array([[1.],[2.],[3.]])
V0 = np.array([[1., .5, .5],[.5, 1., .5],[.5, .5, 1.]])

Vs = scipy.linalg.sqrtm(V0)

X = X0 @ np.ones((1,n)) + Vs @ np.random.normal(size=(m, n))

nodes = [X[:,:5], X[:,5:10], X[:,10:15], X[:,15:20]]

print(result(reduce(reduce_, map(canInf, nodes)))[0])
print(result(reduce(reduce_, map(canInf, nodes)))[1])

print(np.reshape(np.mean(X,axis=1), (m, 1))) 
print(np.cov(X))   

