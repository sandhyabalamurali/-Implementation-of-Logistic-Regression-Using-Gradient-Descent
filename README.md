

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
# Developed by: SANDHYA B N
# RegisterNumber:  212222040144
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
print("Array of X") 
X[:5]
print("Array of y") 
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
print("Prediction value of mean")
np.mean(predict(res.x,X)==y)

*/
```


## Output:
![logistic regression using gradient descent](sam.png)



![5 1](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/2e661a4f-8be1-4fdb-b001-486d96801bad)



![5 2](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/5ab96f20-b88a-412d-8630-28929d75881c)



![5 3](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/a65074aa-1fad-4a3f-8d8c-f360a856fdf7)



![5 4](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/37a46cb7-588e-4872-ad8c-283e6284e692)



![5 5](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/a0935dab-c6f5-4dba-b5f4-e3bbb98a6fda)



![5 6](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/a28e4e3d-1223-4e09-8553-5a3e32c1355b)



![5 7](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/d3be62f7-69c3-4118-9051-ceef0e20e2ce)



![5 8](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/0ca366fb-56ca-4fe1-bfb4-31debb3a4e08)



![5 9](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/c8fa314a-8d98-4481-ac98-a6b33246a911)



![5 10](https://github.com/sandhyabalamurali/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/115525118/eed73175-756e-4f22-ab27-dffbaab0b19f)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

