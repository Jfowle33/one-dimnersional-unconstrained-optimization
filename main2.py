# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:06:38 2020

@author: joefo
"""
import numpy as np
import cmath
from scipy import optimize




 
# a and b are the current bounds; the minimum is between them.
# c is the center pointer pushed slightly left towards a

def goldenSectionSearch(f, a, b, c, xtol, maxiter):
    maxiter = 200
    for k in range(maxiter):
        if b-a < c-b:
            print('We have convergence!, number of itertations:', k )
            return (b+c)/2
        else:
            return (a+b)/2
    # Create a new possible center, in the area between c and b, pushed against c
        #d =(b+c)/2
        #d=(a+b)/2
        #iteration
        if f(d) < f(c):
            
            return goldenSectionSearch(f, c, d, b, xtol, maxiter)
        else:
            return goldenSectionSearch(f, d, c, a, xtol, maxiter)
        #if c-a < xtol:
           # break
       # else:
           # print(c)
           # return c

f = lambda x: np.real(cmath.sqrt(x/2) + 2*cmath.sqrt(x-1/3))


abs(goldenSectionSearch(f, 0, (0.05), 1, 1e-6, maxiter=200)) < 1e-6



gold = goldenSectionSearch(f, 0, (0.05), 1, 1e-6, maxiter=200)
print(gold)
#Fmin bound to test the godlden section search ran above for accuracy
f = lambda x: np.real(-1*(1.1547*(1 - x)**0.5 + 0.707107*x**0.5))
rsi = optimize.fminbound(f, 0, 1, args=(), xtol=1e-6, maxfun=500, full_output=1, disp=3)
print(rsi)



def dx(df, x):
    return abs(0-df(x))
 
def newtons_method(df, df2, x0, e, maxiter):
    maxiter = 200
    for k in range(maxiter):
        delta = dx(df, x0)
        x0 = x0 - df(x0)/df2(x0)
        delta = dx(df, x0)
        if delta > e:
            print('converge', k)
            print ('Root is at: ', x0)
            print ('f(x) at root is: ', df(x0))

def df(x):
    return np.real(1/(2*cmath.sqrt(2)*cmath.sqrt(x)) - 1/cmath.sqrt(3 - 3*x))
def df2(x):
    return np.real(-3/(2*(3 - 3*x)**(3/2)) - 1/(4*cmath.sqrt(2)*x**(3/2)))


x0s = [0.5]
for x0 in x0s:
    newtons_method(df, df2, x0, 1e-6, maxiter=200)
  
    
    
    

    #given some function of u(x,y) = x^(1/2) + 2y^(1/2)
    #solving the one-dimnersional unconstrained optimization of the form
    #max x, such that (x/2)^(1/2) +2[(1-x)/3)]^1/2
    #where x is the decision variable
    #Here we will solve a nonlinear system of equations
    #The form is f(x,y,l):R^2->R^3 through a lagrange multiplier
    #FOC's: 2l = [(1/2)x]^(-1/2) 
    #     : 3l = y^(-1/2) 
    #     : 1 = 2x + 3y

    
#def equations(I):
   # x,y,l = I
    #f1 = [(1/2)*x]**(-1/2) -2*l
   # f2 = y**(-1/2) - 3*l
    #f3 = 2*x + 3*y - 1
   # return(f1,f2,f3)





def jacobian(f, x, dx=1e-6, method='central'):
    x = np.array(x)
    n = len(x)
    jac = np.zeros([n,n])
    e = np.eye(n) * dx
    if method=='forward':   
        for i in range(n):
            for j in range(n):
                jac[i][j] = (f(x+e[j])[i] - f(x)[i]) / dx
        return jac
    elif method=='backward': 
        for i in range(n):
            for j in range(n):
                jac[i][j] = (f(x)[i] - f(x-e[j])[i]) / dx
        return jac
    elif method=='central':           
        for i in range(n):
            for j in range(n):
                jac[i][j] = (f(x+e[j])[i] - f(x-e[j])[i]) / (2*dx)
        return jac
    else:
        print ("\nDo no know which method you choos!" )
if __name__=="__main__":
    def f(x):
        f = np.zeros(3)
        f[0] = 0.5*x[0]**-(1/2) - 2*x[2]
        f[1] = x[1]**-(1/2) - 3*x[2] 
        f[2] = 2*x[0] + 3*x[1] -1
        return f
    x0 = [0.5/2,0.5/3,1.0]     # x can be a list, a tuple, or an array
    jac = jacobian(f, x0, dx=1e-6, method='forward')
    print ("\n The Jacobian of f(x) using forward-difference approximation is: ")
    print (jac)
    jac = jacobian(f, x0, dx=1e-6, method='backward')
    print ("\n The Jacobian of f(x) using backward-difference approximation is: ")







    print (jac)
    jac = jacobian(f, x0, dx=1e-6, method='central')
    print ("\n The Jacobian of f(x) using central-difference approximation is: ")
    print (jac)
    

def func(X):
    x = X[0]
    y = X[1]
    L = X[2]
    return [0.5*x**-0.5 - 2*L, y**-0.5 - 3*L, 2*x + 3*y -1]

#def dfunc(X):
   # dLambda = np.zeros(len(X))
   # e = 1e-6
    #for i in range(len(X)):
      #  dX = np.zeros(len(X))
       # dX[i] = e
       # dLambda[i] = (func(X+dX)-func(X-dX))//(2*e);
        #if dLambda[i] > e:
          # print('We have convergence!, number of iter:', i ) 
   # return dLambda
def jacobian(xyL):
    x, y, L = xyL
    return  np.array([[ -2.,0.,-2.],
                     [ 0.,-7.3485023,-3.],
                     [ 2.,3.,0.]])
                     

def iterative_newton(func, x_init, jacobian):
    max_iter = 200
    epsilon = 1e-6

    x_last = x_init

    for k in range(max_iter):

        J = np.array(jacobian(x_last))
        #print('j, this iteration', J)
        F = np.array(func(x_last))
        #print('f, this iteration', F)

        diff = np.linalg.solve( J, -F )
        print('difference, this iteration', diff)
        x_last = x_last + diff

        # Stop condition:
        if np.linalg.norm(diff) < epsilon:
            print('We have convergence!, number of itertations:', k )
            
            break

    else: # only if the for loop ends 'naturally'
        print('not converged')
        
    return x_last
    
mewtwo = iterative_newton(func, [0.5/2,0.5/3,1.0], jacobian)
print('solution', mewtwo)
