import numpy as np
from gekko import GEKKO
# editing
# load data
xm = np.array([18.3447,79.86538,85.09788,10.5211,44.4556, \
               69.567,8.960,86.197,66.857,16.875, \
               52.2697,93.917,24.35,5.118,25.126, \
               34.037,61.4445,42.704,39.531,29.988])

ym = np.array([5.072,7.1588,7.263,4.255,6.282, \
               6.9118,4.044,7.2595,6.898,4.8744, \
               6.5179,7.3434,5.4316,3.38,5.464, \
               5.90,6.80,6.193,6.070,5.737])

# define GEKKO model
m = GEKKO()
# parameters and variables
a = m.FV(value=0)
b = m.FV(value=0)
c = m.FV(value=0,lb=-100,ub=100)
x = m.Param(value=xm)
ymeas = m.Param(value=ym)
ypred = m.Var()
# parameter and variable options
a.STATUS = 1 # available to optimizer
b.STATUS = 1 #  to minimize objective
c.STATUS = 1
# equation
m.Equation(ypred == a + b/x + c*m.log(x))
# objective
m.Minimize(((ypred-ymeas)/ymeas)**2)
# application options
m.options.IMODE = 2   # regression mode
# solve
m.solve() # remote=False for local solve

# show final objective
print('Final SSE Objective: ' + str(m.options.objfcnval))

# print solution
print('Solution')
print('a = ' + str(a.value[0]))
print('b = ' + str(b.value[0]))
print('c = ' + str(c.value[0]))

# plot solution
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(x,ymeas,'ro')
plt.plot(x,ypred,'bx');
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Measured','Predicted'],loc='best')
plt.savefig('results.png')
plt.show()