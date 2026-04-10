from gekko import GEKKO
import numpy as np

m = GEKKO()

m.time = [0,1,2,3,5,8,12,17,23,30,38,48,60]
p_meas = [0,0,0,100,100,100,100,\
          100,100,100,100,100,100]
v_meas = [0,0,0,0,18.13,39.35,59.34,75.34,\
          86.47,93.28,96.98,98.89,99.67]

mass = 500 # kg

b = m.FV(20,lb=1e-5,ub=100)  # resistive coefficient (N-s/m)
K = m.FV(0.8) # gain (m/s-%pedal)
b.STATUS=1; K.STATUS=1 # adjustable by optimizer

p = m.Param(p_meas,lb=0,ub=100)
v = m.CV(v_meas); v.FSTATUS = 1
tau = m.Intermediate(mass/b)

m.Equation(tau*v.dt()==-v + K*p)

m.options.IMODE = 5
m.options.NODES = 3
m.options.SOLVER= 1

m.solve()

print('')
print('Solution: ')
print('K: ' + str(K.value[0]))
print('b: ' + str(b.value[0]))