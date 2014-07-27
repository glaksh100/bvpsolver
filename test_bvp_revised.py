from __future__ import division
import numpy as np
from bvp import solve_bvp
import matplotlib.pyplot as pt
""" A test script to plot the number of iterations taken by GMRES to converge 
for the slow BVP solver"""
a=-4
b=5

def test_poisson(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x)*np.exp(x)

    def r(x):
        return 10*np.exp(x)*np.cos(5*x)-24*np.exp(x)*np.sin(5*x)

    def p(x):
        return 0

    def q(x):
        return 0

    mesh = np.linspace(a, b, n)
    u, nit = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b),iter_count=True)


    return nit


def test_with_q(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return 0

    def q(x):
        return 1/(x+7)

    def r(x):
        return np.sin(5*x**2)/(x+7)-100*x**2*np.sin(5*x**2)+10*np.cos(5*x**2)

    mesh = np.linspace(a, b, n)
    u,nit  = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b),iter_count=True)


    return nit


def test_with_p(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 0

    def r(x):
        return (
                - 100*x**2*np.sin(5*x**2)
                + 10*x*np.cos(x)*np.cos(5*x**2)+10*np.cos(5*x**2)
                )

    mesh = np.linspace(a, b, n)
    u,nit = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b),iter_count=True)

    

    return nit


def test_full_enchilada(n):
    a = -4
    b = 5

    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 1/(x+7)

    def r(x):
        return (np.sin(5*x**2)/(x+7)
                - 100*x**2*np.sin(5*x**2)
                + 10*x*np.cos(x)*np.cos(5*x**2)+10*np.cos(5*x**2))

    mesh = np.linspace(a, b, n)
    u, nit = solve_bvp(mesh, p, q, r, ua=u_true(a), ub=u_true(b),iter_count=True)


    return nit


if __name__ == "__main__":
    
    for test in [test_poisson, test_with_p, test_with_q, test_full_enchilada]:
                n=[]
                t=[]
                nit=[]
                for i in range(7,12):
                    n.append(2**i-1)
                    val_nit=test(2**i)
                    nit.append(val_nit)
                pt.figure(1)
                pt.plot(n,nit,label='%s' %(test.__name__))
                pt.xlabel("Number of subintervals")
                pt.ylabel("Number of iterations")
                pt.legend(loc=2,prop={'size':7})
                
    print "Done"
    pt.show()
                