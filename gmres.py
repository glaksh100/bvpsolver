from __future__ import division

import numpy as np
import numpy.linalg as la

def back_substitute(U, bb):
    """Solve Ux = bb where U is an upper triangular matrix obtained by Givens Factorization. 
    Used in my_gmres_e to find the value of yk
    """
    n = U.shape[1]
    x = np.zeros(n)
    for j in range(n - 1, -1, -1): # loop backwards over columns
        if U[j, j] == 0:
            raise RuntimeError("singular matrix")
        x[j] = bb[j] / U[j, j]
        for i in range(0, j):
            bb[i] -= U[i, j] * x[j]
    return x

def my_gmres_d(A_func, b, tol=1e-10):
    """Solve Ax = b to an absolute residual norm of at most tol.
    Returns a tuple (x, num_iterations).
    Solution to Part 1(d) of the project assignment.
    """
    
    n=len(b)
    x0=b
    Q=np.zeros((n,n))
    H=np.zeros((n,n))
    k=0
    Q[:,k]=x0/la.norm(x0)
    for k in range(n):
        u=A_func(Q[:,k])
        for j in range(k+1):
            qj=Q[:,j]
            H[j][k]=np.dot(qj,u) 
            u=u-H[j][k]*Q[:,j] # Orthogonalization
        if(k+1 < n):
            H[k+1, k] = la.norm(u)
            Q[:, k+1] = u/la.norm(u)
            Qk=Q[:,:k+1]
            Qkp1=Q[:,0:k+2]
            Hk=H[0:k+2,0:k+1]
            e=np.zeros(k+2) # Building the e vector
            e[0]=1
            vect=la.norm(b)*e
            yk=la.lstsq(Hk,vect) # Finding the minimal residual of Hkyk=||b||e
            rk=np.dot(Hk,yk[0])-vect 
            rk_norm=la.norm(rk)
            if(rk_norm<tol): # Terminate when tolerance is reached 
                xk=np.dot(Qk,yk[0])
                return xk,k+1


    
def my_gmres_e(A_func, b, tol=1e-10):
    """Solve Ax = b to an absolute residual norm of at most tol.
    Returns a tuple (x, num_iterations).
    Solution to Part 1(e) of the project assignment.
    """
    
    n = len(b)
    q = np.zeros((n, n))
    h = h=np.zeros((n,n))
    
    x0 = b
    xk = x0
    
    k=0
    q[:,k] = x0/la.norm(x0)
    
    for k in range (0, n):
        uk = A_func(q[:, k].copy())
        
        for j in range (0, k + 1): 
            h[j, k] = np.dot(q[:, j], uk)
            uk = uk - h[j, k]*q[:, j]
        h[k + 1, k] =  la.norm(uk)

        if (h[k + 1, k] == 0):
            break
        
        q[:,k+1] = uk/h[k+1, k]
        
        
        h_copy = h.copy()
        e= la.norm(b)*np.reshape(np.eye(k + 2)[0, :], (k + 2, 1)) #building the e vector
        
        for j in range(0, k + 1):
            x = h_copy[j, j]
            y = h_copy[j + 1, j]        
            c = x/np.sqrt(x**2 + y**2)
            s = y/np.sqrt(x**2 + y**2) 
            G = np.array([[c, s], [-s, c]])  # 2x2 Given's matrix

            e[j: j + 2, 0] = np.dot(G, e[j: j + 2, 0])

            for i in range(j, k + 1):
                h_copy[j: j+2, i] = np.dot(G, h_copy[j:j+2, i]) # Multiplying by Given's matrix
                
        yk = back_substitute(h_copy[0: k + 1, 0: k + 1], e[0: k + 1, 0]) #Using back_subsitute to solve for yk
        xk = np.dot(q[:, 0: k + 1], yk)
        rk = A_func(np.reshape(xk, n).copy()) - x0        

        if (la.norm(rk) < tol): 
            break

    return xk, k+1 


def test_gmres(gmres_func):
    n = 100
    eigvals = 1 + 10**np.linspace(1, -20, n)
    eigvecs = np.random.randn(n, n)
    A = np.dot(
            la.solve(eigvecs, np.diag(eigvals)),
            eigvecs)

    def A_func(x):
        return np.dot(A, x)

    x_true = np.random.randn(n)
    b = np.dot(A, x_true)
    x, num_it = gmres_func(A_func, b)

    print "converged after %d iterations" % num_it
    print "residual: %g" % (la.norm(np.dot(A, x) - b)/la.norm(b))
    print "error: %g" % (la.norm(x-x_true)/la.norm(x_true))


if __name__ == "__main__":
    print "----------------------------------------"
    print "part(d)"
    print "----------------------------------------"
    test_gmres(my_gmres_d)
    print "----------------------------------------"
    print "part(e)"
    print "----------------------------------------"
    test_gmres(my_gmres_e)
