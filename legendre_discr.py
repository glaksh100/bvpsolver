from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.special as sp


class CompositeLegendreDiscretization:
    """A discrete function space on a 1D domain consisting of multiple
    subintervals, each of which is discretized as a polynomial space of
    maximum degree *order*. (see constructor arguments)

    There are :attr:`nintervals` * :attr:`npoints` degrees of freedom
    representing a function on the domain. On each subinterval, the
    function is represented by its function values at Gauss-Legendre
    nodes (see :func:`scipy.speical.legendre`) mapped into that
    subinterval.

    .. note::

        While the external interface of this discretization
        is exclusively in terms of vectors, it may be practical
        to internally reshape (see :func:`numpy.reshape`) these
        vectors into 2D arrays of shape *(nintervals, npoints)*.

    The object has the following attributes:

    .. attribute:: intervals

        The *intervals* constructor argument.

    .. attribute:: nintervals

        Number of subintervals in the discretization.

    .. attribute:: npoints

        Number of points on each interval. Equals *order+1*.

    .. attributes:: nodes

        A vector of ``(nintervals*npoints)`` node locations, consisting of
        Gauss-Legendre nodes that are linearly (or, to be technically correct,
        affinely) mapped into each subinterval.

    """


    def __init__(self, intervals, order):
        """
        :arg intervals: determines the boundaries of the subintervals to
            be used. If this is ``[a,b,c]``, then there are two subintervals
            :math:`(a,b)` and :math:`(b,c)`.
            (and the overall domain is :math:`(a,c)`)

        :arg order: highest polynomial degree being used
        """
        self.intervals=intervals

        self.nintervals=len(intervals)-1

        self.npoints=order + 1

        #Initializing shifted_nodes
        shifted_nodes=np.zeros(self.npoints*self.nintervals) 

        #Calling the scipy function to obtain the unshifted nodes
        unshifted_nodes=sp.legendre(self.npoints).weights[:,0]

        #Initializing shifted weights
        shifted_weights=np.zeros(self.nintervals*self.npoints)

        #Calling the scipy function to obtain the unshifted weights
        unshifted_weights=sp.legendre(self.npoints).weights[:,1]

        #Linearly mapping the unshifted nodes and weights to get the shifted
        #nodes and weights
        for i in range(self.nintervals):
                shifted_nodes[i*self.npoints:(i+1)*self.npoints]=(self.intervals[i]+ self.intervals[i+1])/2 + (self.intervals[i+1]-self.intervals[i])*(unshifted_nodes[0:self.npoints])/2
                shifted_weights[i*self.npoints:(i+1)*self.npoints]=(self.intervals[i+1]-self.intervals[i])*(unshifted_weights[0:self.npoints])/2
       
        #Setting nodes and weights attributes
        self.nodes=np.reshape(shifted_nodes,(self.nintervals,self.npoints))
        self.weights=np.reshape(shifted_weights,(self.nintervals,self.npoints))
    
        #Obtaining Vandermonde and RHS matrices to get A
        def vandermonde_rhs(m,arr):
            X=np.zeros((m,m))
            RHS=np.zeros((m,m))
            for i in range(m):
                for j in range(m):
                    X[i][j]=arr[i]**j
                    RHS[i][j]=((arr[i]**(j+1))-((-1)**(j+1)))/(j+1)
            return X,RHS
        
        A=np.zeros((self.npoints,self.npoints))
        X,RHS=vandermonde_rhs(self.npoints,unshifted_nodes)

        #Solving for spectral integration matrix
        A=np.dot(RHS,la.inv(X))
        
        self.A=A

        

    def integral(self, f):
        r"""Use Gauss-Legendre quadrature on each subinterval to approximate
        and return the value of

        .. math::

            \int_a^b f(x) dx

        where :math:`a` and :math:`b` are the left-hand and right-hand edges of
        the computational domain.

        :arg f: the function to be integrated, given as function values
            at :attr:`nodes`
        """
        int_val=0
        #Using Composite Gauss Quadrature to evaluate the whole integral

        for i in range(self.nintervals):
            f_val_arr=f[i,:]
            dot_val=np.dot(self.weights[i,:],f_val_arr.T)
            int_val+=dot_val

        return int_val




    def left_indefinite_integral(self, f):
        r"""Use a spectral integration matrix on each subinterval to
        approximate and return the value of

        .. math::

            g(x) = \int_a^x f(x) dx

        at :attr:`nodes`, where :math:`a` is the left-hand edge of the
        computational domain.

        The computational cost of this routine is linear in
        the number of degrees of freedom.

        :arg f: the function to be integrated, given as function values
            at :attr:`nodes`
        """
        #Initializing the left_indefinite_integral
        I=np.zeros((self.nintervals,self.npoints))
        current_quad_val=0

        for i in range(self.nintervals):
            
            if i>0:
                    f_val_prev=f[i-1:i,:]
                    weights_prev=self.weights[i-1:i,:]
                    current_quad_val+=np.dot(weights_prev,f_val_prev.T)
                
            current_f_val=f[i,:]

            #Scaling the Spectral integration matrix by the interval length
            A_scale=np.dot((self.intervals[i+1]-self.intervals[i])/2,self.A)

            spectral_factor=np.dot(A_scale,current_f_val.T)

            #Adding the  spectral factor to the evaluated definite integral
            indefinite_int=current_quad_val + spectral_factor
            I[i,:]=indefinite_int
            J=np.reshape(I,(self.nintervals*self.npoints,1))
        

        return I

    def right_indefinite_integral(self, f):
        r"""Use a spectral integration matrix on each subinterval to
        approximate and return the value of

        .. math::

            g(x) = \int_x^b f(x) dx

        at :attr:`nodes`, where :math:`b` is the left-hand edge of the
        computational domain.

        The computational cost of this routine is linear in
        the number of degrees of freedom.

        :arg f: the function to be integrated, given as function values
            at :attr:`nodes`
        """
        #Initializing the right integral
        I1=np.zeros(self.nintervals*self.npoints)

        #Unravelling the weights and function values at nodes
        f_right=f.ravel()    
        weights_right=self.weights.ravel()  
        

        for i in range(self.nintervals):
            
            #Evaluating Gauss Quadrature for all intervals from current to end
            f_next=f_right[i*self.npoints:self.npoints*self.nintervals]
            weights_next=weights_right[i*self.npoints:self.npoints*self.nintervals]
            current_quad_val=np.dot(weights_next,f_next.T)
            current_f_val=f_right[i*self.npoints:(i+1)*self.npoints]
            #Scaling the spectral integration matrix
            A_scale=np.dot((self.intervals[i+1]-self.intervals[i])/2,self.A)
            spectral_factor=np.dot(A_scale,current_f_val.T)
            
            #Subtracting spectral factor from Gauss Quadrature value
            I1[(i)*self.npoints:(i+1)*self.npoints]=np.dot(weights_next,f_next.T) - spectral_factor
            
            J=np.reshape(I1,(self.nintervals,self.npoints))

    
        return J
        
