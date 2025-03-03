import numpy as np
from collections import Counter
import logging
    
def fast_prox(y, lamb):

    n     = len(y)
    idx_i = np.zeros(n, dtype = np.int64)
    idx_j = np.zeros(n, dtype = np.int64)
    s     = np.zeros(n)
    w     = np.zeros(n)
    x     = np.zeros(n)

    k = 0
    for i in range(n):
        idx_i[k] = i
        idx_j[k] = i
        s[k]     = y[i] - lamb[i]
        w[k]     = s[k]
        while k > 0 and w[k-1] <= w[k]:
            k        = k-1
            idx_j[k] = i
            s[k]     = s[k] + s[k+1]
            w[k]     = s[k] / (i - idx_i[k] + 1)
        k = k + 1
    
    for j in range(k):
        ind    = list(range(idx_i[j],idx_j[j]+1))
        x[ind] = max(w[j], 0)

    return x
            
def solve_prox(y,lamb):
    # sort y and index
    n        = len(y)
    lamb     = np.squeeze(lamb)
    y        = np.squeeze(y)
    sgn      = np.sign(y)
    sorted_y = np.array(sorted(enumerate(np.abs(y)),key=lambda x:x[1],reverse=True))
    index    = sorted_y[:,0]
    index    = index.astype(int)

    value_y  = sorted_y[:,1]
    t        = value_y - lamb 
    x        = np.zeros(n)
    
    # Simplify the problem  
    s        = np.where(t>0)[0]
    last     = s[-1]

    # Compute solution and re-normalize
    if len(s) != 0:
        v1                = value_y[:last+1]
        v2                = lamb[:last+1]
        v                 = fast_prox(v1,v2)
        x[index[:last+1]] = v

    # Restore signs
    x = sgn * x
    return x

def lasso_prox(y,lamb):
    n = len(y)
    lamb     = np.squeeze(lamb)
    y        = np.squeeze(y)
    sgn = np.sign(y)
    x = np.zeros(n)
    t = abs(y) - lamb
    for i in range(n):
        x[i] = sgn[i] * max(t[i],0)
    
    return x

def mainly(A,b,lamb):
    # the row vector: b; the column vector: lamb
    # -------------------------
    # Parse parameters
    # -------------------------
    
    iterations = 10000   # Maximum number of iterations
    gradIter   = 20      # Complete gradient
    optimIter  = 1       # Optimal condition
    tolRelGap  = 1e-6    # Stop criteria
    tolInfeas  = 1e-6   # Maximum value of dual infeasible domain
    
    # Get initial lower bound on the Lipschitz constant 
    n = len(b)
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    x = np.dot(A.T,np.dot(A,x))
    L = np.linalg.norm(x)  

    # Constants for exit status
    STATUS_RUNNING    = 0 
    STATUS_OPTIMAL    = 1 
    STATUS_ITERATIONS = 2 
    STATUS_MSG        = ['Optimal','Iteration limit reached'] 

    # Initialize parameters and iterates
    t       = 1
    eta     = 2
    x       = np.zeros((n,1)) 
    y       = x 
    b       = np.expand_dims(b,axis =1) 
    Ax      = np.dot(A,x)
    status  = STATUS_RUNNING
    Aprods  = 2 # Number of products with A
    ATprods = 1 # Number of products with A.T
    iter    = 0 

    logging.info("%s %10s %10s %10s %10s"%('Iter','||r||_2','Gap','Infeas','Rel.gap'))

    #----------------------------
    # Main loop
    #----------------------------
    while True:
        if np.mod(iter,gradIter) == 0:
            r = np.dot(A,y)-b  
            g = np.dot(A.T,r) 
            f = np.dot(r.T,r)/2 
            
        else:
            r = (Ax + ((tPrev - 1) / t) * (Ax - AxPrev)) - b 
            g = np.dot(A.T,r) 
            f = np.dot(r.T,r)/2  
        
        # Increment iteration count
        iter = iter + 1

        # Check optimality conditions
        if ((np.mod(iter,optimIter) == 0)):
            g_sort = sorted(abs(g),reverse = True) # n*1
            y_sort = sorted(abs(y),reverse = True) # n*1
            infeas = max(max(np.cumsum(g_sort-lamb)),0)

            # Compute primal and dual objective
            objPrimal =  f + np.dot(lamb.T,y_sort)
            objDual   = -f - np.dot(r.T,b)
            gap       =  objPrimal - objDual

        str = "%13s %10f %10s"%(gap,infeas/lamb[1],abs(gap)/max(1,objPrimal))

        # Check primal-dual gap
        if abs(gap)/max(1,objPrimal) < tolRelGap and infeas < tolInfeas * lamb[1]:
            status = STATUS_OPTIMAL
        
        logging.info("%d %10s %10s"%(iter, f, str))
        
        if iter > iterations and status==0 :
            status = STATUS_ITERATIONS
        
        if status:
            logging.info("Existing with status {} -- {}\n".format(status, STATUS_MSG[status-1]))
            break
    
        # Keep copies of previous values 
        AxPrev = Ax 
        xPrev  = x
        fPrev  = f
        tPrev  = t

        num = Counter(lamb.flatten())

        if num[float(lamb[0])] != n:
            # Lipschitz search
            while True:
                # Compute prox mapping
                x = solve_prox(y - 1/L * g,lamb/L)
                x = x[:,np.newaxis] 
                d = x - y 
                Ax = np.dot(A,x) 
                r  = Ax-b 
                f  = np.dot(r.T,r)/2 
                q  = fPrev + np.dot(d.T,g) + L/2 * np.dot(d.T,d) 
                if q >= f * (1-1e-12):
                    break
                else:
                    L = L * eta
        else:
            # Lipschitz search
            while True:
                # Compute prox mapping
                x = lasso_prox(y - 1/L * g,lamb/L)
                x = x[:,np.newaxis] 
                d = x - y 
                Ax = np.dot(A,x)
                r  = Ax-b 
                f  = np.dot(r.T,r)/2 
                q  = fPrev + np.dot(d.T,g) + L/2 * np.dot(d.T,d) 
                if q >= f * (1-1e-12):
                    break
                else:
                    L = L * eta

        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((tPrev - 1) / t) * (x - xPrev)

    x = y
    return x