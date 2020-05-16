#recommend for target user
def rec(pra,ite):  
    import pandas as pd
    import numpy as np
    from scipy import sparse
    from scipy.io import loadmat
    import math
    import csv
    #ciao
    matrate = loadmat(u'ciao/rating.mat')
    ratings = matrate['rating']
    ratings = np.delete(ratings,[2,4],1)

    matfri = loadmat(u'ciao/trustnetwork.mat')
    friends = matfri['trustnetwork']

    lu = 7375
    lm = 106797

    '''
    #epinions
    matrate = loadmat(u'epinions/rating_with_timestamp.mat')
    ratings = matrate['rating_with_timestamp']
    ratings = np.delete(ratings,[2,4,5],1)

    matfri = loadmat(u'epinions/trust.mat')
    friends = matfri['trust']

    lu = 22166
    lm = 296277
    '''

    lr = ratings.shape[0]
    lf = friends.shape[0]
    print(lr,lf)
    shape_kiyo = (lu,2)
    kiyo = np.zeros(shape_kiyo)

    with open('trust111_kiyo.csv','w') as f:
        writer = csv.writer(f, lineterminator='\n')

    shape = (lu, lm)
    R = sparse.lil_matrix(shape)

    shape_h = (lu + lm , lr + lf)
    H = sparse.lil_matrix(shape_h)
    c = np.zeros((lu,1))
    
    shape_w = (lr + lf, lr + lf)
    W = sparse.lil_matrix(shape_w)

    shape_dv = (lu + lm , lu + lm)
    D_v = sparse.lil_matrix(shape_dv)
    D_v_inv = sparse.lil_matrix(shape_dv)
    v_c1 = np.zeros(lu + lm )
    v_c2 = np.zeros(lu + lm )

    shape_de = (lr + lf, lr + lf)
    D_e = sparse.lil_matrix(shape_de) 
    D_e_inv = sparse.lil_matrix(shape_de)

    shape_a = (lu + lm , lu + lm )
    a1 = sparse.lil_matrix(shape_a)
    a2 = sparse.lil_matrix(shape_a)

    for i in range(lr):
        R[ratings[i,0] - 1,ratings[i,1] - 1] = ratings[i,2]
        H[ratings[i,0] - 1,i] = 1
        H[lu + ratings[i,1] - 1,i] = 1
        c[ratings[i,0] - 1] += 1
    sum1 = np.sum(R, axis=1) 
    sum2 = np.sum(R, axis=0)  
    ave = sum1 / c 


    for i in range(lr):
        W[i,i] = ratings[i,2] #/ (np.sqrt(sum1[ratings[i,0] - 1]) * np.sqrt(sum2[0,ratings[i,1] - 1]))
        W[i,i] /= ave[ratings[i,0] - 1]
        v_c1[ratings[i,0] - 1] += W[i,i]
        v_c1[lu + ratings[i,1] - 1] += W[i,i] 
        a1[ratings[i,0] - 1, ratings[i,0] - 1] += (W[i,i] /2)
        a1[ratings[i,0] - 1, lu + ratings[i,1] - 1] += (W[i,i] /2)
        a1[lu + ratings[i,1] - 1, ratings[i,0] - 1] += (W[i,i] /2)
        a1[lu + ratings[i,1] - 1, lu + ratings[i,1] - 1] += (W[i,i] /2)


    for i in range(lf):
        H[friends[i,0] - 1, lr + i] = 1
        H[friends[i,1] - 1, lr + i] = 1
        W[lr + i, lr + i] = 1 
        v_c2[friends[i,0] - 1] += W[lr + i, lr + i]
        v_c2[friends[i,1] - 1] += W[lr + i, lr + i]
        a2[friends[i,0] - 1, friends[i,0] - 1] += (W[i,i] /2)
        a2[friends[i,0] - 1, friends[i,1] - 1] += (W[i,i] /2)
        a2[friends[i,1] - 1, friends[i,0] - 1] += (W[i,i] /2)
        a2[friends[i,1] - 1, friends[i,1] - 1] += (W[i,i] /2)

    for i in range(lu + lm):
        D_v[i,i] = v_c1[i] + v_c2[i]

    e_c = np.sum(H, axis=0)
    for i in range(lr + lf):
        D_e[i,i] = e_c[0,i]
 
    for i in range(lu + lm):
        if D_v[i,i] != 0:
            D_v_inv[i,i] = np.reciprocal(D_v[i,i])
    for i in range(lr + lf):
        if D_e[i,i] != 0:
            D_e_inv[i,i] = np.reciprocal(D_e[i,i])
    
    #遷移行列を求める        
    A = D_v_inv * H * W * D_e_inv * H.T
    
    A = A.tolil()
    '''
    #e1
    for i in range(lu + lm):
        print("i: ",i)
        A[i,:] =  a1[i,:] / v_c1[i]
    '''
    #e2
    for i in range(lu):
        print("i: ",i)
        A[i,:lu] =  a2[i,:lu] / v_c2[i]
    A[:lu,lu:] = 0
    
    B = np.nan_to_num(A, copy=False)
    B = B.tocsr()
    #B = A.copy()
    for u in range(lu):
        print(u)
    
        t_u = u
        #クエリを立てる

        y = B[t_u].copy()
        y[0,t_u] = 1
        y = y / np.sum(y)
        f = y.copy()
        old_f = y.copy()

        #set a pararumeter
        c = pra
        t = ite

        #ランダムウォーク
        for i in range(t):
            f = (c * old_f * B) + ((1-c)*y)
            if(i != t-1):
                old_f = f.copy()
        '''
        ex_u = 0
        ex_m = 0
        
        for i in range(lu):
            ex_u += f[0,i] 
        for i in range(lm):
            ex_m += f[0,lu + i]
        '''
        ex_u = np.sum(f[0,:lu])
        ex_m = np.sum(f[0,lu:])
        kiyo[u,0] = ex_u
        kiyo[u,1] = ex_m

    for u in range(lu):
        with open('trust111_kiyo.csv','a') as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerow([kiyo[u,0],kiyo[u,1]])
    return  
