#recommend for target user
def rec(u,pra,ite,wa,wb):  
    import pandas as pd
    import numpy as np
    from scipy import sparse
    from scipy.io import loadmat
    import math
    
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
    v_c = np.zeros(lu + lm )

    shape_de = (lr + lf, lr + lf)
    D_e = sparse.lil_matrix(shape_de) 
    D_e_inv = sparse.lil_matrix(shape_de)
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
        W[i,i] *= wa
        v_c[ratings[i,0] - 1] += W[i,i]
        v_c[lu + ratings[i,1] - 1] += W[i,i] 


    for i in range(lf):
        H[friends[i,0] - 1, lr + i] = 1
        H[friends[i,1] - 1, lr + i] = 1
        W[lr + i, lr + i] = 1 * wb
        v_c[friends[i,0] - 1] += W[lr + i, lr + i]
        v_c[friends[i,1] - 1] += W[lr + i, lr + i]

    for i in range(lu + lm):
        D_v[i,i] = v_c[i]

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
  
    t_u = u
    #クエリを立てる

    y = A[t_u].copy()


    y[0,t_u] = 1
    y = y / np.sum(y)
    f = y.copy()
    old_f = y.copy()

    #set a pararumeter
    c = pra
    t = ite

    #ランダムウォーク
    for i in range(t):
        f = (c * old_f * A) + ((1-c)*y)
        if(i != t-1):
            old_f = f.copy()
        #print(np.sum(f,axis=1))

    #recommend list
    rec = np.zeros(lm)
    #評価済みを除く
    for i in range(lm):
        if R[t_u,i] == 0:
            rec[i] = f[0,lu + i]
        else:
            rec[i] = 0
        
    
    for i in range(20):
        #print(np.argsort(-rec)[i])
        print('I recommend ' + str( np.argsort(-rec)[i]) )


    #explainable

    ex_u = 0
    ex_m = 0

    for i in range(lu):
        ex_u += f[0,i] 
    for i in range(lm):
        ex_m += f[0,lu + i]

    print('u: ' + str(round(ex_u  * 100,3)), 'm: ' + str(round(ex_m  * 100,3)))
    
    return  
