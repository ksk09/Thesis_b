#recommend for target user
def rec(pra,ite,n):  
    import pandas as pd
    import numpy as np
    from scipy import sparse
    from scipy.io import loadmat
    import math
    from sklearn.model_selection import KFold
    import csv
    
    #ciao
    matrate = loadmat(u'ciao/rating.mat')
    ratings = matrate['rating']
    ratings = np.delete(ratings,[2,4],1)
    np.random.seed(1)
    np.random.shuffle(ratings)
    dR = pd.DataFrame(ratings)
    matfri = loadmat(u'ciao/trustnetwork.mat')
    friends = matfri['trustnetwork']

    lu = 7375
    lm = 106797
    lf = friends.shape[0]
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

    ndcg_sum = np.zeros(lu)
    
    kf = KFold(n_splits=5,shuffle=True,random_state=24)

    with open('trust3.csv','w') as f:
        writer = csv.writer(f, lineterminator='\n')
    co = 0
    for train, test in kf.split(dR):
        train_df = dR.iloc[train]
        train_df = train_df.reset_index(drop=True)
        test_df = dR.iloc[test]
        test_df = test_df.reset_index(drop=True)
        lr = len(train_df)
        lrt = len(test_df)

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
            R[train_df.at[i,0] - 1, train_df.at[i,1] - 1] = train_df.at[i,2]
            H[train_df.at[i,0] - 1,i] = 1
            H[lu + train_df.at[i,1] - 1,i] = 1
            c[train_df.at[i,0] - 1] += 1
    
        sum1 = np.sum(R, axis=1) 
        sum2 = np.sum(R, axis=0)  
        ave = sum1 / c 


        for i in range(lr):
            W[i,i] = train_df.at[i,2] 
            W[i,i] /= ave[train_df.at[i,0] - 1]
            v_c1[train_df.at[i,0] - 1] += W[i,i]
            v_c1[lu + train_df.at[i,1] - 1] += W[i,i] 
            a1[train_df.at[i,0] - 1, train_df.at[i,0] - 1] += (W[i,i] /2)
            a1[train_df.at[i,0] - 1, lu + train_df.at[i,1] - 1] += (W[i,i] /2)
            a1[lu + train_df.at[i,1] - 1, train_df.at[i,0] - 1] += (W[i,i] /2)
            a1[lu + train_df.at[i,1] - 1, lu + train_df.at[i,1] - 1] += (W[i,i] /2)



        for i in range(lf):
            H[friends[i,0] - 1, lr + i] = 1
            H[friends[i,1] - 1, lr + i] = 1
            W[lr + i, lr + i] = 1 
            v_c2[friends[i,0] - 1] += W[lr + i, lr + i]
            v_c2[friends[i,1] - 1] += W[lr + i, lr + i]
            a2[train_df.at[i,0] - 1, train_df.at[i,0] - 1] += (W[i,i] /2)
            a2[train_df.at[i,0] - 1, train_df.at[i,1] - 1] += (W[i,i] /2)
            a2[train_df.at[i,1] - 1, train_df.at[i,0] - 1] += (W[i,i] /2)
            a2[train_df.at[i,1] - 1, train_df.at[i,1] - 1] += (W[i,i] /2)


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
        '''
        B = np.nan_to_num(A, copy=False)
        B = B.tocsr()
        #B = A.copy()

        shape_z = (lu, lm)
        Z = sparse.lil_matrix(shape)

        for i in range(lrt):
            Z[test_df.at[i,0] - 1, test_df.at[i,1] - 1,]  = test_df.at[i,2]
        
        for u in range(lu):
            print(str(co) + ': u: %d'%(u))
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

            #test_dataのtarget_userのratings
            z = np.zeros(lm) 
            z = Z[u]
            z = z.todense()
            z = np.ravel(z)

            x = f[0,lu:].copy()
            x = x.todense()
            x = np.ravel(x)

            x2 = np.zeros(lm)
            x2[np.argsort(x)[::-1]] = z

            x3 = x2[x2 > 0]
            ln = len(x3)
    
            #check efiiciency

            rate1 = np.zeros(n)
            rate2 = np.zeros(n)
            
            for i in range(n):
                if(i < ln):
                    rate1[i] = x3[i]
                else:
                    rate1[i] = 0
                rate2[i] = np.sort(z)[::-1][i]

            dcg = 0
            idcg = 0

            for i in range(n):
                dcg += (2**rate1[i] - 1)/ math.log(i+2,2)
                idcg += (2**rate2[i] - 1)/ math.log(i+2,2)
            ndcg = dcg / idcg

            #print('ndcg: '+ str(ndcg))
            ndcg_sum[u] += ndcg   
        co += 1
    ndcg_sum /=5

    for u in range(lu):
        with open('trust3.csv','a') as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerow([ndcg_sum[u]])     

    return  
