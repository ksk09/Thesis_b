#recommend for target user
def kfold(pra,ite,n,wa,wb):  
    import pandas as pd
    import numpy as np
    from scipy import sparse
    import math
    from sklearn.model_selection import KFold
    import csv

    #データの取得
    df_tags = pd.read_csv('ml-latest-small/tags.csv')
    df_movies = pd.read_csv('ml-latest-small/movies.csv')
    df_movies = df_movies.drop('genres',axis=1)
    df_ratings = pd.read_csv('ml-latest-small/ratings.csv')

    df_tags_combined = df_tags.groupby('movieId').apply(lambda x: list(x['tag'])).reset_index().rename(columns={0:'tags'})
    dfm = pd.merge(df_movies, df_tags_combined, on = 'movieId', how = 'left')
    dfm['tags'] = dfm['tags'].apply(lambda x: x if isinstance(x,list) else [])
    all_tags = set()
    for this_movie_tags in dfm['tags']:
        all_tags = all_tags.union(this_movie_tags)
    all_tags = list(all_tags)
    tagg = pd.DataFrame(all_tags,columns = ['tag'])
    tagg['tag_num'] = range(len(tagg))
    dff = pd.merge(df_tags,tagg)

    lu = 610
    lm = len(df_movies)
    lt = len(df_tags)
    lta = len(tagg)

    df_movies['movieid'] = range(lm)
    dfr = pd.merge(df_movies, df_ratings, on = 'movieId')
    dR = dfr.loc[:,['userId','movieid','rating']]
    dR = dR.sample(frac=1, random_state=0)

    ndcg_sum = np.zeros(lu)
    kf = KFold(n_splits=5,shuffle=True,random_state=24)

    dft = pd.merge(dff,df_movies)
    dT = dft.loc[:,['userId','movieid','tag_num']]
    co = 0

    with open('movielen2.csv','w') as f:
        writer = csv.writer(f, lineterminator='\n')

    for train, test in kf.split(dR):
        print(co)
        train_df = dR.iloc[train]
        train_df = train_df.reset_index(drop=True)
        test_df = dR.iloc[test]
        test_df = test_df.reset_index(drop=True)
        lr = len(train_df)
        lrt = len(test_df)

        shape = (lu, lm)
        R = sparse.lil_matrix(shape)

        shape_h = (lu + lm + lta, lr + lt)
        H = sparse.lil_matrix(shape_h)
        c = np.zeros((lu,1))
        
        shape_w = (lr + lt, lr + lt)
        W = sparse.lil_matrix(shape_w)

        shape_dv = (lu + lm + lta, lu + lm + lta)
        D_v = sparse.lil_matrix(shape_dv)
        D_v_inv = sparse.lil_matrix(shape_dv)
        v_c = np.zeros(lu + lm + lta)

        shape_de = (lr + lt, lr + lt)
        D_e = sparse.lil_matrix(shape_de) 
        D_e_inv = sparse.lil_matrix(shape_de)
        
        for i in range(lr):
            R[train_df.at[i,'userId'] - 1, train_df.at[i,'movieid']] = train_df.at[i,'rating']
            H[train_df.at[i,'userId'] - 1,i] = 1
            H[lu + train_df.at[i,'movieid'],i] = 1
            c[train_df.at[i,'userId'] - 1] += 1
        sum1 = np.sum(R, axis=1) 
        sum2 = np.sum(R, axis=0)  
        ave = sum1 / c 
        
        for i in range(lr):
            W[i,i] = train_df.at[i,'rating'] / (np.sqrt(sum1[train_df.at[i,'userId'] - 1]) * np.sqrt(sum2[0,train_df.at[i,'movieid']]))
            W[i,i] /= ave[train_df.at[i,'userId'] - 1]
            W[i,i] *= wa
            v_c[train_df.at[i,'userId'] - 1] += W[i,i]
            v_c[lu + train_df.at[i,'movieid']] += W[i,i] 


        for i in range(lt):
            H[lu + lm + dT.at[i,'tag_num'], lr + i] = 1
            H[dT.at[i,'userId'] - 1,lr + i] = 1
            H[lu + dT.at[i,'movieid'] ,lr + i] = 1
            W[lr + i, lr + i] = 1 * wb
            v_c[lu + lm + dT.at[i,'tag_num']] += W[lr + i, lr + i]
            v_c[dT.at[i,'userId'] - 1] += W[lr + i, lr + i]
            v_c[lu + dT.at[i,'movieid']] += W[lr + i, lr + i]
        
        for i in range(lu + lm + lta):
            D_v[i,i] = v_c[i]

        e_c = np.sum(H, axis=0)
        for i in range(lr + lt):
            D_e[i,i] = e_c[0,i]

        for i in range(lu + lm + lta):
            if D_v[i,i] != 0:
                D_v_inv[i,i] = np.reciprocal(D_v[i,i])
        for i in range(lr + lt):
            if D_e[i,i] != 0:
                D_e_inv[i,i] = np.reciprocal(D_e[i,i])
        
        #遷移行列を求める        
        A = D_v_inv * H * W * D_e_inv * H.T

        shape_z = (lu, lm)
        Z = sparse.lil_matrix(shape)
        for i in range(lrt):
            Z[test_df.at[i,'userId'] - 1, test_df.at[i,'movieid']]  = test_df.at[i,'rating']
        
        for u in range(lu):
            print('u: %d'%(u))
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

            x = f[0,lu:lu+lm].copy()
            x = x.todense()
            x = np.ravel(x)
            
            #test_dataのtarget_userのratings
            z = np.zeros(lm)
            
            z = Z[u]
            z = z.todense()
            z = np.ravel(z)

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
                #print(rate1[i],rate2[i])
            
            dcg = 0
            idcg = 0
            
            for i in range(n):
                dcg += (2**rate1[i] - 1)/ math.log(i+2,2)
                idcg += (2**rate2[i] - 1)/ math.log(i+2,2)

            ndcg = dcg / idcg

            #print('ndcg: '+ str(ndcg))
            ndcg_sum[u] += ndcg
        co += 1
    ndcg_sum /= 5

    for u in range(lu):
        with open('movielen2.csv','a') as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerow([ndcg_sum[u]])     

    
    return