#recommend for target user
def rec(pra,ite):  
    import pandas as pd
    import numpy as np
    from scipy import sparse
    import math
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
    lr = len(df_ratings)
    lt = len(df_tags)
    lta = len(tagg)

    df_movies['movieid'] = range(lm)
    dfr = pd.merge(df_movies, df_ratings, on = 'movieId')
    dR = dfr.loc[:,['userId','movieid','rating']]

    dft = pd.merge(dff,df_movies)
    dT = dft.loc[:,['userId','movieid','tag_num']]

    shape_kiyo = (lu,3)
    kiyo = np.zeros(shape_kiyo)

    with open('movielen_kiyo.csv','w') as f:
        writer = csv.writer(f, lineterminator='\n')

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
        R[dR.at[i,'userId'] - 1, dR.at[i,'movieid']] = dR.at[i,'rating']
        H[dR.at[i,'userId'] - 1,i] = 1
        H[lu + dR.at[i,'movieid'],i] = 1
        c[dR.at[i,'userId'] - 1] += 1
    sum1 = np.sum(R, axis=1) 
    sum2 = np.sum(R, axis=0)  
    ave = sum1 / c 


    for i in range(lr):
        W[i,i] = dR.at[i,'rating'] / (np.sqrt(sum1[dR.at[i,'userId'] - 1]) * np.sqrt(sum2[0,dR.at[i,'movieid']]))
        W[i,i] /= ave[dR.at[i,'userId'] - 1]
        v_c[dR.at[i,'userId'] - 1] += W[i,i]
        v_c[lu + dR.at[i,'movieid']] += W[i,i] 


    for i in range(lt):
        H[lu + lm + dT.at[i,'tag_num'], lr + i] = 1
        H[dT.at[i,'userId'] - 1,lr + i] = 1
        H[lu + dT.at[i,'movieid'] ,lr + i] = 1
        W[lr + i, lr + i] = 1 
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
    for u in range(lu):
        print(u)
        t_u = u
        
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

        ex_u = 0
        ex_m = 0
        ex_t = 0

        for i in range(lu):
            ex_u += f[0,i] 
        for i in range(lm):
            ex_m += f[0,lu + i]
        for i in range(lta):
            ex_t += f[0,lu + lm + i]
        kiyo[u,0] = ex_u
        kiyo[u,1] = ex_m
        kiyo[u,2] = ex_t

    for u in range(lu):
        with open('movielen_kiyo.csv','a') as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerow([kiyo[u,0],kiyo[u,1],kiyo[u,2]])   

    return