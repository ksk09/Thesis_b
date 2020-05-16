def rec():  
    import pandas as pd
    import numpy as np
    from scipy import sparse
    import json
    '''
    frames_tip = []
    for chunk in pd.read_json('yelp/tip.json', lines=True, chunksize = 10000):
        frames_tip.append(chunk)
    tip=pd.concat(frames_tip)
    
    frames_checkin = []
    for chunk in pd.read_json('yelp/checkin.json', lines=True, chunksize = 10000):
        frames_checkin.append(chunk)
    checkin=pd.concat(frames_checkin)
    '''
    frames = []
    for chunk in pd.read_json('yelp/user.json', lines=True, chunksize = 10000):
        frames.append(chunk)
    user = pd.concat(frames)
    user = user.loc[:,['user_id','friends']]
    user['uid'] = range(len(user))
        
    lu = len(user)
    '''
    frames_business = []
    for chunk in pd.read_json('yelp/business.json', lines=True, chunksize = 10000):
        frames_business.append(chunk)
    business = pd.concat(frames_business)
    business = business.loc[:,['business_id','categories']]
    business['bid'] = range(len(business))
    
    frames_review = []
    for chunk in pd.read_json('yelp/review.json', lines=True, chunksize = 20000):
        frames_review.append(chunk)
    review=pd.concat(frames_review)
    review = review.loc[:,['user_id','business_id','stars']]
    
    #lb = len(business)
    #lr = len(review)
    
    dr = pd.merge(user,review)
    dR = pd.merge(dr,business)

    dR = dR.loc[:,['uid','bid','stars']]
    '''
    friends = user.copy()
    friends = friends[friends['friends'] != 'None']

    friends['friends'] = friends['friends'].apply(lambda x : x.split(','))
    
    friends['number'] = friends['friends'].apply(len)
    lf = len(friends)
    res = friends.set_index(['user_id'])['friends'].apply(pd.Series).stack()
    res = res.reset_index()
    #num = user[user['user_id'] ==  friends.loc[5].friends[4]].uid
    print(res)

    '''
    shape_u = (lu, lu)
    U = sparse.lil_matrix(shape_u)
    
    for i in range(lf):
        print(i)
        lfn = friends.iloc[i,3]
        for j in range(lfn):
            num = user[user['user_id'] ==  friends.loc[i].friends[j]].uid
            s = friends.iloc[i,2]
            t = num.iloc[0]
            U[s,t] = 1
    '''
    return