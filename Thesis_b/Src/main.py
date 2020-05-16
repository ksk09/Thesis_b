from module import inf
from module import trust_inf
from module import inf_kiyo
from module import trust_inf_kiyo

from module import div
import time
t1 = time.time()

c = 0.96
t = 80
n = 20
'''
#weight
lr =100836
lt = 3683

lrc = 284086
lfc= 111781

lre = 922267
lfe = 355813

wa = 1
wb = 1
#kfold_tag.kfold(u,c,t,n,wa,wb)
#kfold_tag_all.kfold(c,t,n,wa,wb)
#tag_rec.rec(u,c,t,wa,wb)
#div.rec(c,t,n,wa,wb)
#trust.rec(u,c,t,wa,wb)
#kfold_trust.kfold(c,t,n,wa,wb)
#tag_kiyo.rec(c,t)
'''
#inf.rec(c,t,n)
#trust_inf.rec(c,t,n)
#inf_kiyo.rec(c,t)
#trust_inf_kiyo.rec(c,t)
div.rec(c,t,n)
#kfold_trust.kfold(c,t,n,1,1)
t2 = time.time()
elapsed_time = t2 - t1
print(f'time: {elapsed_time}')