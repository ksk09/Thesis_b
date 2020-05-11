K = int(input())
c = 0
import math
for i in range(1,K+1):
    for j in range(1,K+1):
        ab = math.gcd(i,j)
        for k in range(1,K+1):
            c += math.gcd(ab,k)
            

print(c)
