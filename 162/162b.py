N = int(input())
c = 0
for i in range(N):
    n = i + 1
    if(n % 3 != 0 and n % 5 != 0):
        c += n

print(c)
