K, N = map(int, input().split())

A = list(map(int, input().split()))
A.sort()

div = [A[i+1] - A[i] for i in range(N-1)]
div.append((K - A[N-1]) + A[0])
print(K - max(div))
