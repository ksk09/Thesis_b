X = int(input())

a, X= divmod(X, 500)

b, X= divmod(X, 5)

print(a * 1000 + b * 5)
