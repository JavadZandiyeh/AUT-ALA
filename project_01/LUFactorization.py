import numpy as np

# setting precision 1
np.set_printoptions(1)

n, m = map(int, input().split())
entry = [list(map(int, input().split())) for i in range(n)]
A = np.array(entry, dtype=np.double)
b = [np.array(list(map(int, input().split())), dtype=np.double).reshape(-1, 1) for j in range(m)]


def LU(A):
    # number of rows in matrix A
    num_rows = A.shape[0]
    U = A.copy()
    # create L as identity matrix
    L = np.eye(n, dtype=np.double)
    for i in range(num_rows):
        x = U[i+1:, i] / U[i, i]
        L[i+1:, i] = x
        # making x in a new axis and...
        U[i+1:] -= x.reshape(-1, 1)*U[i]
    return L, U


L, U = LU(A)


def solving_Ly_equal_b(L, b):
    num_rows_L = L.shape[0]
    # initializing y
    y = np.zeros((L.shape[1], b.shape[1]), dtype=np.double)
    for i in range(num_rows_L):
        product = np.dot(L[i, :i], y[:i])
        y[i] = b[i] - product
    return y


def solving_Ux_equal_y(U, y):
    num_rows_U = U.shape[0]
    # initializing x
    x = np.zeros((U.shape[1], y.shape[1]), dtype=np.double)
    x[-1] = y[-1] / U[-1, -1]
    for i in range(num_rows_U-2, -1, -1):
        product = np.dot(U[i, i:], x[i:])
        x[i] = (y[i] - product) / U[i, i]
    return x


for i in range(m):
    y = solving_Ly_equal_b(L, b[i])
    x = solving_Ux_equal_y(U, y)
    for j in range(n):
        print(round(x[j, 0], 2), end=' ')
    print()
