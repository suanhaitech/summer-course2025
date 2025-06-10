import numpy as np
from scipy.sparse import spdiags
import sympy as syp
import time

# 1   实现矩阵乘
A = np.arange(1, 11).reshape(2, 5)
# print(A)
B = np.einsum("ij->ji", A)
# print(B)

C = np.einsum("ij,jk->ik", A, B)
print(C)


# 2
# 生成矩阵
n = 1000
main_diag = 2 * np.ones(n)
sub_diag = -np.ones(n - 1)
super_diag = -np.ones(n - 1)
data = np.zeros((3, n))
data[0, 1:] = sub_diag
data[1, :] = main_diag
data[2, :-1] = super_diag
diagonals = [-1, 0, 1]
A = spdiags(data, diagonals, n, n)


start = time.perf_counter()
np.dot(A, A)
total = time.perf_counter() - start
print("稀疏矩阵乘用时{}".format(total))

B = A.todense()
start = time.perf_counter()
np.dot(B, B)
total = time.perf_counter() - start
print("稠密矩阵乘用时{}".format(total))


# 3
x, y, z = syp.symbols("x y z")
f = y * syp.cos(3 * x + 2 * z) + x * syp.exp(y * z)

# Hessian阵对角线元素
f_xx = syp.diff(f, x, 2)
f_yy = syp.diff(f, y, 2)
f_zz = syp.diff(f, z, 2)
print(f_xx, f_yy, f_zz)

# Hessian阵其余元素
f_xy = syp.diff(f, x, y)
f_xz = syp.diff(f, x, z)
f_yx = syp.diff(f, y, x)
f_yz = syp.diff(f, y, z)
f_zx = syp.diff(f, z, x)
f_zy = syp.diff(f, z, y)
print(f_xy, f_xz, f_yx, f_yz, f_zx, f_zy)

H = syp.hessian(f, [x, y, z])
print(H)
