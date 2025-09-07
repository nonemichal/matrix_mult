import numpy as np

A = np.load("matrix_a.npy")
B = np.load("matrix_b.npy")
C = np.load("matrix_c.npy")

C_np = A @ B

if np.allclose(C, C_np, atol=1e-12):
    print("The matrices match")
else:
    print("The matrices does not match")
