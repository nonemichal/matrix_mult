import numpy as np

# Load matrices saved from C program
A = np.load("matrix_a.npy")
B = np.load("matrix_b.npy")
C = np.load("matrix_c.npy")

# Multiply in Python
C_np = A @ B

# Check if they are close
if np.allclose(C, C_np, atol=1e-12):
    print("The matrices match")
else:
    print("The matrices does not match")
