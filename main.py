import time
import numpy as np
import torch
import tensorflow as tf
import jax.numpy as jnp


def benchmark_numpy(A, B):
    start = time.time()
    C = A @ B
    end = time.time()
    return end - start, C


def benchmark_torch(A, B):
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    start = time.time()
    C_t = A_t @ B_t
    end = time.time()
    return end - start, C_t


def benchmark_tensorflow(A, B):
    A_tf = tf.convert_to_tensor(A)
    B_tf = tf.convert_to_tensor(B)
    start = time.time()
    C_tf = tf.matmul(A_tf, B_tf)
    end = time.time()
    return end - start, C_tf


def benchmark_jax(A, B):
    A_j = jnp.array(A)
    B_j = jnp.array(B)
    start = time.time()
    C_j = jnp.matmul(A_j, B_j).block_until_ready()
    end = time.time()
    return end - start, C_j


if __name__ == "__main__":
    A = np.load("matrix_a.npy")
    B = np.load("matrix_b.npy")

    print(f"Matrix A: {A.shape}, Matrix B: {B.shape}")

    t_numpy, C_numpy = benchmark_numpy(A, B)
    print(f"NumPy time: {t_numpy:.4f} s")

    t_torch, C_torch = benchmark_torch(A, B)
    print(f"PyTorch (CPU) time: {t_torch:.4f} s")

    t_tf, C_tf = benchmark_tensorflow(A, B)
    print(f"TensorFlow (CPU) time: {t_tf:.4f} s")

    t_jax, C_jax = benchmark_jax(A, B)
    print(f"JAX (CPU) time: {t_jax:.4f} s")
