import tensorflow as tf

def print_tensor(tensor, order):
    print(f"Order {order} Tensor:\n{tensor.numpy()}\n")

# 0th-order tensor (Scalar)
tensor_0th = tf.constant(5)
print_tensor(tensor_0th, 0)

# 1st-order tensor (Vector)
tensor_1st = tf.constant([1, 2, 3])
print_tensor(tensor_1st, 1)

# 2nd-order tensor (Matrix)
tensor_2nd = tf.constant([[1, 2, 3], [4, 5, 6]])
print_tensor(tensor_2nd, 2)

# 3rd-order tensor (Array of Matrices)
tensor_3rd = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print_tensor(tensor_3rd, 3)

# 4th-order tensor (Array of 3rd-order Tensors, or a 4D-Hypercube of Data.)
tensor_4th = tf.random.uniform([2, 2, 2, 2])
print_tensor(tensor_4th, 4)