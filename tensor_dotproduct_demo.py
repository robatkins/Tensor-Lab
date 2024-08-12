import tensorflow as tf

#Tensor Dot-product Demonstration for Tensor up to the 4th-Order.

def print_tensor(tensor, order):
    print(f"Order {order} Tensor with shape {tensor.shape}:\n{tensor.numpy()}\n")

# Dot product function for different orders
def dot_product(tensor1, tensor2):
    return tf.tensordot(tensor1, tensor2, axes=1)

# 0th-order tensor (Scalar)
tensor_0th_1 = tf.constant(5)
tensor_0th_2 = tf.constant(3)
result_0th = tf.multiply(tensor_0th_1, tensor_0th_2)
print_tensor(result_0th, 0)

# 1st-order tensor (Vector)
tensor_1st_1 = tf.constant([1, 2, 3])
tensor_1st_2 = tf.constant([4, 5, 6])
result_1st = dot_product(tensor_1st_1, tensor_1st_2)
print_tensor(result_1st, 1)

# 2nd-order tensor (Matrix)
tensor_2nd_1 = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor_2nd_2 = tf.constant([[7, 8], [9, 10], [11, 12]])
result_2nd = tf.matmul(tensor_2nd_1, tensor_2nd_2)
print_tensor(result_2nd, 2)

# 3rd-order tensor (Array of matrices)
tensor_3rd_1 = tf.random.uniform([2, 2, 3])
tensor_3rd_2 = tf.random.uniform([2, 3, 2])
result_3rd = tf.tensordot(tensor_3rd_1, tensor_3rd_2, axes=[[2], [1]])
print_tensor(result_3rd, 3)

# 4th-order tensor (Array of 3rd-order tensors)
tensor_4th_1 = tf.random.uniform([2, 2, 2, 3])
tensor_4th_2 = tf.random.uniform([2, 2, 3, 2])
result_4th = tf.tensordot(tensor_4th_1, tensor_4th_2, axes=[[3], [2]])
print_tensor(result_4th, 4)