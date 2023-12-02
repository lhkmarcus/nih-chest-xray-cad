import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
print("TensorFlow version:", tf.__version__)  # check version

import numpy as np

# # Zero-dimensional tensor -- scalar
# tensor_0d = tf.constant(4)
# print(tensor_0d)

# # One-dimensional tensor -- vector
# tensor_1d = tf.constant(
#     [2, 3, 4, 1]
# )
# print(tensor_1d)

# # Two-dimensional tensor -- matrix
# tensor_2d = tf.constant(
#     [
#         [1, 2, 9],
#         [4, 2, 3]
#     ]
# )
# print(tensor_2d)

# # Three-dimensional tensor -- matrix
# tensor_3d = tf.constant(
#     [
#         [[1, 1, 5],
#          [4, 5, 1]],
#         [[4, 6, 2],
#          [4, 3, 6]],
#         [[4, 1, 2],
#          [7, 4, 8]]
#     ]
# )
# print(tensor_3d)

# # Four-dimensional tensor -- matrix
# tensor_4d = tf.constant(
#     [
#         [
#             [[1, 1, 5],
#              [4, 5, 1]],
#             [[4, 6, 2],
#              [4, 3, 6]],
#             [[4, 1, 2],
#              [7, 4, 8]]
#         ],
#         [
#             [[1, 1, 5],
#              [4, 5, 1]],
#             [[4, 6, 2],
#              [4, 3, 6]],
#             [[4, 1, 2],
#              [7, 4, 8]]
#         ]
#     ]
# )
# print(tensor_4d)

# casted_tensor_1d = tf.cast(tensor_1d, dtype=tf.int16)
# print(casted_tensor_1d)

# bool_tensor_1d = tf.cast(tensor_1d, dtype=bool)
# print(bool_tensor_1d)

# string_tensor_0d = tf.constant("hello world")
# print(string_tensor_0d)

# string_tensor_1d = tf.constant(
#     ["hello world", "it's me"]
# )
# print(string_tensor_1d)

# np_array = np.array([1, 2, 3])
# print("NumPy array:", np_array)

# converted_tensor = tf.convert_to_tensor(np_array)
# print("Tensor from array:", converted_tensor)

# # Identity matrix
# eye_tensor = tf.eye(
#     num_rows=5,
#     num_columns=5,
#     batch_shape=[2,2],
#     dtype=tf.dtypes.float32,
#     name=None
# )
# print("Identity tensor:", eye_tensor)

# # Fill specified tensor shape with value
# fill_tensor = tf.fill(
#     [1, 3, 5], 5, name=None
# )
# print(fill_tensor)

# # Tensor with ones
# ones_tensor = tf.ones(
#     [5, 3],
#     dtype=tf.dtypes.float32,
#     name=None
# )
# print(ones_tensor)

# ones_like_tensor = tf.ones_like(fill_tensor)
# print(ones_like_tensor)

# zeros_tensor = tf.zeros(
#     [3, 2],
#     dtype=tf.dtypes.float32,
#     name=None
# )
# print(zeros_tensor)

# Four-dimensional tensor -- matrix
# tensor_4d = tf.constant(
#     [
#         [
#             [[1, 1, 5],
#              [4, 5, 1]],
#             [[4, 6, 2],
#              [4, 3, 6]],
#             [[4, 1, 2],
#              [7, 4, 8]]
#         ],
#         [
#             [[1, 1, 5],
#              [4, 5, 1]],
#             [[4, 6, 2],
#              [4, 3, 6]],
#             [[4, 1, 2],
#              [7, 4, 8]]
#         ]
#     ]
# )
# print(tensor_4d)
# print("Rank:", tf.rank(tensor_4d))
# print("Size:", tf.size(tensor_4d))

# random_normal_tensor = tf.random.normal(
#     shape=[5, 2],
#     mean=0.0,
#     stddev=1.0,
#     dtype=tf.dtypes.float32,
#     seed=42,
#     name=None
# )
# print(tf.reduce_mean(random_normal_tensor))

random_uniform_tensor = tf.random.uniform(
    [5,],
    minval=0,
    maxval=100,
    dtype=tf.dtypes.float32,
    seed=42,
    name=None
)

print(random_uniform_tensor)