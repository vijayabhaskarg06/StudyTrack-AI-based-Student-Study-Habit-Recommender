import numpy as np

arr = np.array([[10, 20], [30, 40]])

print("Array:\n", arr)

# Matrix operations
print("\nMatrix Transpose:\n", arr.T)
print("\nMatrix Sum:", np.sum(arr))
print("Matrix Mean:", np.mean(arr))

# Dot product
arr2 = np.array([[2, 1], [1, 2]])
print("\nDot Product:\n", np.dot(arr, arr2))

# Broadcasting
arr3 = np.array([1, 2])
print("\nBroadcasting Addition:\n", arr + arr3)

# Random operations
print("\nRandom Array:\n", np.random.randint(1, 100, (3, 3)))
