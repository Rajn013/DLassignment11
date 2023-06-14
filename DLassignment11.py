#!/usr/bin/env python
# coding: utf-8

# Write the Python code to implement a single neuron.
# 

# In[ ]:


import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
    
    def forward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        output = self.activation(total)
        return output
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))


# Write the Python code to implement ReLU.
# 

# In[ ]:


def relu(x):
    return np.maximun(0, x)


# Write the Python code for a dense layer in terms of matrix multiplication.
# 

# In[ ]:


class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(input_dim, output_dim)
        self.bias = np.random.rand(output_dim)
    
    def forward(self, inputs):
        output = np.dot(inputs, self.weights) + self.bias
        return output


# Write the Python code for a dense layer in plain Python (that is, with list comprehensions and functionality built into Python).
# 

# In[ ]:


class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = [[random.random() for _ in range(input_dim)] for _ in range(output_dim)]
        self.bias = [random.random() for _ in range(output_dim)]
    
    def forward(self, inputs):
        output = [sum(w * x for w, x in zip(weights, inputs)) + b for weights, b in zip(self.weights, self.bias)]
        return output


# What is the “hidden size” of a layer?
# 

# The "hidden size" of a layer refers to the number of neurons or units in that layer. It determines the layer's capacity to learn and represent complex patterns in the data. The choice of hidden size is a hyperparameter that affects the model's complexity and capacity. It is typically set by the user based on the specific task and available computational resources.

# What does the t method do in PyTorch?
# 

# In PyTorch, the t method is used to transpose a tensor. It returns a new tensor that has the dimensions permuted or transposed based on the specified permutation order.

# Why is matrix multiplication written in plain Python very slow?
# 

# Matrix multiplication implemented in plain Python is slow compared to optimized libraries due to factors such as lack of vectorization, interpretation overhead, suboptimal memory access patterns, and the absence of hardware acceleration. Optimized libraries like NumPy, TensorFlow, and PyTorch utilize techniques such as vectorization, memory optimizations, and hardware acceleration to significantly improve the performance of matrix multiplication operations.

# In matmul, why is ac==br?
# 

# In matrix multiplication (matmul), the condition ac == br ensures that the two matrices have compatible dimensions for multiplication. It means that the number of columns in the first matrix (c) must be equal to the number of rows in the second matrix (b). This condition ensures that the operation is well-defined and allows the multiplication to be performed. If ac is not equal to br, the matrix multiplication is not valid.

# In Jupyter Notebook, how do you measure the time taken for a single cell to execute?
# 

# To measure the time taken for a single cell to execute in Jupyter Notebook, you can use %timeit for a single line of code or %%timeit for an entire cell. The execution time will be displayed along with other statistics.

# What is elementwise arithmetic?
# 

# Elementwise arithmetic refers to performing arithmetic operations on corresponding elements of arrays or tensors. It allows for efficient manipulation of arrays and tensors, enabling operations on large datasets in a parallelized manner. It is widely used in numerical computations, machine learning, and scientific computing.

# Write the PyTorch code to test whether every element of a is greater than the corresponding element of b.
# 

# In[ ]:


def test_greater(a, b):
     if len(a) != len(b):
        return False
     for i in range(len(a)):
            if a[i] <= b[i]:
                return False
     
     return True

a = [1, 2, 3, 4]
b = [0, 1, 2, 3]

is_greater = test_greater(a, b)
print(is_greater)


# What is a rank-0 tensor? How do you convert it to a plain Python data type?
# 

# A rank-0 tensor is a scalar tensor with no dimensions or axes, representing a single value. To convert a rank-0 tensor to a plain Python data type:
# 
# Use the .item() method on the tensor.
# Assign the extracted value to a new variable, which will be a plain Python data type.
# You can then use this variable in Python calculations, comparisons, or print statements.

# How does elementwise arithmetic help us speed up matmul?
# 

# Elementwise arithmetic operations enable optimizations such as vectorization and parallelization, which exploit hardware capabilities to speed up matrix multiplication by performing calculations on multiple elements simultaneously and efficiently utilizing memory access patterns.

# What are the broadcasting rules?

# broadcasting rules allow for elementwise operations between arrays of different shapes. The rules state that arrays are compatible for broadcasting if their dimensions are equal or one of them has a dimension of size 1. The arrays are aligned by stretching or repeating dimensions with size 1, and the resulting broadcasted array has dimensions with the maximum sizes from the input arrays. Broadcasting is not allowed if corresponding dimensions do not have the same size or if one of them does not have a size of 1.

# What is expand_as? Show an example of how it can be used to match the results of broadcasting.
#     

# expand_as() is a method in PyTorch that expands the dimensions of a tensor to match the shape of another tensor. It is used to align tensors for elementwise operations when broadcasting. It does not replicate or repeat values along the expanded dimensions.
# 

# In[ ]:




