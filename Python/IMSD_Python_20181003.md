
# 4. Numpy: arrays and matrices

### 4.1 Create arrays


```python
import numpy as np

data1 = [1, 2, 3, 4, 5]
arr1 = np.array(data1)
data2 = [range(1,5), range(5,9)]
arr2 = np.array(data2)
print(arr1)
print(arr2)
print(type(arr1))
print(type(arr2))
```

    [1 2 3 4 5]
    [[1 2 3 4]
     [5 6 7 8]]
    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>
    


```python
print(np.zeros(10)) 
print(np.zeros((3, 6)))
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [[0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]]
    


```python
np.linspace(0, 1, 5)     # 0 to 1 with 5 points
```




    array([0.  , 0.25, 0.5 , 0.75, 1.  ])




```python
np.logspace(0, 3, 4)     # 10^0 to 10^3 with 4 points 
```




    array([   1.,   10.,  100., 1000.])




```python
np.logspace(0, 3, 5)  
```




    array([   1.        ,    5.62341325,   31.6227766 ,  177.827941  ,
           1000.        ])




```python
np.arange(10)  # it's not a list, it's an array
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
int_array = np.arange(5)
float_array = int_array.astype(float)
float_array
```




    array([0., 1., 2., 3., 4.])




```python
print(arr1.dtype)
print(arr2.dtype)
print(arr2.ndim) 
print(arr2.shape)
print(arr2.size)  # the number of values
len(arr2)
```

    int32
    int32
    2
    (2, 4)
    8
    




    2



### 4.3 Reshaping


```python
arr = np.arange(10, dtype=float).reshape((2,5))
print(arr)
print(arr.shape)
print(arr.reshape(5, 2))
print(arr.reshape(1, 10))
```

    [[0. 1. 2. 3. 4.]
     [5. 6. 7. 8. 9.]]
    (2, 5)
    [[0. 1.]
     [2. 3.]
     [4. 5.]
     [6. 7.]
     [8. 9.]]
    [[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]]
    


```python
a = np.array([0, 1])
a_col = a[:, np.newaxis] 
# newaxis: increase the dimension of the existing array 
# by one more dimension: 1D > 2D, 2D > 3D
print(a_col)
a_col2 = a[:, None]
print(a_col2)
```

    [[0]
     [1]]
    [[0]
     [1]]
    


```python
arr
```




    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])




```python
arr.T    # Change the order of matrice
```




    array([[0., 5.],
           [1., 6.],
           [2., 7.],
           [3., 8.],
           [4., 9.]])




```python
arr_flt = arr.flatten()   
# Function flatten allows to copy the array and 
# make a flat matrice with it. matice (1, 0)
arr_flt[0] = 33
print('arr_flt :', '\n', arr_flt)
print('arr :', '\n', arr)
```

    arr_flt : 
     [33.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
    arr : 
     [[0. 1. 2. 3. 4.]
     [5. 6. 7. 8. 9.]]
    


```python
arr_flt = arr.ravel()    
# Function Ravel also allows to bring the original array and 
# make it a flat matrice but it does impact the original value
arr_flt[0] = 33
print('arr_flt :', '\n', arr_flt)
print('arr :', '\n', arr)
```

    arr_flt : 
     [33.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
    arr : 
     [[33.  1.  2.  3.  4.]
     [ 5.  6.  7.  8.  9.]]
    

### 4.4 Stak arrays


```python
a = np.array([0, 1])
b = np.array([2, 3])

ab = np.stack((a, b))  # Stack: pile up the matrices, axis=0(default): horizontal
ab2 = ab.T

print(ab)
print(ab2)
print(type(ab2))
```

    [[0 1]
     [2 3]]
    [[0 2]
     [1 3]]
    <class 'numpy.ndarray'>
    

### 4.5 Selection


```python
arr = np.arange(10, dtype=float).reshape((2, 5))
print(arr)
print(arr[0])
print(arr[0, 3])
print(arr[0][3])
```

    [[0. 1. 2. 3. 4.]
     [5. 6. 7. 8. 9.]]
    [0. 1. 2. 3. 4.]
    3.0
    3.0
    

### 4.5.1 Slicing


```python
print(arr[0, :])
print(arr[:, 0])
print(arr[:, :2])
print(arr[:, 2:])
print(arr[:, 1:4])
```

    [0. 1. 2. 3. 4.]
    [0. 5.]
    [[0. 1.]
     [5. 6.]]
    [[2. 3. 4.]
     [7. 8. 9.]]
    [[1. 2. 3.]
     [6. 7. 8.]]
    


```python
print(ab[:, 0])  # ':' : take all rows and take the first(0) column
print(ab[:][0])  # ':' : take all the value and take the fist line
print(ab[0][0])
```

    [0 2]
    [0 1]
    0
    


```python
arr2 = arr[:, 1:4]
arr2[0, 0] =33
print(arr2)
print(arr)
```

    [[33.  2.  3.]
     [ 6.  7.  8.]]
    [[ 0. 33.  2.  3.  4.]
     [ 5.  6.  7.  8.  9.]]
    


```python
print(arr[0, ::-1])  # ::-1 means reverse order
```

    [ 4.  3.  2. 33.  0.]
    

### 4.5.2 Fancy indexing: integer or boolean array indexing


```python
arr2 = arr[arr > 5]  # return a copy
print('arr :', '\n', arr)
print('arr2 :', '\n', arr2)
```

    arr : 
     [[33.  1.  2.  3.  4.]
     [ 5.  6.  7.  8.  9.]]
    arr2 : 
     [33.  6.  7.  8.  9.]
    


```python
arr2[0] = 44
print('arr :', '\n', arr)
print('arr2 :', '\n', arr2)
```

    arr : 
     [[33.  1.  2.  3.  4.]
     [ 5.  6.  7.  8.  9.]]
    arr2 : 
     [44.  6.  7.  8.  9.]
    


```python
arr[arr > 5] = 0
print('arr :', '\n', arr)
```

    arr : 
     [[0. 1. 2. 3. 4.]
     [5. 0. 0. 0. 0.]]
    


```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
names == 'Bob'
names[names != 'Bob']
(names == 'Bob') | (names == 'Will')  
# keywords 'and/or' don't work with boolean
```




    array([ True, False,  True,  True])




```python
names[names != 'Bob'] = 'Joe'
print(names)
np.unique(names)  # count the unique values
```

    ['Bob' 'Joe' 'Joe' 'Bob']
    




    array(['Bob', 'Joe'], dtype='<U4')



### 4.6 Vectorized operations


```python
nums = np.arange(5)
nums1 = nums* 10
nums2 = np.sqrt(nums1)
print('nums: ', nums)
print('nums1: ', nums1)
print('nums2: ', nums2)
```

    nums:  [0 1 2 3 4]
    nums1:  [ 0 10 20 30 40]
    nums2:  [0.         3.16227766 4.47213595 5.47722558 6.32455532]
    


```python
np.ceil(nums2)   # also floor rint (round to nearest int)
```




    array([0., 4., 5., 6., 7.])




```python
np.isnan(nums)   # Check if there is null value
```




    array([False, False, False, False, False])




```python
print(nums + np.arange(5)) # 
print(np.maximum(nums, np.array([1, -2, 3, -4, 5]))) # compare elements
```

    [0 2 4 6 8]
    [1 1 3 3 5]
    


```python
# Compute Euclidean Distance between 2 vectors
vec1 = np.random.randn(10)
vec2 = np.random.randn(10)
dist = np.sqrt(np.sum((vec1 - vec2)**2))

print(vec1, '\n')
print(vec2, '\n')
print(dist)
```

    [ 0.27197301  0.45793172  0.4740813   2.06869452 -1.79934452  0.38349833
     -1.09794726  0.60664966 -1.6341181  -0.28426646] 
    
    [-0.80784398 -0.93663834 -1.67276128  1.79378845  0.56965117  0.06983035
     -0.71574274  0.69594858 -0.48597068 -1.78049267] 
    
    4.149319823417001
    


```python
# math and stats
rnd = np.random.randn(4, 2)
print(rnd, '\n')
print(rnd.mean(), '\n')
print(rnd.std(), '\n')
print(rnd.argmin(), '\n')  # index of minimum element

print(rnd.sum(axis = 0))  # axis = 0  sum of columns 
print(rnd.sum(axis = 1))  # axis = 1 sum of rows
```

    [[ 2.38225444 -0.83414318]
     [ 0.1034993  -0.79161477]
     [-0.80827946 -0.68009873]
     [-1.52975113  1.24582315]] 
    
    -0.11403879630132673 
    
    1.2202623778700652 
    
    6 
    
    [ 0.14772317 -1.06003354]
    [ 1.54811126 -0.68811547 -1.48837819 -0.28392797]
    


```python
# methods for boolean arrays
print((rnd > 0).sum()) # count number of positive values
print((rnd > 0).any()) # check if any value is true
print((rnd > 0).all()) # check if all value is true (all value > 0)
```

    3
    True
    False
    


```python
# random numbers
print(np.random.seed(12345)) # set the seed
print(np.random.rand(2, 3))  # 2 x 3 matrix in [0, 1]
print(np.random.randn(10))   # random normals (mean 0,  sd 1)
print(np.random.randint(0, 2, 10))  # 10 randomly picked 0 or 1
```

    None
    [[0.92961609 0.31637555 0.18391881]
     [0.20456028 0.56772503 0.5955447 ]]
    [ 0.09290788  0.28174615  0.76902257  1.24643474  1.00718936 -1.29622111
      0.27499163  0.22891288  1.35291684  0.88642934]
    [1 0 0 1 1 1 1 0 1 1]
    

### 4.7 Broadcasting


```python
a = np.array([[0, 0, 0],
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30]])
b = np.array([1, 2, 3])
a + b
```




    array([[ 1,  2,  3],
           [11, 12, 13],
           [21, 22, 23],
           [31, 32, 33]])



### 4.8 Exercises


```python
X = np.random.randn(4, 2)
print(X)
```

    [[ 1.05962632  0.64444817]
     [-0.00779918 -0.44920355]
     [ 2.44896272  0.66722619]
     [ 0.80292551  0.57572085]]
    


```python
X.argmin()
```




    3




```python
X.std()
```




    0.7935310360436236




```python
X.mean()
```




    0.7177383793504474


