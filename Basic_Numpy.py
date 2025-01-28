import numpy as np
lst=[1,2,3,4,2,3,2,5]
# convert lst to array
arr=np.array(lst)
print(type(arr))
# tells array dimensions
print(arr.shape)

# print by indexing
print(arr[2])
# print from index 2
print(arr[2:])
#  print from index 2
print(arr[2:6])
# gives last element
print(arr[-1])
# skip last element and gives all remaining element
print(arr[:-1])
# reverse the arr
print(arr[::-1])
# print arr by jumping 2 element
print(arr[::-2])



lst1=[1,2,3,4,5]
lst2=[6,7,8,9,10]
lst3=[11,12,13,14,15]
# convert multiple lst to array
arr2=np.array([lst1,lst2,lst3])
# tells array dimensions
print(arr2.shape)
# gives all rows and 1st column
print(arr2[:,1])
# gives element in specific rows and column
print(arr2[1:,1:3])
print(arr2[1:,3:])
print(arr2[0:,3:])
print(arr2[0:,[0,-1]])
print(arr2[0:,3:].shape)


# EDA
# shows true and false according to condition
print(arr<2)
# shows exact that value which satiesfy the condition
print(arr[arr<2])
# reshape the arr
print(arr2.reshape(5,3))
# creates an arr using start stop and step value
print(np.arange(1,10,1))
# we can also reshape that array
print(np.arange(1,10,1).reshape(3,3))
# creates an arr using start stop and step value
print(np.arange(1,30,2))
# we can also reshape that array
print(np.arange(1,30,2).reshape(5,3))
# multiplies the arr
print(arr*arr)
# multiply by number
print(arr*2)
# multiply multi D arr
print(arr2*arr2)
# return arr filled with 1s
print(np.ones((5,3)))
# return arr filled with 0s
print(np.zeros((5,3)))
# returns random number
print(np.random.randint(10,100))
print(np.random.randint(50,100,5))
print(np.random.randint(50,100,6).reshape(2,3))