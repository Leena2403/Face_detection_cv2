import numpy as np 
a = 6
a = 8
# we didn't change the value of a, we just changed the pointer to a

it = [3,6,7,11]
n = np.array(it)
type(n)
np.ndarray

np.array({2,5,77})

d = {"apple": "A sweet red fruit", "mango": "king of fruit"}
np.array(d) # all datatypes are same... to fasten the processing

# lists are mutable, arrays are not

twod = np.array([[1,2,3,4]])
twod

y = np.array([2,5,6,"77"]) # Unicode item

# Dataframe can't go for 3D, but numpy can do

a = np.arange(15).reshape(3,5)
print(a)

