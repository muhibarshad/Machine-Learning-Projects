import numpy as np

a = np.array([[1,2,3,4], [4,5,6,6], [2,3,4,5]])
b = np.zeros(2) # [0., 0. ]
c = np.ones(4) # [1.,1.,1.,1.] 
d = np.empty(3) # [0 1 2 3 4 5 6 7]
e = np.arange(8) # [0 1 2 3 4 5 6 7]
f = np.linspace(2, 10, num=5) # [ 2.  4.  6.  8. 10.]
g = np.ones(2, dtype=np.int64) # [1 1]

arr = np.array([3,2,15,6])
sortArr = np.sort(arr)

add = np.append(arr, 90)
insert = np.insert(arr, 2, 30)
delete = np.delete(insert, 2)
multiDel = np.delete(insert, [0,2])
concainate = np.concatenate((arr, multiDel), axis=0)

ndArray = np.array([[[1,2,3,4],
                    [5,6,7,8]],
                    
                    [[1,2,3,4],
                    [5,6,7,8]],
                    
                    [[1,2,3,4],
                    [5,6,7,8]]])

h=ndArray.ndim# 3
i=ndArray.size# 24
j=ndArray.shape # 3, 2, 4

k = np.array([1,2,3,4, 5, 6])
l = k.reshape(2, 3) #[[1 2 3]
                    # [4 5 6]]
m = np.reshape(k, newshape=(1, 6), order='C')
n = k[np.newaxis, : ] # adding row_vector
# print(k) # [1 2 3 4 5 6]
# print(n) # [[1 2 3 4 5 6]] 
# print(n.shape) # (1, 6)
o = k[: , np.newaxis] # adding column_vector
# print(k) #  [1 2 3 4 5 6]
# print(o) 
# '''
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]
#  [6]]
# '''
# print(o.shape) # (6, 1)

p = np.expand_dims(k, axis=0)
q = np.expand_dims(k, axis=1)
# print(p.shape) # (1, 6)
# print(q.shape) # (6 , 1)

r = np.array([1,2,3,4,5,6,7])
# print (r[2:-2]) # [3,4,5]

# To get the coordinates of points where the number is less than 5 
s = np.array([[1 , 8, 9, 3], [5, 1, 7, 2], [9, 2, 11, 4]])
t = np.nonzero(s<5) # (array([0, 0, 1, 1, 2, 2]), array([0, 3, 1, 3, 1, 3])) (row, column)-->indexex
coordinates = list(zip(t[0], t[1]))
for coord in coordinates : 
    print(coord)
# print(t[1]) # Print the numbers that are less than 5
'''
(0, 0)
(0, 3)
(1, 1)
(1, 3)
(2, 1)
(2, 3)
'''

a1 = np.array([[1,2],[3,4]])
a2 = np.array([[5,6],[7,8]])

a3= np.vstack((a1,a2)) 
'''
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
 '''
a4= np.hstack((a1,a2))
'''
[[1 2 5 6]
 [3 4 7 8]]
 for the hspilt 
 np.hsplit(x, 3)
  [array([[ 1,  2,  3,  4],
         [13, 14, 15, 16]]), array([[ 5,  6,  7,  8],
         [17, 18, 19, 20]]), array([[ 9, 10, 11, 12],
         [21, 22, 23, 24]])]
 '''

# for copying 
#b2 = a.copy()

u =np.arange(4)
v = np.ones(4, dtype=int)
# print(u.sum()) # 6
# print(u+v) #[1,2,3,4]
# print(u-v) #[-1  0  1  2]
# print(u*v) #[0 1 2 3]
# print(u/v) #[0. 1. 2. 3.]

w = np.array([[1,2],
              [3,4]])
x = np.array([[5,6],[7,8]])

# print(w.sum(axis=0)) # [4,6]
# print(w.sum(axis=1)) # [3,7]
# print(w+x)
'''
[[ 6  8]
 [10 12]]
 '''
y = np.array([1,2,3])
# print(y*2.6) #  [2.6 5.2 7.8] broadcasting


# axis = 0 means operations are performed in the row wise 
''' 
[1,2,3]
[4,5,6]
=[5,7,9]

'''

# axis = 1 means operations are performed in the column wise 
'''
[1,2,3]
[4,5,6]
=[6, 15]
'''

z=np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
              [0.54627315, 0.05093587, 0.40067661, 0.55645993],
              [0.12697628, 0.82485143, 0.26590556, 0.56917101]])
# print(z.max()) # 0.82485143
# print(z.min()) # 0.05093587
# print(z.sum()) # 4.8595784
# print(z.max(axis=0)) # [0.54627315 0.82485143 0.40067661 0.56917101]
# print(z.min(axis=1)) # [0.17296777 0.05093587 0.12697628]
# print(z.sum(axis=0)) # [1.12378257 1.04875507 1.01034462 1.67669614]

aa = np.array([[1,2,3],[4,5,6]])
ab = np.array([[1,2,3],[4,5,6]])
ac = np.ones(3, dtype=int)
# print(aa+ab)
# print(aa+ac)
ad = np.ones((4, 3,2),dtype=int)
# print(ad)

rng = np.random.default_rng()
ae = rng.random((3,3))
# print(ae)

'''
you can generate random integers from low (remember that this is inclusive with NumPy)
to high (exclusive).
You can set endpoint=True to make the high number inclusive.
You can generate a 2 x 4 array of random integers between 0 and 5 with:
'''

af = rng.integers(5,size=(2,4), endpoint=True)
# print(af)

ag = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 8, 4]])
ah = np.unique(ag)
ai = np.unique(ag, axis=0)
aj = np.unique(ag, axis=1)
values, indexes, counts = np.unique(ag, axis=0, return_index=True, return_counts=True)
# print(values)
# print(indexes)
# print(counts)

ak = np.arange(10)
al = ak.reshape((2,5))
am = al.transpose()
an = am.T
# print(ak)
# print(al)
# print(am)
# print(an)


ao = np.flip(ak)
# print(ao)

ap=  np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])



