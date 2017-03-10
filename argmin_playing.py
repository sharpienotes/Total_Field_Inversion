# this file is used to play with the argmin function and get a grasp on it

import numpy

#numbers = numpy.random.random([3,3,3,5,4])
#values = numpy.arange(15).reshape(3, 5)
#values = numpy.random.random([3,3,3,5,4])
values = numpy.array([10, 9, 4, 9, 4, 1, 2, 3, 4, 5, 3, 3]).reshape(3,4)
print(values)
minmin = numpy.ndarray.argmin(values)
print('Good Afternoon, the minimum is at: '+str(minmin))
squirrel = numpy.unravel_index(minmin, values.shape)
print(squirrel)

#output is:
#[[10  9  4  9]
# [ 4  1  2  3]
# [ 4  5  3  3]]
#Good Afternoon, the minimum is at: 5
#(1, 1)

# unravel translates the number of the coordinate to an actual shape in space for this array
