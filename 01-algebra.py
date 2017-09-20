#from http://www.deeplearning.net/software/theano/tutorial/adding.html
import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y],z)

print(f(2,3))

print(numpy.allclose(f(16.3, 12.1), 28.4))

print(x.type)

print(T.dscalar)

print(x.type is T.dscalar)

from theano import pp

print(pp(z))

print(numpy.allclose(z.eval({x : 16.3, y : 12.1}), 28.4))

a = T.vector("a");

out = a + a ** 10;
print(pp(out))
f = function([a], out)   # compile function
print(f([0, 1, 2]))

b = T.vector("b");

excercise = a ** 2 + b ** 2 + 2 * a * b

print(pp(excercise))
print(excercise.eval({a: [1, 2], b: [4, 5]}))

