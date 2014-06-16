# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

"""Euclidean geometry support module"""

from __future__ import division

import math


class Vector:
    """3D-Vector class"""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """Initialize from three numbers"""
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def fromiterable(cls, itr):
        """Initialize from iterable"""
        x, y, z = itr
        return cls(x, y, z)

    @classmethod
    def fromvector(cls, v):
        """Copy another vector"""
        return cls(v.x, v.y, v.z)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        if index == 1:
            self.y = value
        if index == 2:
            self.z = value

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def copy(self):
        """return a copy of this vector"""
        return Vector(self.x, self.y, self.z)

    def __repr__(self):
        return ''.join((self.__class__.__name__, '(', str(self.x), ', ', str(self.y), ', ', str(self.z), ')'))

    def __eq__(self, other):
        return self.x == other[0] and self.y == other[1] and self.z == other[2]

    def close(self, other, epsilon=1e-10):
        return all([abs(v) <= epsilon for v in self-other])

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return Vector(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Vector(self.x - other, self.y - other, self.z - other)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return Vector(self.x / other, self.y / other, self.z / other)

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self.x, self.y, self.z = self.x + other.x, self.y + other.y, self.z + other.z
        else:
            self.x, self.y, self.z = self.x + other, self.y + other, self.z + other
        return self

    def __isub__(self, other):
        if isinstance(other, Vector):
            self.x, self.y, self.z = self.x - other.x, self.y - other.y, self.z - other.z
        else:
            self.x, self.y, self.z = self.x - other, self.y - other, self.z - other
        return self

    def __imul__(self, other):
        if isinstance(other, Vector):
            self.x, self.y, self.z = self.x * other.x, self.y * other.y, self.z * other.z
        else:
            self.x, self.y, self.z = self.x * other, self.y * other, self.z * other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Vector):
            self.x, self.y, self.z = self.x / other.x, self.y / other.y, self.z / other.z
        else:
            self.x, self.y, self.z = self.x / other, self.y / other, self.z / other
        return self

    def dot(self, other):
        """Dot product with another vector"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Cross product with another vector"""
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)

    def norm2(self):
        """Squared norm of the vector"""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self):
        """Length of the vector"""
        return math.sqrt(self.norm2())

    def normalize(self):
        """Normalize vector to length 1"""
        #noinspection PyMethodFirstArgAssignment
        self /= self.norm()
        return self

    def normalized(self):
        """Return normalized vector, but don't change original"""
        return self / self.norm()

    def rotate(self, l, u):
        """rotate l radians around axis u"""
        cl = math.cos(l)
        sl = math.sin(l)
        x = (cl + u.x * u.x * (1 - cl)) * self.x + (u.x * u.y * (1 - cl) - u.z * sl) * self.y + (
        u.x * u.z * (1 - cl) + u.y * sl) * self.z
        y = (u.y * u.x * (1 - cl) + u.z * sl) * self.x + (cl + u.y * u.y * (1 - cl)) * self.y + (
        u.y * u.z * (1 - cl) - u.x * sl) * self.z
        z = (u.z * u.x * (1 - cl) - u.y * sl) * self.x + (u.z * u.y * (1 - cl) + u.x * sl) * self.y + (
        cl + u.z * u.z * (1 - cl)) * self.z
        self.x, self.y, self.z = x, y, z
        return self

    def rotated(self, l, u):
        """rotate l radians around axis, but don't change original"""
        return self.copy().rotate(l, u)
        
