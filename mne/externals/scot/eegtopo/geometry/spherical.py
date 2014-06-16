# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

"""Spherical geometry support module"""

from __future__ import division

import math

from .euclidean import Vector


################################################################################

eps = 1e-15

################################################################################

class Point:
    """Point on the surface of a sphere"""

    def __init__(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            self._pos3d = Vector(0, 0, 1)
        elif x is not None and y is not None and z is None:
            self._pos3d = Vector(x, y, math.sqrt(1 - x ** 2 - y ** 2))
        elif x is not None and y is not None and z is not None:
            self._pos3d = Vector(x, y, z).normalized()
        else:
            raise RuntimeError('invalid parameters')

    @classmethod
    def fromvector(cls, v):
        """Initialize from euclidean vector"""
        w = v.normalized()
        return cls(w.x, w.y, w.z)

    @property
    def vector(self):
        """position in 3d space"""
        return self._pos3d

    @vector.setter
    def vector(self, v):
        self._pos3d.x = v.x
        self._pos3d.y = v.y
        self._pos3d.z = v.z

    def __repr__(self):
        return ''.join(
            (self.__class__.__name__, '(', str(self._pos3d.x), ', ', str(self._pos3d.y), ', ', str(self._pos3d.z), ')'))

    def distance(self, other):
        """Distance to another point on the sphere"""
        return math.acos(self._pos3d.dot(other.vector))

    def distances(self, points):
        """Distance to other points on the sphere"""
        return [math.acos(self._pos3d.dot(p.vector)) for p in points]

################################################################################

class Line:
    """Line on the spherical surface (also known as grand circle)"""

    def __init__(self, a, b):
        self.a = Point.fromvector(a.vector)
        self.b = Point.fromvector(b.vector)

    def get_point(self, l):
        d = self.a.distance(self.b)
        n = self.a.vector.cross(self.b.vector)
        p = Point.fromvector(self.a.vector)
        p.vector.rotate(l * d, n)
        return p

    def distance(self, p):
        n = Point.fromvector(self.a.vector.cross(self.b.vector))
        return abs(math.pi / 2 - n.distance(p))

################################################################################

class Circle:
    """Arbitrary circle on the spherical surface"""

    def __init__(self, a, b, c=None):
        if c is None:
            self.c = Point.fromvector(a.vector)     # Center
            self.x = Point.fromvector(b.vector)     # a point on the circle
        else:
            self.c = Point.fromvector((b.vector - a.vector).cross(c.vector - b.vector).normalized())    # Center
            self.x = Point.fromvector(b.vector)     # a point on the circle

    def get_point(self, l):
        return Point.fromvector(self.x.vector.rotated(l, self.c.vector))

    def get_radius(self):
        return self.c.distance(self.x)

    def angle(self, p):

        c = self.c.vector * self.x.vector.dot(self.c.vector) # center in circle plane

        a = (self.x.vector - c).normalized()
        b = (p.vector - c).normalized()
        return math.acos(a.dot(b))

    def distance(self, p):
        return abs(self.c.distance(p) - self.c.distance(self.x))


################################################################################

class Construct:
    """Collection of methods for geometric construction on a sphere"""

    @staticmethod
    def midpoint(a, b):
        """Point exactly between a and b"""
        return Point.fromvector((a.vector + b.vector) / 2)

    @staticmethod
    def line_intersect_line(k, l):
        c1 = k.a.vector.cross(k.b.vector)
        c2 = l.a.vector.cross(l.b.vector)
        p = c1.cross(c2)
        return Point.fromvector(p), Point.fromvector(p * -1)

    @staticmethod
    def line_intersect_circle(line, circle):
        cross_line = line.a.vector.cross(line.b.vector)
        cross_lc = cross_line.cross(circle.c.vector)
        dot_circle = circle.c.vector.dot(circle.x.vector)
        if abs(cross_lc.z) > eps:
            a = cross_lc.dot(cross_lc)
            b = 2 * dot_circle * cross_line.cross(cross_lc).z
            circle = dot_circle * dot_circle * (cross_line.x ** 2 + cross_line.y ** 2) - cross_lc.z ** 2
            s = math.sqrt(b ** 2 - 4 * a * circle)
            z1 = (s - b) / (2 * a)
            x1 = (cross_lc.x * z1 - cross_line.y * dot_circle) / cross_lc.z
            y1 = (cross_lc.y * z1 + cross_line.x * dot_circle) / cross_lc.z
            z2 = -(s + b) / (2 * a)
            x2 = (cross_lc.x * z2 - cross_line.y * dot_circle) / cross_lc.z
            y2 = (cross_lc.y * z2 + cross_line.x * dot_circle) / cross_lc.z
            return Point(x1, y1, z1), Point(x2, y2, z2)
        else:
            return None

    @staticmethod
    def circle_intersect_circle(a, b):
        ac = a.c.vector
        bc = b.c.vector
        cross = ac.cross(bc)
        dot_a = ac.dot(a.x.vector)
        dot_b = bc.dot(b.x.vector)
        if abs(cross.z) > eps:
            a = cross.dot(cross)
            b = 2 * (dot_b * ac.cross(cross).z - dot_a * bc.cross(cross).z)
            c = dot_b ** 2 * (ac.x ** 2 + ac.y ** 2) - 2 * dot_a * dot_b * (ac.x * bc.x + ac.y * bc.y) + dot_a ** 2 * (
            bc.x ** 2 + bc.y ** 2) - cross.z ** 2
            s = math.sqrt(b ** 2 - 4 * a * c)
            z1 = (s - b) / (2 * a)
            x1 = (bc.y * dot_a - ac.y * dot_b + cross.x * z1) / cross.z
            y1 = (ac.x * dot_b - bc.x * dot_a + cross.y * z1) / cross.z
            z2 = -(s + b) / (2 * a)
            x2 = (bc.y * dot_a - ac.y * dot_b + cross.x * z2) / cross.z
            y2 = (ac.x * dot_b - bc.x * dot_a + cross.y * z2) / cross.z
            return Point(x1, y1, z1), Point(x2, y2, z2)
        else:
            return None

################################################################################
