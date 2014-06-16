# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

"""Module to generate 3d EEG locations"""

from numpy import *

from .geometry import spherical as geo
from .tools import Struct

#import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
#from mpl_toolkits.mplot3d import Axes3D

Point = geo.Point
Line = geo.Line
Circle = geo.Circle
construct = geo.Construct


def intersection(a, b, expr=lambda c: c.vector.z >= 0):
    pts = construct.circle_intersect_circle(a, b)
    return [c for c in pts if expr(c)]


midpoint = construct.midpoint


#noinspection PyPep8Naming
def construct_1020_easycap(variant=0):
    p = Struct()

    p.Cz = Point(0, 0, 1)
    p.Fpz = Point(0, 1, 0)
    p.Oz = Point(0, -1, 0)
    p.T7 = Point(-1, 0, 0)
    p.T8 = Point(1, 0, 0)

    #horizontal = Circle(p.Cz, p.Fpz)    # grand-circle in horizontal plane
    #sagittal = Circle(p.T7, p.Cz)       # grand-circle in sagittal plane
    #coronal = Circle(p.Oz, p.Cz)        # grand-circle in coronal plane

    horizontal = Line(p.Fpz, p.T7)    # grand-circle in horizontal plane
    sagittal = Line(p.Fpz, p.Cz)       # grand-circle in sagittal plane
    coronal = Line(p.T7, p.Cz)        # grand-circle in coronal plane

    p.Fz = sagittal.get_point(0.5)
    p.Pz = sagittal.get_point(1.5)
    p.C3 = coronal.get_point(0.5)
    p.C4 = coronal.get_point(1.5)
    p.Fp1 = horizontal.get_point(0.2)
    p.Fp2 = horizontal.get_point(-0.2)
    p.F7 = horizontal.get_point(0.6)
    p.F8 = horizontal.get_point(-0.6)
    p.P7 = horizontal.get_point(1.4)
    p.P8 = horizontal.get_point(-1.4)
    p.O1 = horizontal.get_point(1.8)
    p.O2 = horizontal.get_point(-1.8)

    circle_F = Circle(p.F7, p.Fz, p.F8)
    circle_P = Circle(p.P7, p.Pz, p.P8)

    if variant == 0:
        circle_3 = Circle(p.Fp1, p.C3, p.O1)
        circle_4 = Circle(p.Fp2, p.C4, p.O2)
    #elif variant == 1:
    else:
        circle_3 = Circle(p.Fpz, p.C3, p.Oz)
        circle_4 = Circle(p.Fpz, p.C4, p.Oz)

    p.F3 = intersection(circle_3, circle_F)[0]
    p.F4 = intersection(circle_4, circle_F)[0]
    p.P3 = intersection(circle_3, circle_P)[0]
    p.P4 = intersection(circle_4, circle_P)[0]

    p.AFz = midpoint(p.Fpz, p.Fz)
    p.AF7 = midpoint(p.Fp1, p.F7)
    p.AF8 = midpoint(p.Fp2, p.F8)

    circle_AF = Circle(p.AF7, p.AFz, p.AF8)
    angle_AF = circle_AF.angle(p.AF7) / 2
    p.AF3 = circle_AF.get_point(angle_AF)
    p.AF4 = circle_AF.get_point(-angle_AF)

    angle_F2 = circle_F.angle(p.F4) / 2
    angle_F6 = circle_F.angle(p.F4) + (circle_F.angle(p.F8) - circle_F.angle(p.F4)) / 2
    p.F2 = circle_F.get_point(angle_F2)
    p.F1 = circle_F.get_point(-angle_F2)
    p.F6 = circle_F.get_point(angle_F6)
    p.F5 = circle_F.get_point(-angle_F6)

    p.C1 = midpoint(p.C3, p.Cz)
    p.C2 = midpoint(p.C4, p.Cz)
    p.C5 = midpoint(p.C3, p.T7)
    p.C6 = midpoint(p.C4, p.T8)

    angle_P2 = circle_P.angle(p.P4) / 2
    angle_P6 = circle_P.angle(p.P4) + (circle_P.angle(p.P8) - circle_P.angle(p.P4)) / 2
    p.P2 = circle_P.get_point(angle_P2)
    p.P1 = circle_P.get_point(-angle_P2)
    p.P6 = circle_P.get_point(angle_P6)
    p.P5 = circle_P.get_point(-angle_P6)

    circle_5 = Circle(p.F5, p.C5, p.P5)
    circle_1 = Circle(p.F1, p.C1, p.P1)
    circle_2 = Circle(p.F2, p.C2, p.P2)
    circle_6 = Circle(p.F6, p.C6, p.P6)

    p.FCz = midpoint(p.Fz, p.Cz)
    p.FT7 = midpoint(p.F7, p.T7)
    p.FT8 = midpoint(p.F8, p.T8)

    p.CPz = midpoint(p.Cz, p.Pz)
    p.TP7 = midpoint(p.T7, p.P7)
    p.TP8 = midpoint(p.T8, p.P8)

    circle_FC = Circle(p.FT7, p.FCz, p.FT8)
    circle_CP = Circle(p.TP7, p.CPz, p.TP8)

    p.FC5 = intersection(circle_5, circle_FC)[0]
    p.FC3 = intersection(circle_3, circle_FC)[0]
    p.FC1 = intersection(circle_1, circle_FC)[0]
    p.FC2 = intersection(circle_2, circle_FC)[0]
    p.FC4 = intersection(circle_4, circle_FC)[0]
    p.FC6 = intersection(circle_6, circle_FC)[0]

    p.CP5 = intersection(circle_5, circle_CP)[0]
    p.CP3 = intersection(circle_3, circle_CP)[0]
    p.CP1 = intersection(circle_1, circle_CP)[0]
    p.CP2 = intersection(circle_2, circle_CP)[0]
    p.CP4 = intersection(circle_4, circle_CP)[0]
    p.CP6 = intersection(circle_6, circle_CP)[0]

    p.POz = midpoint(p.Pz, p.Oz)
    p.PO7 = midpoint(p.P7, p.O1)
    p.PO8 = midpoint(p.P8, p.O2)

    circle_PO = Circle(p.PO7, p.POz, p.PO8)
    angle_PO = circle_PO.angle(p.PO7) / 2
    p.PO3 = circle_PO.get_point(-angle_PO)
    p.PO4 = circle_PO.get_point(angle_PO)

    # below the equator

    p.Iz = sagittal.get_point(2.25)
    p.T9 = coronal.get_point(-0.25)
    p.T10 = coronal.get_point(2.25)

    circle_9 = Circle(p.T9, p.Iz, p.T10)

    p.O9 = circle_9.get_point(-pi / 2 * 0.2)
    p.O10 = circle_9.get_point(pi / 2 * 0.2)

    p.PO9 = circle_9.get_point(-pi / 2 * 0.4)
    p.PO10 = circle_9.get_point(pi / 2 * 0.4)

    p.P9 = circle_9.get_point(-pi / 2 * 0.6)
    p.P10 = circle_9.get_point(pi / 2 * 0.6)

    p.TP9 = circle_9.get_point(-pi / 2 * 0.8)
    p.TP10 = circle_9.get_point(pi / 2 * 0.8)

    p.FT9 = circle_9.get_point(-pi / 2 * 1.2)
    p.FT10 = circle_9.get_point(pi / 2 * 1.2)

    p.F9 = circle_9.get_point(-pi / 2 * 1.4)
    p.F10 = circle_9.get_point(pi / 2 * 1.4)

    return p


positions = construct_1020_easycap()
