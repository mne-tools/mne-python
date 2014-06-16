# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 Martin Billinger

from numpy import arcsin, sin, cos, pi, sqrt, sum
from numpy import atleast_2d, asarray, zeros, newaxis


def project_radial_to2d(point_3d):
    point_2d = point_3d.copy()
    point_2d.z = 0
    beta = point_2d.norm()
    if beta == 0:
        alpha = 0
    else:
        alpha = arcsin(beta) / beta

    if point_3d.z < 0:
        alpha = pi / beta - alpha

    point_2d *= alpha

    return point_2d


def project_radial_to3d(point_2d):
    alpha = point_2d.norm()
    if alpha == 0:
        beta = 1
    else:
        beta = sin(alpha) / alpha
    point_3d = point_2d * beta
    point_3d.z = cos(alpha)
    return point_3d


def array_project_radial_to2d(points_3d):
    points_3d = atleast_2d(points_3d)
    points_2d = points_3d[:, 0:2]

    betas = sqrt(sum(points_2d**2, -1))

    alphas = zeros(betas.shape)

    mask = betas != 0
    alphas[mask] = arcsin(betas[mask]) / betas[mask]

    mask = points_3d[:, 2] < 0
    alphas[mask] = pi / betas[mask] - alphas[mask]

    return points_2d * alphas[:, newaxis]


def array_project_radial_to3d(points_2d):
    points_2d = atleast_2d(points_2d)

    alphas = sqrt(sum(points_2d**2, -1))

    betas = sin(alphas) / alphas
    betas[alphas == 0] = 1

    x = points_2d[..., 0] * betas
    y = points_2d[..., 1] * betas
    z = cos(alphas)

    points_3d = asarray([x, y, z]).T

    return points_3d
