"""Dependency-free four-point homography: pixels (u,v) to metres (x,y).

A homography maps one plane, not RGB depth. Simulation uses known object
heights to select a parallel top plane explicitly; no depth is inferred.
"""
from itertools import combinations
import math


def _points(points, name):
    try:
        points = tuple(tuple(float(v) for v in point) for point in points)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain four finite 2D points.") from error
    if len(points) != 4 or any(len(point) != 2 or not all(map(math.isfinite, point)) for point in points):
        raise ValueError(f"{name} must contain four finite 2D points.")
    span = max(max(p[d] for p in points)-min(p[d] for p in points) for d in (0,1))
    if span <= 0:
        raise ValueError(f"{name} is degenerate.")
    for a,b,c in combinations(points, 3):
        area = ((b[0]-a[0])/span)*((c[1]-a[1])/span)-((b[1]-a[1])/span)*((c[0]-a[0])/span)
        if abs(area) < 1e-8:
            raise ValueError(f"{name} has duplicate or collinear points.")
    return points


def _normalise(points):
    centre = tuple(sum(p[d] for p in points)/len(points) for d in (0,1))
    scale = math.sqrt(2)/(sum(math.dist(p, centre) for p in points)/len(points))
    matrix = ((scale,0.,-scale*centre[0]),(0.,scale,-scale*centre[1]),(0.,0.,1.))
    inverse = ((1/scale,0.,centre[0]),(0.,1/scale,centre[1]),(0.,0.,1.))
    return tuple(((x-centre[0])*scale,(y-centre[1])*scale) for x,y in points),matrix,inverse


def _multiply(a,b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _solve(rows, rhs):
    augmented = [list(row)+[value] for row,value in zip(rows,rhs)]
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col,n),key=lambda row:abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Degenerate calibration system.")
        augmented[col],augmented[pivot] = augmented[pivot],augmented[col]
        factor=augmented[col][col]
        augmented[col]=[v/factor for v in augmented[col]]
        for row in range(n):
            if row != col:
                factor=augmented[row][col]
                augmented[row]=[a-factor*b for a,b in zip(augmented[row],augmented[col])]
    return tuple(row[-1] for row in augmented)


def compute_homography(image_points, world_points) -> tuple[tuple[float, ...], ...]:
    """Four corresponding points in matching order; returns a 3x3 tuple matrix.

    Coordinates are normalised before solving the eight-parameter linear
    system. Duplicate/collinear/ill-conditioned correspondences are rejected.
    """
    image, world = _points(image_points,"image_points"), _points(world_points,"world_points")
    src,src_matrix,_ = _normalise(image)
    dst,_,dst_inverse = _normalise(world)
    rows,rhs=[],[]
    for (u,v),(x,y) in zip(src,dst):
        rows += [[u,v,1,0,0,0,-x*u,-x*v], [0,0,0,u,v,1,-y*u,-y*v]]
        rhs += [x,y]
    values=_solve(rows,rhs)+(1.,)
    normalised=tuple(tuple(values[3*i:3*i+3]) for i in range(3))
    matrix=_multiply(_multiply(dst_inverse,normalised),src_matrix)
    for uv,xy in zip(image,world):
        if math.dist(pixel_to_world(matrix,*uv),xy)>1e-6:
            raise ValueError("Calibration reprojection residual is too large.")
    return matrix


def pixel_to_world(homography, u: float, v: float) -> tuple[float,float]:
    try:
        h=tuple(tuple(float(x) for x in row) for row in homography)
    except (ValueError,TypeError) as error:
        raise ValueError("Expected a finite 3x3 homography.") from error
    if len(h)!=3 or any(len(row)!=3 or not all(map(math.isfinite,row)) for row in h):
        raise ValueError("Expected a finite 3x3 homography.")
    scale=max(abs(x) for row in h for x in row)
    if scale==0 or not all(map(math.isfinite,(u,v))):
        raise ValueError("Invalid matrix or pixel coordinates.")
    h=tuple(tuple(x/scale for x in row) for row in h)
    determinant=(h[0][0]*(h[1][1]*h[2][2]-h[1][2]*h[2][1])
                 -h[0][1]*(h[1][0]*h[2][2]-h[1][2]*h[2][0])
                 +h[0][2]*(h[1][0]*h[2][1]-h[1][1]*h[2][0]))
    if abs(determinant)<1e-15:
        raise ValueError("Singular or ill-conditioned homography.")
    point=tuple(row[0]*u+row[1]*v+row[2] for row in h)
    if abs(point[2])<1e-12:
        raise ValueError("Pixel maps to infinity.")
    return point[0]/point[2],point[1]/point[2]
