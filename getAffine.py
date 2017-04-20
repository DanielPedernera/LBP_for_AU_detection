#! /usr/bin/env python3

import sys
import cv2
import argparse
import numpy as np
import math
import vtk

_EPS = np.finfo(float).eps * 4.0

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = np.random.random(3)
    >>> n = vector_norm(v)
    >>> np.allclose(n, np.linalg.norm(v))
    True
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> np.allclose(v1, np.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M

def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix([1, 2, 3])
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> np.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> np.allclose(R0, R1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3, ))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        #angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective

def rotation_matrix(a, b):
    w = np.cross(a, b)
    w /= np.linalg.norm(w)
    w_hat = np.mat([[0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]])
    cos_tht = (np.mat(a) * np.mat(b).transpose()
        / np.linalg.norm(a) / np.linalg.norm(b))
    tht = math.acos(cos_tht)
    return np.identity(len(a))+w_hat*math.sin(tht)+w_hat**2*(1-math.cos(tht))
##Return the base face
def load_frontal():
	return [[-62.978, -0.953, 13.127], [-29.558, 0.076, 20.207], [-0.492, 1.63, 21.43], [32.277, 2.935, 11.831], [-21.01, -5.035, 34.272], [-6.033, -5.028, 38.073], [-29.723, -31.066, 38.073], [-11.639, -29.176, 56.925], [5.639, -32.603, 39.442]]

def lolo():
	return [[-60.458, 39.707, -48.737], [-30.204, 38.085, -43.565], [0.289, 39.599, -44.332], [31.951, 40.673, -52.805], [-21.291, 29.33, -25.678], [-9.708, 30.038, -24.96], [-29.705, 2.705, -31.035], [-16.542, 7.472, -12.052], [-1.034, 2.956, -35.874], [-11.711, -67.612, -34.651]]
#    return [list(map(float, l.strip("\r\n").split(",")))
#        for l in open(FRONTAL_FILE, "r")]
def euc_dist(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)

def normalize(lms):
    unit = euc_dist(lms[12], lms[10])
    norm = [(np.asarray(l)/unit).tolist() for l in lms]
    offset = [-a for a in norm[10]]
    norm = [(np.asarray(l)+offset).tolist() for l in norm]

    return norm

def print_model(model, out_file, size=(1024, 1024), unit=128):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    origin = (size[1]/2 - unit/2, size[0]/2)
    for (i, p) in enumerate(model):
        lm = (p[0]*unit+origin[0], p[1]*unit+origin[1])
        cv2.circle(img, tuple(map(int, lm[:2])), 2, (0, 0, 255))
        cv2.putText(img, "%d" % (i+1), (int(lm[0]-5), int(lm[1]-5)),
            cv2.FONT_ITALIC, .5, (0, 255, 0))
    cv2.imwrite(out_file, img)

def average_data(data, fun):
    avg = []
    for i in range(len(data[1])):
        avg.append([fun(r[0])
            for r in np.asarray([d[i] for d in data]).transpose().tolist()])
    return avg


def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def get_affine(subject):
	'''Returns the affine matrix
	   given the landmark 3D name
	   of the BOSPHORUS database'''
	frontal = load_frontal()
	'''lms = []
	with open(subject) as openfilesubject:
		for lm3 in openfilesubject:
			landmarks = lm3.strip("\r\n").split(" ")
			nLand = len(landmarks)
			if  nLand == 3 :
				landmark = [num(it) for it in landmarks]
				if ( not (landmark[0] ==  None) and len(lms) < 22):
					lms.append(landmark)'''
	file = open(subject, 'r')
	lines = file.readlines()
	lines = [line.strip("\r\n") for line in lines]
	dlms = {}
	for i in range(3, len(lines), 2):
		dlms[lines[i]] = lines[i+1]
	lms = [ dlms['Outer left eye corner'], dlms['Inner left eye corner'], dlms['Inner right eye corner'], dlms['Outer right eye corner'], dlms['Nose saddle left'], dlms['Nose saddle right'], dlms['Left nose peak'], dlms['Nose tip'], dlms['Right nose peak']]
	lms = [[num(axe) for axe in landmark.split(" ")] for landmark in lms]
	return affine_matrix_from_points(np.mat(frontal).transpose(), np.mat(lms).transpose())

'''def get_affine(subject):
		Returns the affine matrix
	   given the landmark 3D name
	   of the BOSPHORUS database
	frontal = load_frontal()
	lms = []
	with open(subject) as openfilesubject:
		for lm3 in openfilesubject:
			landmarks = lm3.strip("\r\n").split(" ")
			nLand = len(landmarks)
			if  nLand == 3 :
				landmark = [num(it) for it in landmarks]
				if ( not (landmark[0] ==  None) and len(lms) < 22):
					lms.append(landmark)
	return affine_matrix_from_points(np.mat(frontal).transpose(), np.mat(lms).transpose())'''

def plyToMatrix(name):
	r = vtk.vtkPLYReader()
	r.SetFileName(name)
	r.Update()

	mesh = r.GetOutput()
	pointArray = mesh.GetPoints()
	points = pointArray.GetData()
	npoints = points.GetNumberOfTuples()
	new_points = []
	
	for i in xrange(0, (3 * npoints), 3):
		new_points.append([points.GetValue(i), points.GetValue(i+1), points.GetValue(i+2), 1])

	nppoints = np.array(new_points)
	
	return nppoints

def objToMatrix(name):
	r = vtk.vtkOBJReader()
	r.SetFileName(name)
	r.Update()
	mesh = r.GetOutput()
	pointArray = mesh.GetPoints()
	points = pointArray.GetData()
	npoints = points.GetNumberOfTuples()
	new_points = []
	for i in xrange(0, (npoints), 3):
		new_points.append([points.GetValue(i), points.GetValue(i+1), points.GetValue(i+2), 1])
	nppoints = np.array(new_points)
	return nppoints

def createPointCloud(newpc, name):
	#print name
	f = open(name, 'wr')
	nroelems = newpc.shape[0]
	f.writelines('OFF\n')
	f.writelines(str(nroelems) + ' 0 0\n')
	for point in newpc:
		lines = ''
		for i in range(0, 3):
			lines += str(point[i]) + ' '
		f.writelines(lines+'\n')
	print name + '... saved'

###############################MAIN#############################################



if __name__=="__main__":
	filename = str(sys.argv[1])
	basepath = str(sys.argv[2])
	with open(filename) as f:
		for subject_au in f:
			subject = subject_au.split(' ')[0]
			#subject = str(sys.argv[2])
		#	subland = basepath + subject + '.lm3'
		#	localpath = basepath + 'preprocessing/elements/' + subject
			localpath = basepath + subject
			ps = localpath +'_d2d.ply'
			ns = localpath +'_d2d.off'
			#pr = localpath +'_pr.off'
			##print subland, ps, ns
		#	affine = get_affine(subland)
			#print affine.shape
			nppoints = plyToMatrix(ps)
			#print nppoints.shape
		#	newpc = np.dot(nppoints, affine)
			#print newpc.shape
		#	createPointCloud(newpc, ns)
			createPointCloud(nppoints, ns)
	#result = str(affine).replace("[","").replace("]","").replace(" ","").replace("\n,", "f").replace("\nnn","lolo")
	#print result
	#print str(affine).replace("[","").replace("]","").replace("  ",",")
	#print(str(affine.tolist()).replace("], ", "f").replace("]","").replace("[", "").replace(" ",""));
	sys.exit(0)



