from collections import namedtuple

GLVertexData = namedtuple('GLVertexData', 'indices vertices uvCoords normals')
GLVertexData.__doc__ = '''\
Data structure storing all vertex information of an object.

indices: A numpy array of size N holding indices to other data arrays where N
is the vertex count.

vertices: A numpy array of size 3*N holding the positions of each vertex. Each
x, y, z coordinate is written consecutively.

uvCoords: A numpy array of size 2*N holding the UV coordinates of each vertex.

normals: A numpy array of size 3*N holding the normal of each vertex. Each
normal is stored as x, y, z coordinates.
'''
