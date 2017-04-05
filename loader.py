import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image


def loadTextureBMP(filepath):
    """
    Loads the BMP file given in filepath, creates an OpenGL texture from it
    and returns the texture ID.
    """
    data = np.array(Image.open(filepath))
    width = data.shape[0]
    height = data.shape[1]
    textureID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureID)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_BGR,
        GL_UNSIGNED_BYTE,
        data,
    )
    # default parameters for now. Can be parameterized in the future
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    return textureID


def loadShaders(vertex_filepath, fragment_filepath):
    """
    Compiles and links a shader program from the given vertex and fragment
    shader source files.

    Returns the linked shader program ID.
    """
    vertexShaderID = glCreateShader(GL_VERTEX_SHADER)
    fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER)
    with open(vertex_filepath, 'r') as f:
        vertexShaderCode = f.read()
    with open(fragment_filepath, 'r') as f:
        fragmentShaderCode = f.read()
    glShaderSource(vertexShaderID, vertexShaderCode)
    glCompileShader(vertexShaderID)
    msg = glGetShaderInfoLog(vertexShaderID)
    if msg:
        print("VERTEX SHADER COMPILE ERROR: " + msg.decode('utf-8'))
    glShaderSource(fragmentShaderID, fragmentShaderCode)
    glCompileShader(fragmentShaderID)
    msg = glGetShaderInfoLog(fragmentShaderID)
    if msg:
        print("FRAGMENT SHADER COMPILE ERROR: " + msg.decode('utf-8'))
    programID = glCreateProgram()
    glAttachShader(programID, vertexShaderID)
    glAttachShader(programID, fragmentShaderID)
    glLinkProgram(programID)
    msg = glGetProgramInfoLog(programID)
    if msg:
        print("PROGRAM LINK ERROR: " + msg.decode('utf-8'))
    glDetachShader(programID, vertexShaderID)
    glDetachShader(programID, fragmentShaderID)
    glDeleteShader(vertexShaderID)
    glDeleteShader(fragmentShaderID)
    return programID


def loadOBJ(filepath):
    """
    Loads a Wavefront Obj file given in the filepath, extracts and returns
    vertex, texture and normal data in it as three float32 numpy arrays.

    If the file doesn't contain texture or normal information, the corresponding
    return values are None.
    """
    index_lines = []
    data_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('f '):
                index_lines.append(line.strip())
            else:
                data_lines.append(line.strip())
    # buffers
    v  = []  # vertices
    vn = []  # normals
    uv = []  # uv coords
    for line in data_lines:
        words = line.split()
        if words[0] == 'v':     # vertex line
            v = v + [float(num) for num in words[1:]]
        elif words[0] == 'vt':  # uv coord. line
            uv = uv + [float(num) for num in words[1:]]
        elif words[0] == 'vn':  # normal line
            vn = vn + [float(num) for num in words[1:]]
    # indices
    v_index  = []
    vn_index = []
    uv_index = []
    for line in index_lines:
        words = line.split()
        for w in words[1:]:
            indices = w.split('/')
            v_index.append(int(indices[0]))
            if uv:  # if contains uv coords.
                uv_index.append(int(indices[1]))
            if vn:  # if contains normals
                vn_index.append(int(indices[2]))
    v_out  = []
    vn_out = []
    uv_out = []
    for i in v_index:
        j = 3*(i - 1)
        v_out = v_out + v[j:j + 3]
    for i in uv_index:
        j = 2*(i - 1)
        uv_out = uv_out + uv[j:j + 2]
    for i in vn_index:
        j = 3*(i - 1)
        vn_out = vn_out + vn[j:j + 3]
    v_res = np.array(v_out, dtype='float32')
    if vn_out:
        vn_res = np.array(vn_out, dtype='float32')
    else:
        vn_res = None
    if uv_out:
        uv_res = np.array(uv_out, dtype='float32')
    else:
        uv_res = None
    return v_res, uv_res, vn_res


def indexVBO(vertices, uvs=None, vns=None):
    """
    Given separate numpy arrays for vertex position/uv coords/normals, reduces
    the data size by constructing an index array indexing the smaller
    counterparts of vertex, uv, normal arrays.

    Returns index_array, indexed_vertices, indexed_uvs, indexed_normals.
    All return arrays that correspond to a None input parameter is also None.

    Arguments
    --------------
    vertices: Numpy array of size N*3 where N is the vertex count. A vertex is
    stored as consecutive x, y, z coordinates according to OpenGL conventions.

    uvs: Numpy array of size N*2; u, v coordinates.

    vns: Numpy array of size N*3; x, y, z coordinates of the normal vector.

    Outputs
    --------------
    indices: Numpy array of size N, containing an index to the accompanying data
    arrays.

    indexed_vertices: Numpy array of size 3*N. Data is stored in the format
    (x, y, z)

    indexed_uvs: Numpy array of size 2*N containing the u, v coordinates.

    indexed_normals: Numpy array of size 3*N containing x, y, z coordinates of
    normal of each vertex.

    Complexity
    --------------
    O(NlogN) where N is vertex count.

    Todo
    --------------
    Generalize the function in the case of more attributes for a vertex.
    """
    num_v = len(vertices)//3
    v_dim = 3
    try:
        uv_dim = 2
        assert len(vertices)*uv_dim == len(uvs)*v_dim
    except:
        uv_dim = 0
    try:
        vn_dim = 3
        assert len(vertices)*vn_dim == len(vns)*v_dim
    except:
        vn_dim = 0
    num_rows = num_v
    num_cols = v_dim + uv_dim + vn_dim + 1  # 1 slot to keep index
    # each row of combined: (x, y, z, [u, v,] [xn, yn, zn,] index)
    combined = np.zeros((num_rows, num_cols), dtype='float32')
    for i in range(len(combined)):
        j, k, m = i*v_dim, i*uv_dim, i*vn_dim
        combined[i][:v_dim]  = vertices[j:j + v_dim]
        try:
            combined[i][v_dim:v_dim + uv_dim] = uvs[k:k + uv_dim]
        except:
            pass
        try:
            combined[i][v_dim + uv_dim:v_dim + uv_dim + vn_dim] = normals[m:m + vn_dim]
        except:
            pass
        combined[i][v_dim + uv_dim + vn_dim] = i
    # this is the same as first sorting the 1st column, then 2nd column, etc.
    # in a stable manner. That is, subsequent sorts doesn't change the order of
    # the previous columns.
    combined = combined[np.lexsort(np.flip(combined.T, axis=0))]
    # accumulator arrays
    indices = np.zeros(num_v, dtype='int16')
    indexed_vertices = np.zeros(len(vertices), dtype='float32')
    try:
        indexed_uvs = np.zeros(len(uvs), dtype='float32')
    except TypeError:
        indexed_uvs = None
    try:
        indexed_vns = np.zeros(len(vns), dtype='float32')
    except TypeError:
        indexed_vns = None
    # indices to be used
    j = int(combined[0][-1]*v_dim)   # vertex index of first row
    k = int(combined[0][-1]*uv_dim)  # uv index of first row
    m = int(combined[0][-1]*vn_dim)  # normal index of first row
    # initialize accumulators
    indices[int(combined[0][-1])] = 0
    indexed_vertices[:v_dim] = vertices[j:j + v_dim]
    try:
        indexed_uvs[:uv_dim] = uvs[k:k + uv_dim]
    except TypeError:
        pass
    try:
        indexed_vns[:vn_dim] = vns[m:m + vn_dim]
    except TypeError:
        pass
    # number of distinct vertices
    v_count = 0
    for i in range(1, len(combined)):
        # if neighbours are not the same, use a yet unused vertex. Don't compare
        # the last columns since they are indices.
        if not np.allclose(combined[i][:-1], combined[i - 1][:-1]):
            v_count += 1
            j = int(combined[i][-1]*v_dim)
            k = int(combined[i][-1]*uv_dim)
            m = int(combined[i][-1]*vn_dim)
            indexed_vertices[v_count*v_dim:v_count*v_dim + v_dim] = vertices[j:j + v_dim]
            try:
                indexed_uvs[v_count*uv_dim:v_count*uv_dim + uv_dim] = uvs[k:k + uv_dim]
            except TypeError:
                pass
            try:
                indexed_vns[v_count*vn_dim:v_count*vn_dim + vn_dim] = vns[m:m + vn_dim]
            except TypeError:
                pass
        indices[int(combined[i][-1])] = v_count
    # trim the zeros from the end of data arrays
    indexed_vertices = np.trim_zeros(indexed_vertices)
    try:
        indexed_uvs = np.trim_zeros(indexed_uvs)
    except TypeError:
        pass
    try:
        indexed_vns = np.trim_zeros(indexed_vns)
    except TypeError:
        pass
    return indices, indexed_vertices, indexed_uvs, indexed_vns
