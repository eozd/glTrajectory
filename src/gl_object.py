from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ctypes

import loader
import matrix_utils as mu


class GLObject():
    """
    An OpenGL 'object' that has its own vertex data, and which can be drawn onto
    a QOpenGLWidget.
    """

    modelVertexDictionary = {}
    """
    Dictionary for storing vertex data with names as identifiers. Used by all
    GLObjects.
    """

    MatrixID = None
    MID = None
    ColorID = None
    """
    Uniform IDs holding the corresponding data structures in the shader program.
    """

    def __init__(self, modelName):
        """
        modelName: Name of the GLVertexData model stored in modelVertexDictionary.
        color: <r, g, b, a> color of the object as a numpy array.
        """
        self.modelName = modelName
        self.initBuffers()

    def initBuffers(self):
        """
        Creates OpenGL buffers for storing vertex data.
        """
        glBindVertexArray(glGenVertexArrays(1))
        vertexData = GLObject.modelVertexDictionary[self.modelName]
        # vertex array (positions)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, vertexData.vertices, GL_STATIC_DRAW)
        # normal array (normal vector of each triangle)
        self.normalBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalBuffer)
        glBufferData(GL_ARRAY_BUFFER, vertexData.normals, GL_STATIC_DRAW)
        # index array (to be used for VBO indexing)
        self.elementBuffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.elementBuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertexData.indices, GL_STATIC_DRAW)

    def initUniforms(programID):
        """
        Initializes uniforms in the shader program to be used by painting
        functions of GLObject.
        """
        GLObject.MatrixID = glGetUniformLocation(programID, "MVP")
        GLObject.MID = glGetUniformLocation(programID, "M")
        GLObject.ColorID = glGetUniformLocation(programID, "MaterialDiffuseColor")

    def initModelVertexDic(namePathDic):
        """
        Initializes GLObject.modelVertexDictionary.
        """
        for name, path in namePathDic.items():
            GLObject.modelVertexDictionary[name] = loader.indexVBO(*loader.loadOBJ(path))

    def paint(self, M, V, P, color):
        """
        Paint GLObject using the transformations given.

        M: Model matrix.
        V: View matrix.
        P: Projection matrix.
        matrixID: Uniform ID of MVP matrix in the vertex shader.
        MID: Uniform ID of M matrix in the vertex shader.
        """
        vertexData = GLObject.modelVertexDictionary[self.modelName]
        MVP = mu.mul(P, mu.mul(V, M))
        glUniformMatrix4fv(GLObject.MatrixID, 1, False, MVP)
        glUniformMatrix4fv(GLObject.MID, 1, False, M)
        glUniform4fv(GLObject.ColorID, 1, color)
        # vertex attribute array
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(
            0,                   # must match the layout id in the shader
            3,                   # size
            GL_FLOAT,            # data type
            GL_FALSE,            # normalized?
            0,                   # stride. offset in between
            ctypes.c_void_p(0),  # offset to the beginning
        )
        # normal coordinates attribute array
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalBuffer)
        glVertexAttribPointer(
            1,
            3,
            GL_FLOAT,
            GL_FALSE,
            0,
            ctypes.c_void_p(0),
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.elementBuffer)
        glDrawElements(
            GL_TRIANGLES,
            len(vertexData.indices),
            GL_UNSIGNED_SHORT,
            ctypes.c_void_p(0),
        )
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
