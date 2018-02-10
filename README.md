## glTrajectory ![build status](https://travis-ci.org/eozd/glTrajectory.svg?branch=master)
This is a project to visualize 3D trajectory data using python and OpenGL. It
uses PyOpenGL to direct GPU using OpenGL, and PyQt5 to create a context window for
OpenGL graphics.

### Installation
glTrajectory requires python3.5+. To install the requirements type
```
pip install -r requirements.txt
```

### Running
To visualize a trajectory using sample data, type
```
./run
```

```run``` is a simple shell script to pass the necessary configuration and data parameters
to the main program. To customize how you run the project, simply edit the ```run``` script.

### Configuration
To configure various parts of the visualization, edit ```conf/config_glwidget.json``` file.

### Help
To see the help related to program parameters, type
```
./run -h
```
