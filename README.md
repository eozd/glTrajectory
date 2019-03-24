## glTrajectory ![build status](https://travis-ci.org/eozd/glTrajectory.svg?branch=master)
This is a project to visualize 3D trajectory data using python and OpenGL. It
uses PyOpenGL to direct GPU using OpenGL, and PyQt5 to create a context window for
OpenGL graphics.

### Installation
glTrajectory requires python3.5+. To install the requirements type
```
pip install -r requirements.txt
```

### Configuration
You can configure various parameters of the visualization by editing ```config.json``` file.

### Running
After you are content with the current configuration, simply start running the visualization via
```
python src/glTrajectory/main.py -c config.json
```

Currently, the program visualizes a simple circular motion trajectory around origin.
You can use your own data producer methods, and visualize the data they produce. To do
this, you need to create a function as

```python
def myProducer(t, *args, **kwargs)
```

that takes the current tick count ```t```, and returns the current object position as a
numpy array of size 3. Then, you can pass this function as parameter to
```data_producer.DataProducer``` object, and the rest will be handled. See the
simple example at ```main.py``` for further details.

### Tests
To run the current set of tests, simply run ```pytest``` project root directory.

### Help
To see the help related to program parameters, type
```
python src/glTrajectory/main.py
```
