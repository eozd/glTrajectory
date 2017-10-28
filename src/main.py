import argparse
import sys
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout)
import numpy as np
import queue

import data_producer
from gl_trajectory_widget import GLTrajectoryWidget


if __name__ == '__main__':
    sys.path.insert(0, '..')
    parser = argparse.ArgumentParser(description="Animate trajectory using GLTrajectoryWidget.")
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument('-c', metavar='config_file', help="Configuration file path", required=True)
    requiredNamed.add_argument('-d', metavar='data_file', help="Data file path", required=True)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(-1)
    args = parser.parse_args()
    dataFile = args.d
    configFile = args.c

    fmt = QSurfaceFormat()
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(3)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setDepthBufferSize(24)
    fmt.setAlphaBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    # read data file
    data = np.genfromtxt(dataFile, delimiter=',', dtype=str)
    data = np.char.replace(data, '[', '')
    data = np.char.replace(data, ']', '')
    data = data.astype(float)

    # create the synchronized queue
    dataQueue = queue.Queue()
    dataThread = data_producer.DataProducerThread()
    dataThread.setData(data, dataQueue)
    dataThread.setDaemon(True)
    dataThread.start()

    app = QApplication(sys.argv)
    window = QWidget()
    layout = QHBoxLayout()
    layout.addWidget(GLTrajectoryWidget(1368, 768, dataQueue, configFile))
    window.setLayout(layout)

    window.show()
    sys.exit(app.exec_())
