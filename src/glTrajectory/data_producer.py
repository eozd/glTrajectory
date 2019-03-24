from threading import Thread
import time


class DataProducerThread(Thread):
    """
    Given a numpy data object, produces a new data point and puts it into a
    synchronized queue at each timestamp in real time.
    """
    def __init__(self, producerCallable, dataQueue, delay):
        Thread.__init__(self)
        self.callable = producerCallable
        self.dataQueue = dataQueue
        self.delay = delay

    def run(self):
        """
        Produces a new data point at each timestamp after the start of this
        Thread object and puts it into dataQueue.
        """
        t = 0
        while True:
            time.sleep(self.delay)
            self.dataQueue.put(self.callable(t))
            t += 1
