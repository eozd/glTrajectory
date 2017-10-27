from threading import Thread
import time


class DataProducerThread(Thread):
    """
    Given a numpy data object, produces a new data point and puts it into a
    synchronized queue at each timestamp in real time.
    """

    def setData(self, data, dataQueue):
        self.data = data
        # we need the interarrival times
        self.data[1:, 0] = self.data[1:, 0] - self.data[:-1, 0]
        self.data[0, 0] = 0
        self.dataQueue = dataQueue

    def run(self):
        """
        Produces a new data point at each timestamp after the start of this
        Thread object and puts it into dataQueue.
        """
        for row in self.data:
            time.sleep(row[0])
            self.dataQueue.put(row[1:4])
