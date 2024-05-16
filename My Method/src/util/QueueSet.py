#Picked up idea from https://stackoverflow.com/questions/5997189/how-can-i-make-a-unique-value-priority-queue-in-python

from collections import deque

class QueueSet(object):
    def __init__(self):
        self.queue = deque()
        self.set = set()

    def add(self, ele):
        if not ele in self.set:
            self.queue.append(ele)
            self.set.add(ele)

    def pop(self):
        return self.queue.popleft()
    
    def isempty(self):
        if len(self.queue):
            return 0
        return 1