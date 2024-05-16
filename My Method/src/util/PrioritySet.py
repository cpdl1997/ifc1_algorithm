#Picked up idea from https://stackoverflow.com/questions/5997189/how-can-i-make-a-unique-value-priority-queue-in-python

import heapq

class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, d, pri):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)

    def pop(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return d
    
    def isempty(self):
        if len(self.heap):
            return 0
        return 1
    
    def clear(self):
        self.heap.clear()
        self.set.clear()