import numpy as np

class BaseCF(object):
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def generate_counterfactuals(self):
        raise NotImplementedError


  