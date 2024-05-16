'''
value: Value of feature

feature_name: Some string indicating feature name

feature_index: Indicates the index of feature in input vector (useful for deriving the equation from equations matrix in SCM class)

parents: Indicates the addresses of the parents

children: Indicates the address of the children



NOTE: GRAPH MUST HAVE AT LEAST ONE ROOT NODE WITH NO PARENTS
'''

import torch

class Node():
    
    def __init__(self, feature_value=0, feature_name=None, feature_level:int = 0, feature_index:int=0, parent_list=None, children_list = None):
        self.value = feature_value
        self.name = feature_name
        self.level = feature_level
        self.index = feature_index
        self.parents = parent_list
        self.children = children_list
    
    #comparison between Node objects when level matches during insertion in PrioritySet queues
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self is other)

    #Check level first -> if matching then value -> if matching then go by index
    def __lt__(self, other):
        if self.level<other.level:
            return 1
        elif self.level>other.level:
            return 0
        else:
            if self.value<other.value:
                return 1
            elif self.value>other.value:
                return 0
            else:
                return (self.index < other.index)

    def __gt__(self, other):
        if self.level<other.level:
            return 0
        elif self.level>other.level:
            return 1
        else:
            if self.value<other.value:
                return 0
            elif self.value>other.value:
                return 1
            else:
                return (self.index > other.index)

    def __le__(self, other):
        return (self.level <= other.level)

    def __ge__(self, other):
        return (self.level >= other.level)
    
    #Need to define __hash__ function for using sets in PrioritySet
    def __hash__(self):
        return hash((self.index, self.level))
        
    
