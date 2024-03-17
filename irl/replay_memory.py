from collections import namedtuple
import random
import torch

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'states_all', 'actions_all', 'rewards_all'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory
    def get_actions(self):
        """Returns a list of all action tensors stored in the memory."""
        return [transition.state for transition in self.memory]
    def print_structure_and_dimensions(self,data, level=0):
        indent = '  ' * level
        if isinstance(data, tuple):
            print(f"{indent}Tuple: Length {len(data)}")
            for i, item in enumerate(data):
                print(f"{indent}  Element {i}:")
                self.print_structure_and_dimensions(item, level + 2)
        elif isinstance(data, list):
            print(f"{indent}List: Length {len(data)}")
            for i, item in enumerate(data):
                print(f"{indent}  Element {i}:")
                self.print_structure_and_dimensions(item, level + 2)
        elif isinstance(data, torch.Tensor):
            print(f"{indent}Tensor: Shape {data.size()}")
        else:
            print(f"{indent}Unknown type: {type(data)}")
    def __len__(self):
        return len(self.memory)
