from abc import abstractmethod
from typing import List, Union

import torch.nn as nn


class BaseTextEmbedding(object):
    """
    Base class for all embeddings
    """

    @abstractmethod
    def get_embedding(self, x: Union[List[str], str]):
        """
        Forward pass logic
        :return: Model output
        """
        return self.model(x)
