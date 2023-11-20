from typing import Iterable, List, Union

import numpy as np
import torch
from overrides import overrides
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast

from embedders.base_embedding import BaseTextEmbedding


class HFSentenceTransformer(BaseTextEmbedding):
    """
    Wrapper for Huggingface Sentence Transformer
    """

    def __init__(
        self,
        model_name_or_path: Union[str, type(None)] = None,
        modules: Union[Iterable[torch.nn.modules.module.Module], type(None)] = None,
        device: Union[str, type(None)] = None,
        cache_folder: Union[str, type(None)] = None,
    ):
        super(HFSentenceTransformer, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.modules = modules
        self.device = device
        self.cache_folder = cache_folder

        self.transformer_kwargs = {
            "model_name_or_path": self.model_name_or_path,
            "modules": self.modules,
            "device": self.device,
            "cache_folder": self.cache_folder,
        }

        self.model = SentenceTransformer(**self.transformer_kwargs)

    @overrides
    def get_embedding(self, x: Union[List[str], str]):
        """
        Forward pass logic
        :return: Model output
        """
        return self.model.encode(x)
