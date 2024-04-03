from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM

from metaseq import utils
from metaseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class HfBaseDecoder(nn.Module):
    def __init__(self, args, dictionary, hf_config):
        super().__init__()
        self.dictionary = dictionary
        self.max_positions = args.tokens_per_sample
        self.max_position_embeddings = args.tokens_per_sample
        self.model = AutoModelForCausalLM.from_config(hf_config)

    def forward(self, prev_output_tokens, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`. Note that this
                dictionary is modified inline iff incremental_state is not None.

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        return self.model(prev_output_tokens)

    def extract_features(self, prev_output_tokens, incremental_state=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """

    def get_normalized_probs(self, logits: Tensor, log_probs: bool):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number
