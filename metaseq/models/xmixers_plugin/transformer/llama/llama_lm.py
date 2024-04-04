import logging

from metaseq.models import register_model, register_model_architecture

logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
from transformers import LlamaConfig

from metaseq.models.hf_base_decoder import HfBaseDecoder
from metaseq.models.hf_base_model import HfBaseModel
from metaseq.models.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerLanguageModelConfig,
)
from metaseq.utils import convert_to_multiple_of_base


@register_model("llama_lm", dataclass=TransformerLanguageModelConfig)
class LlamaLanguageModel(HfBaseModel):
    def __init__(self, decoder):
        super(LlamaLanguageModel, self).__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        task.source_dictionary.pad_to_multiple_(8)
        task.target_dictionary.pad_to_multiple_(8)

        hf_config = LlamaConfig.from_dict(OmegaConf.to_container(args, resolve=True))
        hf_config.update({"vocab_size": len(task.target_dictionary)})

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        decoder = HfBaseDecoder(args, task.target_dictionary, hf_config)
        return cls(decoder)


##### base test
@register_model_architecture("llama_lm", "llama_lm_385m")
def llama_lm_385m(args):
    # llama config
    args.num_hidden_layers = 26
    args.hidden_size = 1024
    args.intermediate_size = convert_to_multiple_of_base(int(8 * args.hidden_size / 3))
    args.num_attention_heads = 8
    args.bias = False
    args.hidden_act = "silu"
