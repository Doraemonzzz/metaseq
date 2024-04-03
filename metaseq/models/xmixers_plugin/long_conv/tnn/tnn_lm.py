import logging

from metaseq.models import register_model, register_model_architecture

logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM
from xmixers.models import TnnConfig

from metaseq.models.hf_base_model import HfBaseModel
from metaseq.models.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerLanguageModelConfig,
)
from metaseq.utils import convert_to_multiple_of_base


@register_model("tnn_lm", dataclass=TransformerLanguageModelConfig)
class TnnLanguageModel(HfBaseModel):
    def __init__(self, decoder):
        super(TnnLanguageModel, self).__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        task.source_dictionary.pad_to_multiple_(8)
        task.target_dictionary.pad_to_multiple_(8)

        hf_config = TnnConfig.from_dict(OmegaConf.to_container(args, resolve=True))
        hf_config.update({"vocab_size": len(task.target_dictionary)})

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        decoder = AutoModelForCausalLM.from_config(hf_config)
        return cls(decoder)


##### base test
@register_model_architecture("tnn_lm", "tnn_lm_385m")
def tnn_lm_385m(args):
    # gtu config
    args.num_layers = 26
    args.embed_dim = 1024
    args.expand_ratio = 1
    args.bias = False
    args.gtu_activation = "silu"
    args.causal = True
    args.norm_type = "simplermsnorm"
    args.use_decay = True
    args.rpe_in_dim = 1
    args.rpe_feature_dim = 32
    args.rpe_layers = 3
    args.dims = [-2]
    # glu config
    args.mid_dim = convert_to_multiple_of_base(int(3 * args.embed_dim))
    args.glu_activation = "none"
