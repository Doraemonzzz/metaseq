import logging

from metaseq.models import register_model, register_model_architecture

logger = logging.getLogger(__name__)

from transformers import AutoModel
from xmixers.models import TnnConfig

from metaseq.models.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
)


@register_model("tnn_lm", dataclass=TransformerLanguageModelConfig)
class TnnLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TnnLanguageModel, self).__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        task.source_dictionary.pad_to_multiple_(8)
        task.target_dictionary.pad_to_multiple_(8)
        print(type(args))
        from omegaconf import OmegaConf

        hf_config = TnnConfig.from_dict(OmegaConf.to_container(args, resolve=True))
        hf_config.update({"vocab_size": len(task.target_dictionary)})

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        decoder = AutoModel.from_config(hf_config)
        return cls(decoder)


##### base test
@register_model_architecture("tnn_lm", "tnn_lm_test")
def tnn_lm_test(args):
    pass

    hf_config = TnnConfig().to_dict()
    for k in hf_config:
        args.k = hf_config[k]
    # print(hf_config)

    # print(type(args))
    # print(args)
    # args = Namespace(**vars(args), **vars(hf_config))

    # print(args)
    # from omegaconf import OmegaConf
    # OmegaConf.update(args, {"config": TnnConfig()})
