

# Metaseq
A codebase for working with [Open Pre-trained Transformers](projects/OPT), originally forked from [fairseq](https://github.com/facebookresearch/fairseq).


## Install
"""
pip install -e .
"""
Optional:
```
pip install fairscale
```

## Todo
- [x] Support training models compliant with the HF interface.
- [x] Support DDP training resume.
- [ ] Support HF, Sp, Tik tokenizers.
- [ ] Support continue training for hf models.
- [ ] Add a script to convert to hf format.
- [ ] Add more stat for hf logging.

## License

The majority of metaseq is licensed under the MIT license, however portions of the project are available under separate license terms:
* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
