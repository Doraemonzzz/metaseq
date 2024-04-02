#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fire

from metaseq.distributed.stitch_fsdp_ckpt import consolidate_fsdp_shards

if __name__ == "__main__":
    # This is expected to be used before evaluation, not during training.
    fire.Fire(consolidate_fsdp_shards)
