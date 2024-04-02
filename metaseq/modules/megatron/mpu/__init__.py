# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .cross_entropy import vocab_parallel_cross_entropy
from .initialize import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
)
from .layers import (
    ColumnParallelLinear,
    LinearWithGradAccumulationAndAsyncCommunication,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from .mappings import (
    _gather_along_first_dim,
    _reduce_scatter_along_first_dim,
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import (
    gather_split_1d_tensor,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
    split_tensor_into_1d_equal_chunks,
)
from .utils import (
    VocabUtility,
    divide,
    ensure_divisibility,
    split_tensor_along_last_dim,
)
