"""LatentPrism: Mechanistic Interpretability, Crosscoders, Conformal Probes, and Nullspace Steering."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from latent_prism.activation_corpus import (
    MultiLayerActivationDataset,
    TransformerBackboneRunner,
    build_structured_prompt_corpus,
    extract_multilayer_activation_dataset,
)
from latent_prism.benchmark_runner import run_full_latent_prism_benchmark
from latent_prism.concept_probes_and_elk import (
    CalibratedLatentConceptProbeSuite,
    LinearConceptProbe,
    binary_auroc,
)
from latent_prism.crosscoder_and_transcoder import (
    MultiLayerSparseCrosscoderAndTranscoderSuite,
    orthogonal_procrustes_2d,
)
from latent_prism.grid_and_conformance import (
    DomainAgnosticNamingGate,
    GridLedger,
    NodeGrid,
    NodeParameter,
    OperationCostLedger,
    ParameterBounds,
    ZeroHardcodedValuesAstGate,
)
from latent_prism.matryoshka_sae import (
    MatryoshkaTopKAndJumpReluSuite,
    TrainedLayerSAE,
    train_and_evaluate_layer_sae,
)
from latent_prism.nullspace_steering_and_gguf import (
    NullspaceGuardedActivationSteerer,
    parse_gguf_v3_control_vector,
    serialize_gguf_v3_control_vector,
)
from latent_prism.portable_exporter import (
    PortableJumpReluCartridge,
    TenTargetPortableExporter,
    build_sae_probe_cartridge,
    parse_safetensors_bytes,
    reference_f32_cartridge_forward,
    serialize_safetensors_bytes,
)

__version__ = "1.0.0"

__all__ = [
    "CalibratedLatentConceptProbeSuite",
    "DomainAgnosticNamingGate",
    "GridLedger",
    "LinearConceptProbe",
    "MatryoshkaTopKAndJumpReluSuite",
    "MultiLayerActivationDataset",
    "MultiLayerSparseCrosscoderAndTranscoderSuite",
    "NodeGrid",
    "NodeParameter",
    "NullspaceGuardedActivationSteerer",
    "OperationCostLedger",
    "ParameterBounds",
    "PortableJumpReluCartridge",
    "TenTargetPortableExporter",
    "TrainedLayerSAE",
    "TransformerBackboneRunner",
    "ZeroHardcodedValuesAstGate",
    "__version__",
    "binary_auroc",
    "build_sae_probe_cartridge",
    "build_structured_prompt_corpus",
    "extract_multilayer_activation_dataset",
    "orthogonal_procrustes_2d",
    "parse_gguf_v3_control_vector",
    "parse_safetensors_bytes",
    "reference_f32_cartridge_forward",
    "run_full_latent_prism_benchmark",
    "serialize_gguf_v3_control_vector",
    "serialize_safetensors_bytes",
    "train_and_evaluate_layer_sae",
]
