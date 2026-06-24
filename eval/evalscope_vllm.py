#!/usr/bin/env python3
"""
Wrapper for running evalscope evaluations against a vLLM server (OpenAI-compatible API).

Usage:
    python eval/evalscope_vllm.py <yaml_path> \\
        --model <served_model_name> \\
        --work-dir <output_dir> \\
        --dataset-dir <dataset_cache_dir> \\
        --model-id <model_id> \\
        --api-url <vllm_api_url> \\
        --generation-config '{"seed": 42, "max_tokens": 32768}'

The YAML config provides dataset selection and default generation parameters.
CLI arguments override values in the YAML at runtime.
"""

import argparse
import json
import sys
from pathlib import Path

from evalscope.config import TaskConfig
from evalscope.run import run_task
from evalscope.utils.io_utils import yaml_to_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evalscope evaluation via vLLM OpenAI-compatible API"
    )
    parser.add_argument(
        "yaml_path",
        type=str,
        help="Path to the YAML task config file (eval/yamls/vllm/<base_model>/<task>.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Served model name (as registered in vLLM --served-model-name)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        dest="work_dir",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        dest="dataset_dir",
        help="Local directory for cached datasets (ModelScope cache root)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        dest="model_id",
        help="Model identifier used for labeling results (e.g. Qwen3-4B)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        dest="api_url",
        help="vLLM OpenAI-compatible chat completions URL, e.g. http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--generation-config",
        type=str,
        dest="generation_config",
        help='JSON string of generation config overrides, e.g. \'{"seed": 42, "max_tokens": 32768}\'',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base config from YAML
    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        print(f"Error: YAML config file not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    config_dict = yaml_to_dict(str(yaml_path))

    # Override with CLI arguments
    if args.model:
        config_dict["model"] = args.model
    if args.work_dir:
        config_dict["work_dir"] = args.work_dir
    if args.dataset_dir:
        config_dict["dataset_dir"] = args.dataset_dir
    if args.model_id:
        config_dict["model_id"] = args.model_id
    if args.api_url:
        config_dict["api_url"] = args.api_url

    # Merge generation config (CLI overrides YAML defaults)
    if args.generation_config:
        gen_cfg_override = json.loads(args.generation_config)
        existing_gen_cfg = config_dict.get("generation_config", {})
        if isinstance(existing_gen_cfg, dict):
            existing_gen_cfg.update(gen_cfg_override)
        else:
            existing_gen_cfg = gen_cfg_override
        config_dict["generation_config"] = existing_gen_cfg

        # Also propagate seed to top-level TaskConfig.seed
        if "seed" in gen_cfg_override:
            config_dict["seed"] = gen_cfg_override["seed"]

    # Build TaskConfig and run evaluation
    task_cfg = TaskConfig.from_dict(config_dict)
    result = run_task(task_cfg)

    print("\n=== Evaluation Results ===")
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
