"""
Configuration file for MathAgent.
Uses AWS Bedrock for Claude API access.
"""

# AWS Bedrock configuration
USE_AWS_BEDROCK = True
AWS_REGION = "<region>"  # AWS region of the inference profile, e.g. us-east-2
# Cross-region inference profile
MODEL_ID = "arn:aws:bedrock:<region>:<account-id>:application-inference-profile/<profile-id>"

# Fallback: Direct Anthropic API (not used if USE_AWS_BEDROCK=True)
ANTHROPIC_API_KEY = "your-api-key-here"
MODEL_NAME = "claude-sonnet-4-6-20250514"

# Edge inference settings
EDGE_INFERENCE_MODEL_ID = "arn:aws:bedrock:<region>:<account-id>:application-inference-profile/<profile-id>"
SIMILARITY_THRESHOLD = 0.75  # Min confidence to keep an edge
EDGE_INFERENCE_PARALLEL = True   # Use ThreadPoolExecutor for parallel inference
EDGE_INFERENCE_MAX_WORKERS = 50  # Max concurrent API calls for edge inference

# Benchmark evaluation settings
EVAL_PARALLEL = True             # Use ThreadPoolExecutor for parallel evaluation
EVAL_MAX_WORKERS = 192           # Max concurrent Lean compilations (~3GB RAM each, needs ~576GB)
                                 # Sized for a 768 GB machine; override with --workers on smaller machines.

# KG Search settings (Explorer agent)
KG_SEARCH_SCORING_METHOD = "llm" # "llm" (LLM scores candidates) or "keyword" (algorithmic)
KG_SEARCH_NUM_SEEDS = 5          # Number of seed theorems from scoring
KG_SEARCH_HOP_DEPTH = 1          # Edge traversal depth from seeds (default 1)
KG_SEARCH_MAX_CANDIDATES = None  # Max candidates to Prover (None = no limit)

# Lean 4 verification settings
LEAN_TIMEOUT_SECONDS = 1200      # Max seconds (20 min) to wait for lean CLI per proof
LEAN_TEMP_DIR = "/tmp/mathagent_lean"  # Temp directory for .lean files
LEAN_MATHLIB_PROJECT = "/path/to/mathlib4"  # Path to Mathlib4

# Agent settings
AGENT_TEMPERATURE = 0.7
AGENT_MAX_PROOF_ATTEMPTS = 10   # Hard cap on Prover retries per conjecture
AGENT_CONJECTURE_CONFIDENCE_THRESHOLD = 0.30  # Min confidence to pass conjecture to Prover (low for small dataset)
AGENT_MAX_ITERATIONS = 5        # Default discovery loop iterations
