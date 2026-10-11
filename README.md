# When Does Structured Knowledge Help Neural Theorem Proving?

[![arXiv](https://img.shields.io/badge/arXiv-2609.34460-b31b1b.svg)](https://arxiv.org/abs/2609.34460)
[![Project page](https://img.shields.io/badge/project-page-blue.svg)](https://sarehnabi.github.io/mathagent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code, data, and results for **When Does Structured Knowledge Help Neural Theorem Proving?**
(Sareh Nabi, Roland Vogl, Marzieh Nabi; Stanford University), [arXiv:2609.34460](https://arxiv.org/abs/2609.34460).

The paper asks whether structured mathematical knowledge helps LLMs prove theorems in Lean 4, for
which models, and whether the answer varies by problem. **MathAgent** augments LLM theorem provers
with **MathKG**, a semantic knowledge graph over 364 Mathlib theorems and definitions connected by
9,434 typed edges, and with Mathlib library retrieval. We run a controlled ablation across four
augmentation modes (`prover_only`, `with_kg`, `with_mathlib`, `full_system`) and five models
(Qwen3-8B/32B, Goedel-Prover-V2-8B/32B, Claude Sonnet 4.6) on miniF2F, PutnamBench, and
MathOlympiadBench. Three findings emerge: Lean specialization dominates augmentation; augmentation's
aggregate effect depends on the model's capability; and the augmentation modes solve different problems, so
choosing the mode per problem beats any fixed mode. Together these results motivate adaptive
strategies that select augmentation by model capability and problem.

This repository releases everything behind the paper: the complete implementation (the Explorer,
Conjecturer, Prover, and Integrator agents, the Orchestrator, MathKG construction, and the benchmark
evaluator with its Lean 4 verifier), the MathKG graph and node files
(364 nodes, 9,434 typed edges with confidence and reasoning), the per-problem results of all 28
evaluation runs (status, attempts, final Lean code, and errors for every problem), and the scripts
that launched the runs.

The project page is at https://sarehnabi.github.io/mathagent/. It gives an overview of the paper
and an interactive view of MathKG.

## Setup

The open-weight runs used a single multi-GPU machine with vLLM; the Sonnet runs used AWS Bedrock.

1. **Create the Python environment** (Python 3.10+).
   ```bash
   pip install -r requirements.txt
   ```

2. **Install `elan` (Lean toolchain manager):**
   ```bash
   curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
   ```

3. **Set up Mathlib.** Clone mathlib4 and download its prebuilt cache. The runs used `v4.30.0-rc2`
   (Sonnet) and `v4.32.0-rc1` (Goedel-Prover-V2 and Qwen3-32B):
   ```bash
   git clone https://github.com/leanprover-community/mathlib4
   cd mathlib4 && git checkout v4.32.0-rc1 && lake exe cache get
   ```
   Point the verifier at it in `config.py` (`LEAN_MATHLIB_PROJECT`) or with the environment
   variable of the same name.

4. **Configure the model backend.** For Claude via AWS Bedrock, set `MODEL_ID` in `config.py` to your
   inference profile; for the Anthropic API instead, set `USE_AWS_BEDROCK = False` and
   `ANTHROPIC_API_KEY`. For an open-weight model, start a vLLM server for it (for example
   `vllm serve Goedel-LM/Goedel-Prover-V2-8B`) and point the code at that server:
   ```bash
   export USE_LOCAL_VLLM=true VLLM_ENDPOINT=http://localhost:8000/v1 VLLM_MODEL=Goedel-LM/Goedel-Prover-V2-8B VLLM_THINKING=off
   ```

5. **Fetch the benchmarks and the Mathlib declaration index.** The benchmarks are not
   redistributed; the index is downloaded and preprocessed by the script.
   ```bash
   python -m src.benchmarks.download --benchmark all
   bash scripts/download_mathlib_data.sh
   ```

## Usage

### Reproducing the graph

```bash
python main.py --mode pipeline --data data/mathlib_fundamentals_verified.json
```

runs relation extraction over all 132,132 ordered node pairs, post-processes the
edges, and writes `mathkg.json` and `edges.json` to a timestamped `outputs/experiments/` directory.

### Running one ablation cell

```bash
python -m src.benchmarks.evaluate --benchmark minif2f --eval-mode with_kg \
    --kg-path outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json \
    --max-attempts 32 --workers 20 --output-dir outputs/experiments/my_run/minif2f_with_kg
```

`scripts/run_full_ablation.sh` loops this over the four augmentation modes and the three benchmarks (next subsection).

Add `--save-trajectories` to also write `trajectories_<benchmark>_<mode>.jsonl`: one line per problem with
the Explorer's context block, the supporting MathKG node ids, and every attempt's Lean code and verdict.
The result file is unchanged by the flag.

### Running the full experiment

Run one model through all four modes and all three benchmarks with:

```bash
bash scripts/run_full_ablation.sh --workers 20 --max-attempts 32 \
    --kg-path outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json
```

## Repository structure

| Path | Description |
|---|---|
| `main.py` | Graph construction (`--mode pipeline`) and the open-ended agent loop (`--mode agent`, implemented but not evaluated in the paper) |
| `src/agents/` | Explorer, Conjecturer, Prover, Integrator, Orchestrator, shared LLM client |
| `src/extract_mathlib.py`, `src/infer_edges.py`, `src/build_graph.py` | Node-file loading, LLM relation extraction with post-processing, NetworkX graph construction |
| `src/benchmarks/evaluate.py` | Targeted proving on miniF2F, PutnamBench, MathOlympiadBench under the four augmentation modes |
| `src/lean4_verifier.py`, `src/statement_check.py` | Lean 4 compilation and the statement-fidelity check (paper, Appendix F) |
| `scripts/` | Run launchers, the PutnamBench merge, data downloads, and the Figure 4 plot (table below) |
| `site/` | The Astro project page (deployed to GitHub Pages by `.github/workflows/astro.yml`) |

Scripts:

| Script | Role |
|---|---|
| `scripts/run_full_ablation.sh` | Main experiment: one model, four augmentation modes, on miniF2F, PutnamBench, and MathOlympiadBench; writes one `full_ablation_*` directory |
| `scripts/run_putnam_rerun.sh` | PutnamBench answer-slot problems (346) with the benchmark's answers inlined and the statement check on (paper, Appendix F) |
| `scripts/merge_putnam_rerun.py` | Merge the answer-slot run with the main run's 326 slot-free problems into one 672-problem PutnamBench result per mode |
| `scripts/download_mathlib_data.sh` | Download and preprocess the Mathlib declaration index used by Layer 2 retrieval |
| `scripts/reprocess_edges.py` | Re-apply edge post-processing (type restrictions, deduplication, conflict resolution, inverses) to a cached `edges.json` without re-running the LLM |
| `scripts/plot_venn_complementarity.py` | Generate Figure 4 (complementarity Venn diagrams) from the result files |
| `scripts/compute_paper_numbers.py` | Compute the paper's tables from the released result files (Tables 2, 3, 4, 8, 9, 10, 12, Figures 3 and 4, McNemar p-values); `--check` compares every number with the paper |

## Released data and results

| Path | Description |
|---|---|
| `data/mathlib_fundamentals_verified.json` | MathKG node file: 364 theorems and definitions with name, domain, natural-language statement, keywords, and verified Mathlib declaration name and import |
| `outputs/experiments/exp_04302026_0611_mathlib_fundamentals_verified_364n/mathkg_clean.json` | The released graph: 364 nodes, 9,434 edges |
| `outputs/experiments/<run>/<benchmark>_<mode>/eval_*.json` | Per-problem results of the 28 evaluation runs (status, attempts, final Lean code, errors) with each run's `run_config.json` |

Each record in a run's `<benchmark>_<mode>/eval_<benchmark>_<mode>.json` has the fields:

- `problem_id`, `domain`
- `status`: `verified` (a solve counted in the paper), `failed`, `error`, or `statement_check_failed`
  (a proof that compiles but fails the statement-fidelity check of Appendix F)
- `attempts`: the number of attempts used, 1 to 32
- `lean4_code`: the final attempt's Lean file
- `errors`: the Lean error messages of the final attempt, up to three; empty when solved
- `time_seconds`, `rejected_empty_verifications`

The evaluator kept only the final attempt of each problem, so per-attempt trajectories and the
Explorer's retrieval contexts were not persisted for these runs. The evaluator now has a
`--save-trajectories` flag (see Usage) that writes them for new runs.

`python scripts/compute_paper_numbers.py` computes every table of the paper from these files;
`--check` verifies each number against the published values.

Each run directory holds one subfolder per augmentation mode (`prover_only`, `with_kg`,
`with_mathlib`, `full_system`):

| Run directory | Model | Benchmarks |
|---|---|---|
| `full_ablation_05162026_2359_claude_sonnet` | Claude Sonnet 4.6 | miniF2F, MathOlympiadBench |
| `putnam_merged_09262026_claude_sonnet` | Claude Sonnet 4.6 | PutnamBench (672 problems; see note) |
| `full_ablation_05182026_2052_qwen3_8b` | Qwen3-8B | miniF2F |
| `full_ablation_08292026_qwen3_32b` | Qwen3-32B | miniF2F |
| `full_ablation_08272026_goedel_v2_8b` | Goedel-Prover-V2-8B | miniF2F |
| `full_ablation_08272026_goedel_v2_32b` | Goedel-Prover-V2-32B | miniF2F |

Note: the PutnamBench directory combines the 346 answer-slot problems from a dedicated run with the
benchmark's answers inlined and the 326 slot-free problems from the main Sonnet run, merged by
`scripts/merge_putnam_rerun.py`; each record names its source and `run_config.json` holds both
source configurations.

The paper's tables map to these files as follows:

- Tables 2, 3, 4 and Figures 2, 3 (miniF2F, five models): the `minif2f_<mode>` results of the
  five run directories; a solve is a record with status `verified`.
- Tables 8, 9, 10 and Figure 4 (Sonnet, three benchmarks): the Sonnet run directory and the
  merged PutnamBench directory.

## Versions

| Tag | What it is |
|---|---|
| `v1.0` | The code, data, and results as released with the arXiv paper (arXiv:2609.34460v1). |
| `v1.1` | Adds the conference poster, the `--save-trajectories` evaluator flag, `scripts/compute_paper_numbers.py`, and this documentation. Released data and results are unchanged. |

To get exactly the paper's version: `git checkout v1.0`.

## Project page

The project page, with an overview of the paper and an interactive view of MathKG, is at
https://sarehnabi.github.io/mathagent/.

The page is an [Astro](https://astro.build) project in `site/`; GitHub Pages deploys it from `main`
via `.github/workflows/astro.yml`. To preview it locally, run `cd site && npm install && npm run dev`
and open http://localhost:4321/.

## Citation

```bibtex
@article{nabi2026structured,
  title={When Does Structured Knowledge Help Neural Theorem Proving?},
  author={Nabi, Sareh and Vogl, Roland and Nabi, Marzieh},
  journal={arXiv preprint arXiv:2609.34460},
  year={2026}
}
```

## License

Code and released data are under the MIT License (see `LICENSE`).

## Acknowledgments

We gratefully acknowledge the Laude Institute for providing the compute resources that made this
research possible. The project page is built with
[Roman Hauksson's academic project page template](https://github.com/RomanHauksson/academic-project-astro-template).
