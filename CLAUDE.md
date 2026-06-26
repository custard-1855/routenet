# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A TensorFlow 2 reimplementation of RouteNet, a graph neural network (message-passing) that predicts per-path network performance metrics (delay, jitter, packet drops) from network topology, routing, and traffic-matrix inputs. Originally a TF1/Estimator codebase; `routenet/routenet.py` was migrated to TF2 Keras (see `PLAN.md` for the full migration rationale and before/after diffs of every changed API).

`mpnn/` is a separate, older Session/Graph-based MPNN implementation (`graph_nn.py`/`graph_nn2.py`/`eval.py`/`samples.py`) supporting a different paper ("Message-Passing Neural Networks Learn Little's Law"). It is **out of scope** for the TF2 migration — do not port it unless explicitly asked.

## Commands

This project uses `uv`. TensorFlow is pinned to `2.17.0` (in `[dependency-groups].dev`) because `tensorflow-metal==1.2.0` is ABI-incompatible with newer TF — do not bump TF without also checking `tensorflow-metal` compatibility.

```bash
uv sync                                      # install deps

uv run pytest -v                             # unit tests (tests/, per pyproject testpaths)
uv run pytest tests/ -v -k routenet          # model forward-pass tests only
uv run pytest tests/ -v -k collate           # graph-batching (_collate_graphs) tests only
uv run pytest tests/ -v -k pearson           # PearsonCorrelation metric tests only

uv run pytest tasks/ -s -v                   # full training/eval pipeline (see below)
uv run pytest tasks/ -s -v -k download       # just download the dataset
uv run pytest tasks/ -s -v -k extract        # just extract tfrecords
uv run pytest tasks/ -s -v -k train          # smoke training run (~12 min)
uv run pytest tasks/ -s -v -k evaluate       # evaluate, overwrites eval_results.json
uv run pytest tasks/ -s -v -k cdf            # plot relative-error CDF
uv run pytest tasks/ -s -v -k train_full     # paper-scale training run (hours)
```

`tasks/test_pipeline.py` uses pytest as a task runner, not a test suite: each `test_NN_*` function is a numbered, idempotent pipeline step (skips work that's already done — e.g. won't re-download an existing archive). `-s` is required to see progress output. Steps must generally run in order since later steps assert on artifacts from earlier ones (e.g. `test_03_train` requires `test_02_extract`'s tfrecords).

## Architecture

### Core model (`routenet/routenet.py`)

- `HParams` — dataclass of model hyperparameters (`link_state_dim`, `path_state_dim`, `T` message-passing rounds, `readout_units`, etc). `_parse_hparams(hparams, "k=v,k=v")` mutates one from a CLI-style string.
- `RouteNet(tf.keras.Model)` — the message-passing GNN. Built manually in `build()` (not inferred from `input_shape`) because the model operates on flattened/batched graph features, not a fixed tensor shape:
  - `edge_update`: a `GRUCell` that updates per-link hidden state from aggregated path messages.
  - `path_rnn`: a `keras.layers.RNN(GRUCell)` that runs per-path over its sequence of links (replaces the old `tf.nn.dynamic_rnn`).
  - `call()` runs `hparams.T` rounds of: gather link states onto path sequences via `tf.scatter_nd` → run `path_rnn` (masked by `tf.sequence_mask` for variable path length) → scatter messages back to links via `tf.math.unsorted_segment_sum` → update link state with `edge_update`. Finally projects path state through a small dense `readout` stack to `output_units`.
  - `get_config`/`from_config` exist because `hparams` (a dataclass) isn't natively JSON-serializable, which Keras requires for `.keras` checkpoint saving — see `local/KNOWN_ISSUES.md` for the exact error this avoids.
- `DelayRouteNet(RouteNet)` / `DropRouteNet(RouteNet)` — task-specific subclasses overriding `train_step`/`test_step` (this codebase does NOT use `model.compile(loss=...)`; loss is computed manually inside these methods):
  - `DelayRouteNet` predicts `[loc, scale]` for delay and jitter and minimizes `_heteroscedastic_loss` (a custom Gaussian NLL weighted by packet count).
  - `DropRouteNet` predicts drop logits and minimizes `_binomial_loss` (packet-weighted sigmoid cross-entropy).
- `PearsonCorrelation` / `MeanRelativeError` — custom `tf.keras.metrics.Metric` subclasses (replace TF1's `tf.contrib.metrics.streaming_pearson_correlation`, which has no TF2 equivalent).
- Data pipeline: `parse()` decodes one `tf.train.Example` from a TFRecord (sparse `VarLenFeature`s densified, then optionally normalized via `scale_fn`). `tfrecord_input_fn()` builds the `tf.data.Dataset`: interleave files → shuffle+repeat (train) or 10%-sample filter (eval, matches the original paper's eval sampling) → `parse` → `ragged_batch` → `_collate_graphs`.
- `_collate_graphs()` is the key trick for batching graphs of different sizes: it flattens a ragged batch of `B` separate graphs into one big graph, offsetting each graph's `links`/`paths` indices by the cumulative sum of previous graphs' `n_links`/`n_paths`, so a batch is processed as a single concatenated graph rather than padded/looped per-sample.
- `scale_fn` — normalizes `traffic` (mean/std) and `capacities` (divide by 10); must be applied identically at train, eval, and serving time (see the `@tf.function`-decorated `serve()` closure inside `train()`, which re-applies it before inference).

### Pipeline orchestration (`tasks/test_pipeline.py`)

Defines two parallel hyperparameter/scale tracks used across all steps:
- `HPARAMS`/`MODEL_DIR` (`models/delay/`) — small smoke-test config (~20 epochs, ~4k steps total).
- `HPARAMS_FULL`/`MODEL_DIR_FULL` (`models/delay_full/`) — paper-scale config (~300k steps, `EarlyStopping`), intentionally written to a separate directory so it never overwrites the smoke checkpoint.

Both tracks share the same step shape: train → evaluate (MAE/MRE/Pearson ρ → `eval_results.json`) → CDF plot (`cdf_relative_error.png`). There's also cross-topology evaluation (steps 10-13): a model trained on NSFNet (14 nodes) is evaluated against the GBN dataset (17 nodes) to test generalization across network topologies, writing `eval_results_gbn.json`.

`data/` and `models/` are gitignored — datasets are downloaded fresh via `urllib.request` from the KDN dataset host, and checkpoints/eval results are regenerated locally, not committed.

### Known environment gotchas (Apple Silicon / Metal GPU)

These are load-bearing constraints, not stylistic choices — see `local/KNOWN_ISSUES.md` for full repro/error text:
- `model.compile(..., run_eagerly=True)` is required for training (`DelayRouteNet`/`DropRouteNet` use `RNN`/GRUCell internally); without it, `tf-metal`'s graph compilation of the RNN backward pass crashes with `InternalError: stream cannot wait for itself`.
- TF must stay at `2.17.0` to match `tensorflow-metal==1.2.0`'s ABI.
- `ModelCheckpoint` paths must end in `.keras` (Keras 3 dropped the SavedModel-dir format).
- When loading a saved checkpoint, pass `custom_objects={"DelayRouteNet": DelayRouteNet, "HParams": HParams}` to `tf.keras.models.load_model` (see every `test_04_evaluate`-style step in `tasks/test_pipeline.py` for the pattern).

### `routenet/upcdataset.py`

Converts raw UPC/OMNeT++ simulation output (`.ned` topology + `Routing.txt` + `delayGlobal.txt` inside per-sample `.tar.gz` archives) into the `.tfrecords` format consumed by `tfrecord_input_fn`. Only used to build datasets from scratch; the NSFNet/GBN datasets used in `tasks/test_pipeline.py` are pre-built and downloaded directly as tfrecords, so this script is rarely needed in normal workflows.
