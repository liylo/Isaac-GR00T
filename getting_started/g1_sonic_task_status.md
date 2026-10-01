# G1 Sonic task progress for `walk_and_place`

This status head adds two inference values without changing the pretrained action policy:

- `progress`: normalized task progress. It increases toward the terminal hold window, then
  remains at `1.0` for that window.
- `completion`: probability that the current observation is in the terminal hold window.

The implementation uses the G1 Sonic tag `UNITREE_G1_SONIC` (`unitree_g1_sonic`) and the dataset
at `/data/liuyilong/dataset/walk_and_place` by default.

## Label choice

The dataset is recorded at 50 Hz. Its final two seconds often still contain placement or hand
retraction motion, while the final 0.5 seconds are substantially more stationary. The default
completion window is therefore the final 25 frames:

```text
terminal_start = episode_length - 25
completion = frame_index >= terminal_start
progress = min(frame_index / max(terminal_start, 1), 1)
```

This produces 2,600 positive frames out of 114,371. The launcher computes the natural-distribution
BCE positive weight from `meta/episodes.jsonl` (`42.9888` for this dataset), so it does not reuse
the unrelated value from the original 30 Hz task.

## Train only the status head

Start from the checkpoint that already performs `walk_and_place`, not from an untrained/base action
policy:

```bash
bash examples/train_unitree_g1_sonic_task_status.sh \
  /path/to/walk_and_place/checkpoint \
  outputs/walk_and_place_task_status
```

Useful environment overrides:

```bash
MAX_STEPS=3000 GLOBAL_BATCH_SIZE=64 TASK_STATUS_WINDOW_FRAMES=10 \
  bash examples/train_unitree_g1_sonic_task_status.sh <checkpoint> <output-dir>
```

The launcher freezes the VLM backbone and action head, trains only the approximately 18M-parameter
attention-pooling status head, and enables terminal padding so the final 39 frames omitted by the
normal 40-frame action horizon are available for status supervision.

## Inference

`Gr00tPolicy.get_action()` keeps its existing action return and exposes status values in `info`:

```python
action, info = policy.get_action(observation)
print(float(info["progress"][0]), float(info["completion"][0]), bool(info["done"][0]))
```

`done` uses thresholds saved in the model config. The dataset-specific launcher defaults to one
positive policy call because the 0.5-second completion window is shorter than a 40-frame action
chunk (0.8 seconds). If status inference runs independently at a higher cadence, increase
`TASK_STATUS_CONSECUTIVE_STEPS` and calibrate the threshold on held-out episodes.

## Eight-GPU `walk_and_place_ep10` run

The exact eight-GPU setup based on `output_walk/checkpoint-20000` is available as:

```bash
bash examples/train_unitree_g1_sonic_task_status_ep10.sh
```

It is equivalent to the requested `torchrun --nproc_per_node=8 --master_port=12345` launch and
keeps `max_steps=50000`. Environment variables can override the important paths or duration:

```bash
MAX_STEPS=3000 OUTPUT_DIR=output_walk_status_ep10 \
  bash examples/train_unitree_g1_sonic_task_status_ep10.sh
```

For this 10-episode subset, the launcher derives `completion_pos_weight=71.972` from its metadata.

## Task-status open-loop plot

The status plot runs the VLM backbone and task-status head only; it does not run action diffusion:

```bash
python gr00t/eval/plot_task_status_open_loop.py \
  --model-path output_walk_dual2_ep10/checkpoint-50000 \
  --dataset-path /data/liuyilong/dataset/walk_and_place_ep10 \
  --embodiment-tag UNITREE_G1_SONIC \
  --episode-ids 0 \
  --frame-stride 10 \
  --tail-dense-frames 50 \
  --batch-size 4 \
  --device cuda:0 \
  --output-dir output_walk_dual2_ep10/open_loop_task_status
```

Each episode produces a PNG curve, a frame-level CSV, and aggregate metadata in `summary.json`.
