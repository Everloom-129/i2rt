# openpi / pi0 Training (JAX)

> **Status:** Dataset conversion is ready. Full JAX training setup is pending.

This directory contains tools to prepare i2rt grasp data for training with
[openpi](https://github.com/Physical-Intelligence/openpi) (the open-source
release of the π0 foundation model for robot manipulation).

---

## Dataset Conversion

Convert a LeRobot v3 dataset to the RLDS/TFRecord format expected by openpi:

```bash
python grasp_lab/openpi_training/convert_dataset.py \
    --dataset-dir ./data/grasp_mug \
    --output-dir ./data/grasp_mug_rlds

# Dry-run (prints shapes, no writing)
python grasp_lab/openpi_training/convert_dataset.py \
    --dataset-dir ./data/grasp_mug \
    --output-dir /tmp/test_rlds \
    --dry-run
```

Feature mapping:

| LeRobot v3 key                  | openpi / RLDS key           |
|---------------------------------|-----------------------------|
| `observation.state`             | `observation/state`         |
| `action`                        | `action`                    |
| `observation.images.wrist`      | `observation/image`         |
| `observation.images.top`        | `observation/image_top`     |

---

## Full openpi Setup (TODO)

1. **Install JAX + openpi**

   ```bash
   # JAX with CUDA
   pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # openpi
   pip install git+https://github.com/Physical-Intelligence/openpi.git
   ```

2. **Download a pretrained π0 checkpoint** from the openpi model hub.

3. **Configure fine-tuning** — create an openpi config that points to the
   converted RLDS dataset and the pretrained checkpoint.

4. **Launch training**

   ```bash
   python -m openpi.training.train \
       --config my_task.py \
       --workdir ./runs/grasp_mug_pi0
   ```

5. **Evaluate** using the openpi rollout utilities or export the policy
   back to a LeRobot-compatible format.

---

## References

- [openpi repository](https://github.com/Physical-Intelligence/openpi)
- [π0 paper](https://arxiv.org/abs/2410.24164)
- [RLDS data format](https://github.com/google-research/rlds)
