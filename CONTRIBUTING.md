# Contributing

Thanks for helping improve CrossModalMedNet.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r project-group-5/requirements.txt
pip install -e ".[dev,metrics]"
```

## Pull Request Checklist

- Keep raw medical data, checkpoints, logs, and virtual environments out of git.
- Add or update documentation for new training or evaluation commands.
- Include the manifest path, split file, checkpoint name, and hardware details for new reported results.
- Run a relevant smoke test before submitting:

```bash
cd project-group-5
python scripts/verify_checkpoints.py
```

## Reproducibility

Research changes should make it clear which cohort, split, model family, checkpoint, image resolution, and metric implementation were used. Prefer small, reviewable changes over broad rewrites.
