# merlin edge-case audit

this folder tests where Merlin's default CT image-text model behaves reliably and where it needs more caution.

the audit uses the official sample CT from the Merlin Hugging Face repository. it does not contain patient identifiers, and none of the results should be interpreted as clinical validation.

## what is tested

the text cases cover matched findings, negated findings, incorrect laterality, contradictions, unrelated clinical text, instruction-like text, an empty report, and a report long enough to trigger tokenizer truncation.

the image cases cover left-right flipping, reversed slice order, inverted intensities, an all-zero scan, a corrupt NIfTI file, and an undersized volume that must be padded.

## run it

download the official sample CT:

```python
from merlin.data import download_sample_data

download_sample_data("edge_cases/data")
```

then run:

```bash
python edge_cases/run_edge_cases.py
```

the detailed measurements are written to `edge_cases/results.json`. that file is ignored by git so results from different machines do not get mixed together.

the recorded CPU run is documented in [RESULTS.md](RESULTS.md).

## scope

this machine has a CPU-only PyTorch installation. the audit therefore covers the already-downloaded default Merlin checkpoint, including image-text embeddings and phenotype outputs.

the radiology report generator was not evaluated. its checkpoint is roughly 25 GB, and the Merlin documentation says it was tested on a 48 GB NVIDIA A6000 GPU. claiming report-generation results from this machine would not be honest.

## how to read the results

passing means the behavior matched the expectation written into the harness. failing does not automatically mean the model is unusable. it identifies a condition that deserves a closer test, a guardrail, or a more clinically grounded benchmark.

the numeric thresholds are exploratory and should not be treated as published performance claims.
