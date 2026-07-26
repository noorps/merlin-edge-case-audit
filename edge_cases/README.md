# trying weird inputs with merlin

i wanted to get Merlin running and then see what happened when i gave it inputs that were not perfectly clean.

i used the official sample CT from Merlin's Hugging Face page. this is a small software experiment with one example scan, not a medical study or anything that should be used for diagnosis.

## what i tried

for the text side, i tried a matching report, the same findings written as negatives, the wrong side of the body, statements that contradict each other, completely unrelated text, a fake instruction, a blank string, and one very long report.

for the scan side, i flipped the image left to right, reversed its slice order, inverted the intensities, replaced the whole scan with zeros, broke a NIfTI file on purpose, and made a tiny volume that the preprocessing code had to pad.

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

the script writes all of the raw measurements to `edge_cases/results.json`. that file stays local so results from different computers do not get mixed together.

my recorded run and notes are in [RESULTS.md](RESULTS.md).

## what i could actually run

my computer is running the CPU-only version of PyTorch. i tested the default Merlin checkpoint that was already downloaded, including its image-text matching and phenotype outputs.

i did not test the radiology report generator. that checkpoint is around 25 GB, and the Merlin docs say it was tested on a 48 GB NVIDIA A6000 GPU. my computer does not have that setup, so i did not want to pretend i tested it.

## what pass and fail mean here

a pass just means the result matched the expectation i wrote into the script. a fail does not mean Merlin is a bad model. it means the result surprised me or would need an extra input check before someone built on top of it.

the cutoffs are ones i chose for this experiment. they are not numbers from the Merlin paper or official performance claims.
