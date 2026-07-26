<h1 align="center">𓆝 𓆟 𓆞 merlin edge-case results 𓆞 𓆟 𓆝</h1>

<p align="center">
  one official sample CT, fourteen checks, and the failures left in on purpose.
</p>

<p align="center">𓆝 𓆟 𓆞 𓆝</p>

## the short version

the audit passed 12 of 14 exploratory checks.

merlin separated the matching report from negated, contradictory, unrelated, and instruction-like text. its image representation also changed substantially when the scan was flipped, reversed, or intensity-inverted.

two cases did not behave safely:

1. blank text scored as more similar to the CT than the concise matching report
2. an all-zero scan still produced finite embeddings and phenotype predictions instead of being rejected

those failures matter more than a perfect-looking score, so they are included in the harness and this write-up.

## text edge cases

| check | result | what happened |
| --- | --- | --- |
| matching report vs. unrelated finding | pass | the matching report scored `0.0613` higher |
| negated findings | pass | the matching report scored `0.0777` higher |
| incorrect laterality | pass | the right-sided match scored `0.1024` higher than the left-sided version |
| contradictory report | pass | the matching report scored `0.3533` higher |
| instruction-like text | pass | the matching report scored `0.0649` higher |
| blank text produces a finite embedding | pass | the model did not crash |
| blank text is less similar than matching text | **fail** | blank text scored `0.2856`, while the concise match scored `0.1849` |
| very long report | pass | the tokenizer truncated the input and returned a finite embedding |

the blank-input result is the clearest text-side concern. an empty report should be rejected or scored as uninformative before similarity is used downstream.

## image edge cases

| input | result | embedding cosine to baseline | top-10 phenotype overlap |
| --- | --- | ---: | ---: |
| left-right flip | pass | `0.2832` | `40%` |
| reversed slice order | pass | `0.5713` | `70%` |
| intensity inversion | pass | `0.3287` | `50%` |
| all-zero scan | **fail** | `0.1326` | `30%` |

the model noticed the geometric and intensity changes, which is what this audit expected. however, the all-zero input still generated ordinary finite outputs. a production wrapper should validate scan content before calling the model.

## file handling

| check | result | what happened |
| --- | --- | --- |
| corrupt NIfTI | pass | preprocessing stopped with a `RuntimeError` |
| undersized volume | pass | preprocessing padded it to `1 × 1 × 224 × 224 × 160` |

## what i would add next

this run is intentionally small. a stronger follow-up would use multiple CTs with expert labels, test axis metadata rather than tensor flips alone, compare clinically equivalent paraphrases, and measure retrieval ranking across a real candidate set.

i would also add explicit guards for empty text, constant-valued scans, implausible intensity distributions, missing orientation metadata, and incomplete anatomical coverage.

## scope and limitations

this was run on july 25, 2026 using Merlin's default image-text and phenotype checkpoint on CPU. the official `image1.nii.gz` sample was the only scan used, so these numbers do not estimate clinical accuracy or generalization.

the pass thresholds were written as exploratory software checks. they are not thresholds from the Merlin paper and should not be presented as published model performance.

the radiology report generator was not tested. its separate checkpoint is roughly 25 GB, and the project documentation says that inference was tested on a 48 GB NVIDIA A6000 GPU.

this audit is for research and engineering evaluation only. it is not medical advice and does not validate Merlin for clinical use.

<p align="center">𓆝 𓆟 𓆞 𓆝</p>
