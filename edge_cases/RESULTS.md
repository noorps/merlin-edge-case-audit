<h1 align="center">𓆝 𓆟 𓆞 what happened when i tested merlin 𓆞 𓆟 𓆝</h1>

<p align="center">
  one sample CT, fourteen slightly weird tests, and two results i did not expect.
</p>

<p align="center">𓆝 𓆟 𓆞 𓆝</p>

## the quick version

12 of my 14 checks worked the way i expected.

merlin could tell the difference between the matching report and versions with negated findings, the wrong side of the body, contradictions, unrelated medical text, or a random instruction.

it also reacted when i flipped the scan, reversed the slices, or inverted the image. that was good because those versions should not look identical to the model.

the two weird results were:

1. blank text matched the CT more closely than my short, correct description did
2. a scan made entirely of zeros still produced regular numbers instead of being rejected

i kept both failures in here because they were the most interesting part of the experiment.

## the text tests

| what i tried | result | what happened |
| --- | --- | --- |
| matching report vs. an unrelated injury | pass | the matching text scored `0.0613` higher |
| turning the findings into negatives | pass | the matching text scored `0.0777` higher |
| changing right to left | pass | the correct side scored `0.1024` higher |
| putting contradictions in the report | pass | the matching text scored `0.3533` higher |
| adding a fake instruction | pass | the matching text scored `0.0649` higher |
| sending blank text without crashing | pass | the model returned an embedding |
| making sure blank text scored lower | **fail** | blank text scored `0.2856`, while my short match scored `0.1849` |
| sending a very long report | pass | the text was shortened by the tokenizer and still ran |

the blank-text result surprised me the most. if i were building an app with Merlin, i would block empty text before it ever reached the model.

## the scan tests

| what i changed | result | similarity to the original | shared top predictions |
| --- | --- | ---: | ---: |
| flipped it left to right | pass | `0.2832` | `40%` |
| reversed the slice order | pass | `0.5713` | `70%` |
| inverted the intensities | pass | `0.3287` | `50%` |
| replaced everything with zeros | **fail** | `0.1326` | `30%` |

the first three changes clearly affected what Merlin produced. the zero scan also looked different to the model, but it still returned normal-looking outputs. i expected it to reject an input with no actual scan information.

## broken and tiny files

| what i tried | result | what happened |
| --- | --- | --- |
| a fake, corrupted NIfTI file | pass | preprocessing stopped with a `RuntimeError` |
| a very small volume | pass | preprocessing padded it to `1 × 1 × 224 × 224 × 160` |

## what i would try next

this was only one sample scan, so the biggest next step would be repeating everything across more CTs.

i would also test more ways of saying the same finding, scans with missing orientation information, partially cropped anatomy, and several kinds of empty or broken image data. it would be useful to see whether the blank-text issue happens consistently or was specific to this one scan.

## important context

i ran this on july 25, 2026 using Merlin's default checkpoint and a CPU-only computer. the sample was the official `image1.nii.gz` file.

the pass and fail cutoffs are expectations i wrote for this experiment. they are not official Merlin benchmarks, and these results do not measure whether the model is clinically accurate.

i could not test the separate report-generation checkpoint because it is around 25 GB and the project documentation describes running it on a 48 GB NVIDIA A6000 GPU.

this is just a software experiment for learning about model edge cases. it is not medical advice and should not be used to diagnose anything.

<p align="center">𓆝 𓆟 𓆞 𓆝</p>
