# CHIME: A Constrained Harmonic Inductive Bias for Melody Extraction

CHIME is a **C**onstrained **H**armonic **I**nductive bias towards exploiting harmonic patterns for **M**elody **E**xtraction. CHIME consists of two modules designed for pitch and voicing, where the predicted pitch distribution is provided as an additional input to the voicing module to aid in voicing estimation.
<img width="1009" height="460" alt="CHIMEPerformance" src="https://github.com/user-attachments/assets/d51abe55-8383-4faf-a8c2-6257ce10bb51" />

CHIME achieves state-of-the-art results across all test datasets, while also currently being the fastest melody extration method (up to 106.6x faster than previous state-of-the-art methods), indicating the value in researching melody extraction methods that focus on specific characteristics of music.


## Jupyter Notebook and Code

We provide a Jupyter Notebook that allows for CHIME to be easily trained and evaluated. This notebook also contains a section
for conveniently using CHIME to generate a predicted melody contour, and its corresponding synthesized audio, for any song, provided that a file path is given.

If you do not wish to use the Jupyter Notebook, the source code can still be found in this GitHub repository, containing all comments and information necessary to reproduce
the results reported in the paper and to use CHIME to generated a sequence of melody pitches. 

To retrain CHIME, use the corresponding python files listed in this GitHub to process the data to use in training. To reproduce the figures and tables, use the provided notebook, as
it has all necessary information.

## CHIME Checkpoints
We provide two model checkpoints: PaperCHIME (which was used to obtain the results detailed in the paper) and UltimateCHIME, which is a version of CHIME trained on all validation and
testing datasets. 

For reproducing the results in the paper, use CHIME. If you intend to use this model to generate melodic contours for personal use, we recommend CHIME-Ultimate, although CHIME 
still achieves great performance

