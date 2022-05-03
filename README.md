# Binaural WSJ0Mix dataset

Clean binaural separation dataset for two or three speakers in [Real-time binaural speech separation with preserved spatial cues](https://ieeexplore.ieee.org/abstract/document/9053215). Briefly, we randomly sampled two or three speaker locations in the HRTF database from the CIPIC, convolved with randomly sampled two or three utterances from the wsj0 and mixed them all.
## Requirements

These scripts require the Numpy, Scipy, Pandas, and Pysoundfile packages.

## Prerequisites

The original [wsj0](https://catalog.ldc.upenn.edu/LDC93S6A/) dataset and the [CIPIC HRTF Database](https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/). The CIPIC HRTF Database contains real recorded HRIR filters across 25 different interaural-polar azimuths from −80◦ to 80◦ and 50 different interaural-polar elevations from −90◦ to 270◦.  We separated subjects into 27 for training, 9 for validation and 9 for test set, ensuring that the model is evaluated in a listener-independent way. Here we used the [WAV version](https://ucdavis.app.box.com/s/046degs88ultkgevrud54cvvtq0ufz3p) of CIPIC.


## Usage

```sh
$ python create_wav_2speakers.py
    --wsj0-root  /path/to/wsj/wsj0/
    --hrtf-root /path/to/hrtf/wav_database/
    --output-dir /path/to/the/output/directory/
```
The arguments for the script are:
* **wsj0-root**:  Path to the folder containing `wsj0/`
* **hrtf-root**: Folder where the unzipped `wav_database` was downloaded.
* **output-dir**: Where to write the new dataset.