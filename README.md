# News
Now you can train models of this dataset with speechbrain [here](https://github.com/speechbrain/speechbrain/tree/develop/recipes/BinauralWSJ0Mix/separation).

## Binaural WSJ0Mix dataset

Binaural separation dataset for two or three speakers in [Real-time binaural speech separation with preserved spatial cues](https://ieeexplore.ieee.org/abstract/document/9053215). Briefly, we randomly sampled 2 or 3 speaker locations in the HRTF database from the CIPIC, convolved with randomly sampled two or three utterances from the wsj0 and mixed them all. Also, we created 2 speakers mixture with DEMAND noise or simulated BRIR reverberance.

## Requirements

These scripts require the Numpy, Scipy, Pandas packages.

## Prerequisites

The original [wsj0](https://catalog.ldc.upenn.edu/LDC93S6A/) dataset and the [CIPIC HRTF Database](https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/). The CIPIC HRTF Database contains real recorded HRIR filters across 25 different interaural-polar azimuths from −80◦ to 80◦ and 50 different interaural-polar elevations from −90◦ to 270◦.  We separated subjects into 27 for training, 9 for validation and 9 for test set, ensuring that the model is evaluated in a listener-independent way. Here we used the WAV version of CIPIC.


## Usage

```sh
$ python create_wav_2speakers.py
    --wsj0-root  /path/to/wsj/wsj0/
    --output-dir /path/to/the/output/directory/
```
The arguments for the script are:
* **wsj0-root**:  Path to the folder containing `wsj0/`
* **output-dir**: Where to write the new dataset.

```sh
$ python create_wav_2speakers_noise.py
    --wsj0-root  /path/to/wsj/wsj0/
    --output-dir /path/to/the/output/directory/
```
The arguments for the script are:
* **wsj0-root**:  Path to the folder containing `wsj0/`
* **output-dir**: Where to write the new dataset. It will download [DEMAND dataset](https://deepai.org/dataset/demand) automatically.

```sh
$ python create_wav_2speakers_reverb.py
    --wsj0-root  /path/to/wsj/wsj0/
    --output-dir /path/to/the/output/directory/
```
The arguments for the script are:
* **wsj0-root**:  Path to the folder containing `wsj0/`
* **output-dir**: Where to write the new dataset. It will download [Simulated Room Impulse Responses](https://iosr.uk/software/index.php) automatically.
