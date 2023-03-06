# Usage

[日本語](../ja/setup.md) | English

## Prerequisites

- [Python](https://www.python.org/) 3.7
- [pip](https://pip.pypa.io/)
- (on Linux) libsndfile1, which is required by [soundfile](https://github.com/bastibe/python-soundfile)
- [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset (or [MUSDB18](https://zenodo.org/record/1117372) dataset)

## Installation

1. Clone the remote repository to your local environment with ```git clone```.

```bash
$ git clone https://github.com/torakyun/music_source_separation.git
````

2. Install the packages from ```requirements.txt``` with ```pip```.

```bash
$ cd mss
$ pip install -r requirements.txt
```

3. Install [PyTorch](https://pytorch.org/) 1.10.0 (see link for installation instructions).
   - The version that matches your GPU (CUDA) environment

## Directory structure

```sh .
. /
   conf/  # Hyperparameter management
      dataset/  # Dataset
      device/  # Device
      loss/  # Loss
      model/  # Model
         generator/  # Generator
         discriminator/  # Discriminator
      optimizer/  # Optimizer
         generator/  # Generator
         discriminator/  # Discriminator
      config.yaml  # Experimental configuration
   docs/  # Documents
      en/  # English version
         README.md  # README
         setup.md  # Setup manual
      ja/  # Japanese version
         setup.md  # Setup manual
   mss/  # Implementation of music source separation
      layers/  # Implementation of the neural network layers
         __init__.py  # Top of the module
         pqmf.py  # Pseudo-Quadrature Mirror Filters (PQMF)
      losses/  # Implementation of the loss functions
         __init__.py  # Top of the module
         adversarial_loss.py  # Adversarial loss
         cac_loss.py  # CAC loss
         feat_match_loss.py  # Feature matching loss
         mel_loss.py  # Mel spectrogram loss
         mfcc_loss.py  # MFCC loss
         stft_loss.py  # STFT loss
      models/  # Implementation of the models
         __init__.py  # Top of the module
         demucs_v2.py  # Demucs (v2)
         demucs.py  # Demucs
         hdemucs.py  # Hybrid Demucs
         hifigan.py  # HiFi-GAN discriminator
         melgan.py  # MelGAN discriminator
         parallel_wavegan.py  # Parallel WaveGAN discriminator
         hifigan.py  # StyleMelGAN discriminator
         tasnet.py  # Conv-Tasnet
      optimizers/  # Implementation of the optimizers
         __init__.py  # Top of the module
         radam.py  # RADAM
   (musdb18hq/)  # MUSDB18-HQ dataset
   (out/)  # Outputs
      (checkpoints/)  # Checkpoints of the training progress
      (valids/)  # Validation results per epoch
      (evals/)  # Evaluation results per epoch
      (mlruns/)  # MLflow logs
      (models/)  # Trained models
   .gitignore  # Configuration for ignoration of the git management
   LICENSE  # LICENSE
   make_mos_set.py  # Making evaluation set for the subjective metric MOS
   mos.py  # Tabulation of the subjective metric MOS results
   README.md  # README (Japanese version)
   requirements.txt  # Python package requirements (pip)
   run.py  # Performing multi-GPU learning
   sdr.py  # Tabulation of the objective metric SDR results
   separate.py  # Prediction of the separate sounds with the trained model
````

## Command examples

### Model training (single GPU).

1. (Recommended) To specify hyperparameters that don\'t exist in `config.yaml`, duplicate the corresponding file in the `conf` directory.

2. Specify hyperparameters from the command line. (e.g. sampling rate: 2.4 kHz, generator: `hdemucs`)
   ```bash
   $ cd mss
   $ python3 -m mss [-dataset.samplerate=24000] [-model/generator=hdemucs]
   ````

3. Start MLflow GUI and monitor the logs.
   ```bash
   $ cd mlruns/...
   $ mlflow ui
   ````

4. (Recommended) If you set ```valid_interval```, the separated sounds obtained from the verification will be stored in the `valids` directory at each interval.

- (Recommended) When there are multiple GPUs, single GPU training should be performed in parallel under different conditions.
- (Recommended) Lighten the weight of the data and model in order to train efficiently in terms of time or computational resources.
- (Note) When GPUs do not have enough memory, use ```batch_divide``` to split mini-batch and train little by little.

### Learning of models (multi-GPU)

Change from running the `mss` module to running the `run.py` file.
```bash
$ cd mss
$ python3 run.py [-dataset.samplerate=24000] [-model/generator=hdemucs]
```

### Evaluate trained models with subjective metric MOS

1. Make evaluation sets consisting of 20 data for one person.
   ```bash
   $ cd mss
   $ python3 make_mos_set.py -n [multiple model names] -r [number of evaluation sets]
   ```
   The command will create the following steps.
   - Randomly select 8 seconds per song from `test` in MUSDB18. (the interval where all sound sources are present)
   - Randomly extract 20 pieces and randomly assign sound sources and models (+ correct answers).

2. Edit `HumanEvaluation.ipynb`.
   - Set up the evaluation items.
   - Edit the id to the evaluation set placed in Google Drive.

3. Open `HumanEvaluation.ipynb` link to the Colaboratory to the subjects and ask them to evaluate it on their browsers.
   - The evaluation results will be downloaded as a csv file.

4. Tally up the evaluation results.
   ```bash
   $ cd mss
   $ python3 mos.py -f [path to the directory containing the csv file]
   ```

### Evaluate learned model with objective metric SDR.

```bash
$ cd mss
$ python3 sdr.py -n [multiple model names]
```

- (Note) If memory is insufficient, evaluate models one at a time.

### Run music source separation with trained models for any song.

```bash
$ cd mss
$ python3 separate.py [multiple songs] -n [model name]
```

- By default, the file is written out as a wav file.  
You can change it to mp3 file with option ```--mp3```.
