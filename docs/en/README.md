# Verification of music source separation based on generative optimization

[日本語](../setup.md) | English

## Overview and current status of music source separation
Music source separation refers to a technique for decomposing a musical piece, once synthesized, into individual musical notes.
By enabling individual manipulation of musical tones, it is possible to
This leads to the realization of advanced functions on music distribution services.
In fact, music source separation can be used to create remix and karaoke sound sources, to search for music information, and to provide music education.
In fact, music source separation has been attempted to be applied to functions such as remixing, karaoke sound source creation, music information retrieval, and music education.
However, with current technology, problems such as omission of separation from other sound sources and loss of the target sound source itself can be completely eliminated.
However, current technology has not been able to completely eliminate problems such as omission of separation from other sound sources or loss of the target sound source itself.
In addition, noise may be generated in some of the sounds.
These factors reduce the subjective evaluation.

<img width="800" src="https://user-images.githubusercontent.com/67317828/217169110-5bfc0204-bf78-4a7c-9a9f-8bb44e5d6c31.gif">
  
## Music source separation based on generative optimization
Music source separation has been intensively studied over the past decade, and has made a quantum leap with the advent of deep learning.
The advent of deep learning has led to dramatic advances.
Musical source separation based on deep learning has evolved through two main approaches.
Masking approaches in the frequency domain and
the End-to-End approach in the time domain.
In the latter approach
direct reconstruction from the extracted features to the temporal waveform.
Such structures have been commonly used in the field of speech synthesis.
However, while models incorporating generative structures have been proposed
optimization for these models differs from that in the speech synthesis field.
In conventional optimization, waveform loss has been minimized over the time domain.
The waveform loss is obtained by taking the difference from the teacher at each time of the time waveform.
In other words, a single signal at the same time is compared.
In contrast, optimization in the speech synthesis field considers the time variation of the serial signal.
Minimizing STFT loss allows learning the frequency characteristics of sound.
The human auditory system also performs frequency analysis, and this method is based on human hearing.
In addition, minimization of adversarial loss can be used to learn the authenticity of sound.
Adversarial loss minimization is based on maximizing the probability that the output series is a real sound, and
enable the generation of authenticity.
Therefore, we propose a musical source separation based on generative optimization.
In addition to waveform loss, STFT loss and adversarial loss are minimized simultaneously.
By better learning not only the proximity as waveforms, but also the difference in timbre, the
separation accuracy and subjective evaluation of sound pollution can be improved.

<img width="800" alt="architecture3" src="https://user-images.githubusercontent.com/67317828/217165468-09999921-727f-4427-999d-d29c7e9862b8.png">

## Validation results for generative optimization

**Dataset - [MUSDB-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)**

The largest dataset for music source separation.
Data extensions were applied and used.

**Generator - [Hybrid Demucs](https://arxiv.org/pdf/2111.03600.pdf)**

The End-to-End model Demucs in the time domain was extended to the hybrid domain.
extended to the hybrid domain.

**Discriminator - [StyleMelGAN](https://arxiv.org/pdf/2011.01557.pdf) discriminator**

Lightweight and high quality generation.

**Wave loss - Mean Absolute Error (MAE)**

Loss function used in Hybrid Demucs.

**STFT loss - Spectrogram difference**.

Calculated by ```(Linear Scale + Power Scale) x 3 Resolutions```.
Please refer to [Parallel WaveGAN](https://arxiv.org/pdf/1910.11480.pdf).

**Adversarial Loss - Squared Error**

Please refer to [Least Squares GAN](https://arxiv.org/pdf/1611.04076.pdf).

**Subjective Index - Mean Opinion Score (MOS)**.

The percentage of votes for each category, weighted by the grade (38 subjects in this study).

<img width="600" alt="mos_item" src="https://user-images.githubusercontent.com/67317828/221777752-251a920e-d3b4-49a0-ac2d-cbf9872f522f.png">

**Objective Index - [Source-to-Distortion Ratio (SDR)](https://hal.inria.fr/inria-00544230/document)**

Measures the degree of distortion from the correct solution by comparing signals.
It has been commonly used for easy and uniform evaluation, but it does not correlate well with human hearing.
However, it has been reported that it does not correlate well with human hearing, and its validity is questionable.
The validity of this method is questionable.

（Optimization with the addition of STFT loss (at multiple scales and resolutions) has been shown to improve
The optimization with the addition of STFT loss (at multiple scales and resolutions) was found to improve the overall separation accuracy and subjective assessment of sound pollution.
In fact, by drawing a log mel spectrogram
the contribution of STFT loss could be seen.
Although only the base sound quality contamination was slightly reduced
We believe that this is due to the removal of high-frequency components that should not have been present.
High-frequency components are components that form the detailed shape of sound waves.
The removal of the components that were making the waveforms add up
The elimination of the components that were used to make the waveforms add up may have slightly disturbed the consistency of the waveforms and degraded the sound quality.
Further optimization with the addition of hostile loss resulted in a decrease in the separation accuracy.
The optimization with additional hostile loss resulted in lower separation accuracy, but improved the sound quality contamination of the base.
The reason for the reduced separation accuracy could be attributed to the fact that the discriminator that calculates the
The reason for the reduced separation accuracy may be that the discriminator that calculates the hostile loss has overlearned.
Possible countermeasures include making the discriminator lighter and introducing techniques to suppress over-learning.
Eliminating instability in learning adversarial losses is a future challenge.

<img width="800" alt="mos" src="https://user-images.githubusercontent.com/67317828/221762798-6f628f2f-f2c5-4d20-b70d-242bf8e42203.png ">
<img width="800" alt="sdr" src="https://user-images.githubusercontent.com/67317828/221762357-ec2ab102-bb3c-460b-8165-cc3dde6eb93a.png ">

Since music is created to be listened to by humans, the
It is important to improve the subjective evaluation of music source separation systems.
The generative optimization-based music source separation implemented in this study
The subjective evaluation results of the music source separation based on generative optimization implemented in this study exceeded those of conventional methods in many aspects.
It is hoped that the optimization will be reviewed in the future and lead to further quality improvement.
However, while subjective evaluation is necessary
In practice, it is common that only objective evaluation is performed.
In the future, it will be necessary to establish clear criteria for subjective evaluation and to create a platform for subjective evaluation.
platform for subjective evaluation, etc.
It is also necessary to set up a system that allows for the active testing of subjective evaluation by establishing clear criteria for subjective evaluation, creating a platform, and so on.

# Usage

Please refer to [here](docs/setup.md).

# License

Please refer to [here](LICENSE).
This system was created by duplicating [Implementation of Hybrid Demucs](https://github.com/torakyun/music_source_separation/tree/64ed2cb029301743b2714b3c8fe930c00945842c).
We are very grateful for the great contribution to music source separation.
We would like to thank the following people for their significant contributions to music source separation.
The introduction of the ``Trainer`` class has allowed us to create a large system of iterative learning, validation, and evaluation.
Although we have made significant changes to the general framework of the system through repeated training, validation, and evaluation, we are still working on the implementation of the individual data processing and sound source separation models.
However, the implementation files for the individual data processing and source separation models remain largely unchanged.
In addition, in incorporating optimizations in the field of speech synthesis, we have used
We also referred to the implementation by [kan-hayashi](https://github.com/kan-bayashi/ParallelWaveGAN).
Thank you very much for publishing an example implementation of a typical speech synthesis model.
