# Verification of music source separation based on generative optimization

[日本語](../../README.md) | English

## Overview and current status of music source separation
Music source separation refers to the technique for decomposing a music, once synthesized, into its constitutive sources. By enabling individual manipulation of sources, it leads to the realization of advanced functions in music distribution services. In fact, music source separation has been attempted to be applied to functions such as remixing, making karaoke tracks, music information retrieval, and music education. However, its current technology has not been able to completely eliminate problems such as "bleeding" from other sources or "missing" of the target source itself. In addition, "noise" may occur in parts of the sound. These factors decrease subjective evaluation.
<p align="center"><img width="600" src="https://user-images.githubusercontent.com/67317828/221925985-efe62bcf-41a0-4aee-90ff-3b3ea55142a7.gif">

## Music source separation based on generative optimization

Music source separation has been intensively studied over the past decade and has made a quantum leap with the advent of deep learning. Music source separation based on deep learning has evolved through two main approaches: the masking approach in the frequency domain and the End-to-End approach in the time domain. In the latter approach direct reconstruction from the extracted features to the waveforms. Such structures have been commonly used in the field of speech synthesis. However, while models incorporating generative structures have been proposed, optimization for these models differs from that in the field of speech synthesis. In the previous method, the waveform loss has been minimized over the time domain. The waveform loss is obtained by taking the difference from its label at each time of the waveform. In other words, a single signal at the same time is compared. In contrast, optimization in the field of speech synthesis considers the time variation of the series signal. Minimizing the STFT loss allows for learning the frequency characteristics. The human auditory system also performs frequency analysis, and this method is based on human hearing. In addition, the minimization of the adversarial loss can be used to learn the authenticity of the sound. The minimization of the adversarial loss is based on the maximization of the probability that the output series is a real sound, enabling authenticity generation. Therefore, I propose a music source separation based on generative optimization. In addition to the waveform loss, the STFT loss and the adversarial loss are minimized simultaneously. By better learning, not only the similarity of waveforms but also the difference in timbre, the subjective evaluation about separation accuracy and sound contamination can be improved.
<p align="center"><img width="600" alt="architecture3" src="https://user-images.githubusercontent.com/67317828/217165468-09999921-727f-4427-999d-d29c7e9862b8.png"></p>

## Validation results for generative optimization

**Dataset - [MUSDB-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)**  
The largest dataset for music source separation.
Data extensions were applied and used.

**Generator - [Hybrid Demucs](https://arxiv.org/pdf/2111.03600.pdf)**  
The End-to-End model Demucs in the time domain was extended to the hybrid domain.

**Discriminator - [StyleMelGAN](https://arxiv.org/pdf/2011.01557.pdf) discriminator**  
Lightweight and high quality generation.

**Wave loss - Mean Absolute Error (MAE)**  
Used in Hybrid Demucs.

**STFT loss - Spectrogram difference**.  
Calculated by ```(Linear Scale + Power Scale) x 3 Resolutions```.
Please refer to [Parallel WaveGAN](https://arxiv.org/pdf/1910.11480.pdf).

**Adversarial Loss - Squared Error**  
Please refer to [Least Squares GAN](https://arxiv.org/pdf/1611.04076.pdf).

**Subjective metric - Mean Opinion Score (MOS)**.  
The percentage of votes for each category, weighted by the grade (38 people in this study).

<p align="center"><img width="600" alt="mos_item" src="https://user-images.githubusercontent.com/67317828/222974185-9fdfbdeb-7159-4753-9e67-5961b4c15e2a.png"></p>

**Objective metric - [Source-to-Distortion Ratio (SDR)](https://hal.inria.fr/inria-00544230/document)**  
Measures the degree of distortion from the correct solution by comparing signals.
It has been commonly used for easy and uniform evaluation.
However, it has been reported that [it does not correlate well with human hearing](https://ieeexplore.ieee.org/document/7760550), and its validity is questionable.
The validity of this method is questionable.

Optimization with the addition of the STFT loss (at multiple scales and resolutions) has been shown to improve the overall subjective evaluation of separation accuracy and sound contamination.
In fact, by drawing a log-mel spectrogram
the contribution of the STFT loss could be seen.
Although only sound contamination of bass was slightly reduced.
We believe that this is due to the removal of high-frequency components that should not have been present. High-frequency components are components that form the detailed shape of waveform. The elimination of the components that were used to make the waveforms add up may have slightly disturbed the consistency of the waveforms and degraded the sound contamination. The optimization with additional adversarial loss resulted in lower separation accuracy, but improved the sound contamination of the bass. The reason for the reduced separation accuracy may be that the discriminator that calculates the adversarial loss has overlearned. Possible countermeasures include making the discriminator lighter and introducing techniques to suppress over-learning. Eliminating instability in learning the adversarial loss is a future challenge.

<p align="center"><img width="600" alt="mos_en" src="https://user-images.githubusercontent.com/67317828/222974424-568ea64a-fd10-4d1d-8497-d05959fe9a83.png"></p>
<p align="center"><img width="600" alt="sdr_en" src="https://user-images.githubusercontent.com/67317828/222974429-c2b52a4b-f3e4-4b58-adfb-b98f64ce0ae0.png"></p>

Since music is created to be listened to by humans, it is important to improve the subjective evaluation of music source separation systems. The subjective evaluation results of the music source separation based on generative optimization implemented in this study exceeded those of conventional methods in many aspects. It is hoped that the optimization will be reviewed in the future and lead to further quality improvement. However, while subjective evaluation is necessary, in practice, it is common that only objective evaluation is performed. In the future, it is also necessary to set up a system that allows for the active testing of subjective evaluation by establishing clear criteria for subjective evaluation, creating a platform, and so on.

# Usage
Please refer to [here](setup.md).

# License
Please refer to [here](../../LICENSE).
This system was created by duplicating the [implementation of Hybrid Demucs](https://github.com/torakyun/music_source_separation/tree/64ed2cb029301743b2714b3c8fe930c00945842c).
We would like to thank the following people for their significant contributions to music source separation. The introduction of the ``Trainer`` class has allowed us to create a large system of iterative learning, validation, and evaluation. Although we have made significant changes to the general framework of the system through repeated training, validation, and evaluation, we are still working on the implementation of the individual data processing and sound source separation models. However, the implementation files for the individual data processing and source separation models remain largely unchanged. In addition, in incorporating optimizations in the field of speech synthesis, we have used We also referred to the implementation by [kan-hayashi](https://github.com/kan-bayashi/ParallelWaveGAN).
Thank you very much for publishing an example implementation of a typical speech synthesis model.
