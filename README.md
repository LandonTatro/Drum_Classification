# Drum_Classification - AI Academy Capstone

# Purpose
Speech recognition as well as other sound recognition technologies have been a major point of interest in the data science field for many years now, though interest in machine learning and artificial intelligence in the arts is on the rise. This project has a long term goal of creating a model that will predict common drum sounds in real time. Theoretically, this can be deployed in a phone app where a plain text notation of all of the notes played will be transcribed to the device.

An example of a theoretical output could be:

![image](https://user-images.githubusercontent.com/108957599/211441573-8442a67c-0390-4202-a6d6-ab5d3e1395c3.png)

In its current state, the model is capable of predicting among seven categores with high accuracy.

The seven categories are:

  - Snare
  - Rack Tom
  - Floor Tom
  - Bass
  - Hi-hat
  - Crash
  - Ride

# Data

## Dataset

The dataset used for the training and scoring of the models was downloaded [here](https://www.dropbox.com/s/p736vokha3240e6/MDLib2.2.zip?dl=0).

Within this dataset, there are drum sound variants such as "strike" (indicating a normal hit on the drum head), "rim" (hitting the metal rim of the drum), and "buzz" (letting the drumstick quietly drag/vibrate on the drumhead). For the purposes of this project, the strike variant was used for all drums. For cymbals, all variants were used (usually included "tip", "crash", and "clamp").

This project uses exactly 4,000 sounds from the dataset, and it is distributed as follows:

  - hi-hat: 1,280 (tip + foot)
  - crash: 640 (crash + tip + clamp)
  - snare: 576 (strike)
  - kick: 480 (long kick + dead kick)
  - rack tom: 384 (strike)
  - floor tom: 384 (strike)
  - ride: 256 (tip)

## Data Preparation

The data was in .wav file format, which is fine to use with the Python library "librosa", which was used through the duration of the project. The initial function used to load the wav files provides the audio's amplitude in time-series format while also providing the audio sample rate (how many times audio is recorded per second - 22,050 times per second in this case). The audio files were trimmed down by cutting the remaining audio when the volume fell below 25 decibels. From here, the length of the audio file was calculated by dividing the amount of samples (length of the time series) by the sample rate.

Feature engineering became important as the length of audio alone doesn't tell us much. The following features were extracted from our trimmed time series:

  - root mean square values
  - spectral centroid
  - spectral bandwidth
  - spectral rolloff
  - zero crossing rate

If you are interested in learning more about the other features offered within librosa, check out the [feature documentation](https://librosa.org/doc/main/feature.html).

These values alone are not directly usable within the model, but inferred summary statistics from each calculation were effective. The following summary statistics were used as features:

  - mean
  - standard deviation
  - minimum value
  - Q1
  - median
  - Q3
  - maximum
  - Interquartile Range (IQR)

In total, 41 features were used (trimmed length + each summary statistic variation of each sound feature).

# Modeling

## Baseline Model & Results

The data was split into training and test sets, 75% and 25% respectively... This resulted in 3,000 training samples and 1,000 test samples.

As a baseline, a random forest classifier ensemble approach was taken. Random forest classifiers are known to be robust ensemble methods capable of making reasonably quick predictions. When designed correctly, random forests can mitigate issues with overfitting by tuning various hyperparameters. Basic experimentation was done tuning the hyperparameters, but the out-of-the-box model performed best (excluding the number of estimators, which was set to 300 trees).

### Results

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The overall accuracy followed by more in-depth results are as follows:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Balanced accuracy score: 97.81%

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="350" alt="rf_classification_report" src="https://user-images.githubusercontent.com/108957599/211449231-f75f21ea-d3eb-4fac-a1ed-c63186ac9361.PNG">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="350" alt="rf_cm" src="https://user-images.githubusercontent.com/108957599/211449241-2d251b9e-1981-4603-b8b2-08c7f552844e.PNG">


## Final Model & Results

The final model used was the extreme gradient boost (known as XGBoost or XGB). The XGBoost is a more powerful form of the random forest that is capable of learning from the error coefficient in the prior decision trees. There are many hyperparameters to tune, such as the learning rate or gamma, that assist in avoiding overfitting while reaching impressive results quickly. In this case, mostly default settings proved to be effective, while the number of estimators was also set to 300 trees.

### Results

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The overall accuracy followed by more in-depth results are as follows:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Balanced accuracy score: 98.39%

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="350" alt="xgb_classification_report" src="https://user-images.githubusercontent.com/108957599/211449449-5b100266-49df-4262-a692-528b4f4ff7ad.PNG">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="350" alt="xgb_cm" src="https://user-images.githubusercontent.com/108957599/211449457-71413fc5-a6ab-4c6d-acd7-c7fccca1361e.PNG">

# Next Objective

Identifying individual drum sounds is useful for linear drum patterns where only one drum or cymbal is played at a given time. Next, training a model on combinations of hits, such as a hihat being played with a snare, could be valuable. Additionally, if any sort of timing vectorization step is taken as real time predicting is integrated, it is likely that simple time signature and notes would be best to start with.


