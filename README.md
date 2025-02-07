[English/[Japanese](README_JA.md)]

> **Note**
> <br>This is a modified and "enhanced" version based on:
> <br>→ [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
> <br>→ [kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)

# hand-gesture-recognition-with-mediapipe (HGR)
Estimates hand pose using MediaPipe (Python version).<br> This is a sample program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

This repository contains the following contents.
* Sample program (Inference and Data generation for new training)
* Multiple hand sign recognition models (TFLite)
> **NOTE**
> <br>Hand sign recofnition models were taken from different forks that were collected in:
> <br>→ [Kazuhito00/hand-keypoint-classification-model-zoo](https://github.com/Kazuhito00/hand-keypoint-classification-model-zoo)
* Finger gesture recognition model(TFLite)
* Learning data for hand sign recognition (hsr)
* Notebook for hsr training
* Learning data for finger gesture recognition (fgr)
* Notebook for fgr training

# Requirements

* Python 3.9.11 or Later (Only tested until 3.9.11)

### Only inference
* opencv-contrib-python 4.8.1.78 or Later
* mediapipe 0.10.20
* Tensorflow 2.13.0

### Additional for Training
* ipykernel 6.29.5 or Later
* scikit-learn 1.6.1 or Later
* matplotlib 3.3.2 or Later
* pandas 2.2.3 or Later
* seaborn 0.13.2 or Later

# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo:

* --device<br>
Specifying the camera device number, OpenCV webcam index (Default：0)
* --width<br>
Width of camera output (Default：960)
* --height<br>
Height of camera output (Default：540)

* --keypoint_model<br>
Set Keypoint classifier model for inference and data gathering (Default: Kazuhito00) 
* --use_static_image_mode<br>
Enables static_image_mode option for MediaPipe inference (Default: Unspecified) 
* --min_detection_confidence<br>
Hand detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Hand tracking confidence threshold (Default：0.5)
* --keypoint_classifier_confidence<br>
Keypoint classifier confidence threshold (Default: 0.80)

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│
├─model
│  ├─inference
│  │  │
│  │  │  keypoint_classifier.py
│  │  └─ point_history_classifier.py
│  │
│  ├─keypoint_classifier_models
│  │  └─{model_name}
│  │     │  keypoint.csv
│  │     │  keypoint_classifier.hdf5
│  │     │  keypoint_classifier.tflite
│  │     └─ keypoint_classifier_label.csv
│  │
│  └─point_history_classifier_models
│     └─{model_name}
│        │  point_history.csv
│        │  point_history_classifier.hdf5
│        │  point_history_classifier.tflite
│        └─ point_history_classifier_label.csv
│
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a sample program for inference.<br>

In addition, it works for collecting data for hand gesture recognition (keypoint classifier models) and pointer motion recognition (point_history classifier models).

### keypoint_classification.ipynb
This is a model training script for hand sign/gesture recognition (Keypoint classifier models).

### point_history_classification.ipynb
This is a model training script for hand/finger motion recognition (point history classifier models).

### model/inference

This directory contains both inference modules for hand gesture recognition and hand/finger motion recognition.

### model/keypoint_classifier_models
This directory stores a variety of models for hand sign/gesture recognition.<br>

Each model directory contains the following files:
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Labeled data(keypoint_classifier_label.csv)

### model/point_history_classifier_models
This directory stores a variety of models<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)

><span style="color:yellow;">⚠️**WARNING**⚠️</span><br>
> The only hand/finger motion recognition (point history) model available is [Kazuhito00 point history model](./model/point_history_classifier_models/), therefore it's guaranteed to work as expected only with [Kazuhito00 keypoint model](./model/keypoint_classifier_models/Kazuhito00/).
>
>However **app.py** allows you to experiment mixing different [Keypoint models](./model/keypoint_classifier_models/) and [Point History models](./model/point_history_classifier_models/) 

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand sign/gesture recognition and hand/finger motion recognition data can be added for retraining over an existing model or to train a new model.

### Hand sign/gesture recognition training 

Examples are described with based on Kazuhito00 models.
#### 1. Learning data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>

Up to 10 different gestures can be labeled if you are training a new model.

If you press "0" to "9", the key points will be added to "model/keypoint_classifier_models/{model_name}/keypoint.csv" as shown below.<br>

1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
In the initial state of [Kazuhito00 model](./model/keypoint_classifier_models/Kazuhito00/), three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2).<br>
If necessary, add 3 or more (maximum of 10 classes), or prepare a new csv to train a new [Keypoint Classifier model](./model/keypoint_classifier_models/).<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" <br>and modify the labels of "[model/keypoint_classifier/Kazuhito00/keypoint_classifier_label.csv](./model/keypoint_classifier_models/Kazuhito00/keypoint_classifier_label.csv)" or any other keypoint classifier model label csv file as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows (Kazuhito00 model parameters).
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### Hand/Finger motion recognition training
#### 1.Learning data collection
Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>

Up to 10 Hand/Finger motion patterns can be classified.

If you press "0" to "9", the key points will be added to "model/point_history_classifier_models/{model_name}/point_history.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>
In the initial state of Kazuhito model, 4 types of learning data are included: stationary (class ID: 0), clockwise (class ID: 1), counterclockwise (class ID: 2), and moving (class ID: 4). <br>

If necessary, add 5 or more (maximum of 10 classes), or prepare a new csv to train a new [Point History Classifier model](./model/point_history_classifier_models/).<br>

<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

#### 2.Model training
Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 4" and <br>modify the label of "[model/point_history_classifier_models/Kauhito00/point_history_classifier_label.csv](./model/point_history_classifier_models/Kazuhito00/point_history_classifier_label.csv)" or any other point history classifier model label csv file as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[point_history_classification.ipynb](point_history_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
The model using "LSTM" is as follows. <br>Please change "use_lstm = False" to "True" when using (tf-nightly required (as of 2020/12/16))<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# Application example
Here are some application examples.
* [Control DJI Tello drone with Hand gestures](https://towardsdatascience.com/control-dji-tello-drone-with-hand-gestures-b76bd1d4644f)
* [Classifying American Sign Language Alphabets on the OAK-D](https://www.cortic.ca/post/classifying-american-sign-language-alphabets-on-the-oak-d)

# Reference
* [MediaPipe](https://mediapipe.dev/)
* [Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)

# Author
- Kazuhito Takahashi (https://twitter.com/KzhtTkhs)

# General Improvements
- Israel Félix (https://github.com/Leckrosh)

# License

- This project is a fork of [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/tree/main) which is licensed under [Apache License 2.0]()
- Modifications made in this fork are licensed under the [Apache License 2.0] ().

> **NOTE** <br>
> All modifications made on this fork are being described [release notes](RELEASE_NOTES.txt) file
