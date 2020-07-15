# Multimodal Emotion Recognition

The aim of this project is to classify 7 emotions within the EMO dataset provided by FG2020 conference.

#### The dataset

<img src="./pics/1.png"  width="150" height="150"> <img src="./pics/2.png"  width="150" height="150">




### Augmentation
Given the small dataset, we implied 2 types of augmentation:
* Geometric augmentation on frames

<img src="/pics/normal.png"  width="150" height="150">
<img src="/pics/mirror.png"  width="150" height="150">
<img src="/pics/perspective.png"  width="150" height="150">
<img src="/pics/rotate.png"  width="150" height="150">
<img src="/pics/rotate+perspective.png"  width="150" height="150">

* Normalizing the sequence in N segments and drawing a frame from each segment with a normal distribution on that segment

<img src="/pics/aug.png"  width="150" height="150">

### Models
There are 6 models, the best performing one (VAK 6) is detailed in the picture below
<img src="/pics/vak_final.png"  width="2600" height="1300">
 

### Results
| Model name  | Acc on train data  | Acc on test data  |
|---|---|---|
| VAK 6  | 50.9%  | 57.5%  |

Where accuracy per class is

| Model name  | Neutral state  | Sadness | Surprise | Fear | Anger | Disgust | Happiness | 
|---|---|---|---|---|---|---|---|
| 89.2 % | 46.4 % | 42.8 %| 57.1 % | 71.4 % | 32.1 %| 53.5 % | 
