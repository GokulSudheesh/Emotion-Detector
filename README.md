# Emotion-Detector
## Notes
The `requirements.txt` file contains all the dependencies needed for the project.
```
pip install -r requirements.txt
```
## Detecting from images:
<img src="github-images/1.jpeg" width="300" height="240">
<img src="github-images/2.jpeg" width="300" height="240">
<img src="github-images/3.jpeg" width="300" height="240">

## Detecting from web cam:
<img src="github-images/5.jpeg" width="300" height="240">

### How the Network Architecture works (networkArchitecture1.py):
Stacking conv layers one after the other has more flexibility in expressing non-linear transformations without loosing information. Maxpool removes information from the signal, dropout forces distributed representation, thus both effectively make it harder to propagate information. If, for given problem, highly non-linear transformation has to be applied on raw data, stacking multiple convs (with relu) will make it easier to learn, that's it. 
(src: https://stackoverflow.com/questions/46515248/intuition-behind-stacking-multiple-conv2d-layers-before-dropout-in-cnn)

### How the AlexNet Works:
AlexNet has a total of 8 layers consisting of 5 convolutional layers followed by 3 fully connected layers.
(src: https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637)
<img src="https://miro.medium.com/max/3072/1*qyc21qM0oxWEuRaj-XJKcw.png" width="800" height="300">
