# Computational Intelligence
ACIT 4620 
----------

## Traffic sign classification using a Deep Convolutional Neural Network in Python

Changing directory to the project folder:
```bash
 cd src
```

In order to install all dependencies, which can be found in requirements.txt:
```bash
python -m pip install -r requirements.txt
```

> :warning: **If you are using a custom environment**: Be sure to install all dependencies in the desired one!

> :warning: **Python version**: You may need to use python3 / python3.x depending on your installation

In order to run the project:
```bash
python main.py
```
> :warning: **Python version**: You may need to use python3 / python3.x depending on your installation

Tensorboard:
```bash
tensorboard --logdir logs
```

> :heavy_check_mark: **Tensorflow GPU utilization**: If you want to enable GPU training, make sure to follow this [guide](https://www.tensorflow.org/install/gpu) step by step

Classes:

| FClassId  | SignName|
| ------------- | ------------- |
| 0 | Speed limit (20km/h)  |
| 1 | Speed limit (30km/h)  |
| 2 | Speed limit (50km/h)  |
| 3 | Speed limit (60km/h)  |
| 4 | Speed limit (70km/h)  |
| 5 | Speed limit (80km/h)  |
| 6 | End of speed limit (80km/h)  |
| 7 | Speed limit (100km/h)  |
| 8 | Speed limit (120km/h)  |
| 9 | No passing  |
| 10 |  No passing for vehicles over 3.5 metric tons |
| 11 | Right-of-way at the next intersection  |
| 12 |  Priority road |
| 13 |  Yield |
| 14 |  Stop |
| 15 | No vehicles  |
| 16 | Vehicles over 3.5 metric tons prohibited  |
| 17 | No entry  |
| 18 |  General caution |
| 19 |  Dangerous curve to the left |
| 20 |  Dangerous curve to the right |
| 21 |  Double curve |
| 22 |  Bumpy road |
| 23 | Slippery road  |
| 24 |  Road narrows on the right |
| 25 | Road work  |
| 26 | Traffic signals  |
| 27 | Pedestrians  |
| 28 | Children crossing  |
| 29 | Bicycles crossing  |
| 30 | Beware of ice/snow  |
| 31 | Wild animals crossing  |
| 32 | End of all speed and passing limits  |
| 33 | Turn right ahead  |
| 34 | Turn left ahead  |
| 35 | Ahead only  |
| 36 | Go straight or right  |
| 37 | Go straight or left  |
| 38 | Keep right  |
| 39 | Keep left  |
| 40 | Roundabout mandatory  |
| 41 | End of no passing  |
| 42 | End of no passing by vehicles over 3.5 metric tons  |
