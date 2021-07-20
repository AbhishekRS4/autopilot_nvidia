# Autopilot - end to end learning for self-driving cars

## Notes
* Implementation of Autopilot based on paper from Nvidia Research
* This implementation is to predict the steering angle using an image
* The images are resized to 200x66 for training and prediction as mentioned in the paper

## Implementation details
* The data is divided into train/valid/test sets randomly

## Instructions to run training and inference script
* To list training options
```
python3 src/autopilot_train.py --help
```
* To list inference options
```
python3 src/autopilot_infer.py --help
```

## Reference
* [Nvidia Autopilot dataset](https://drive.google.com/open?id=1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B)
* [Nvidia's Autopilot - End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf)
