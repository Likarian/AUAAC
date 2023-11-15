# AUAAC
Official implementation of the paper 'AUAAC: Area Under Accuracy-Accuracy Curve for Evaluating Out-of-Distribution Detection'

---
# How to use
### Dependencies
* pytorch >= 2.0
  
### Quick start
Trained network weight should be in model directory.
<br>
You can find the weight of ResNet34 trained with cifar10 in the following URL.
<br>
https://drive.google.com/file/d/1iIBWgmym2U1VACOXWUCnA_h5INWJI9QG/view?usp=sharing

#### For calculating ACC-IND and ACC-OOD, and saving scores in ./result/*.csv

> python cal_accind&accood.py --net resnet34 --dataset cifar10 --ood svhn --specific normal

#### For calculating AUAAC and drawing the curve

> python csv2curve.py --net resnet34 --dataset cifar10 --ood svhn --specific normal

#### For calculating AUAAC with entire csv files in ./result

> python cal_score.py


