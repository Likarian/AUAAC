# AUAAC
Official implementation of the paper 'AUAAC: Area Under Accuracy-Accuracy Curve for Evaluating Out-of-Distribution Detection'

---
# How to use
### Dependencies
* pytorch >= 2.0
  
### Quick start
*  Trained network weight should be in model directory
<br>
Run the following code.
<br>
For calculating ACC-IND and ACC-OOD, and saving scores in ./result/*.csv

> python cal_accind&accood.py --net resnet34 --dataset cifar10 --ood svhn --specific normal

<br>
For calculating AUAAC and drawing the curve

> python csv2curve.py --net resnet34 --dataset cifar10 --ood svhn --specific normal

<br>
For calculating AUAAC with entire csv files in ./result

> python cal_score.py


