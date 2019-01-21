# Data Difficulty Specified Crowdsourcing

This is our team project for *Statistical Learning* (2018 Fall) by Yilun Xu, Tianyuan Zhang, Dinghuai Zhang and Weiyi Zhang.

This project focuses on a data difficulty specified algorithm of crowdsourcing.

The final report is in `report`.

-----
Required environment:

```
cuda >= 8.0
python >= 3.6
pytorch >= 0.4
numpy >= 1.14.3
pillow >= 5.1.0
scipy >= 1.1.0
```



- To run the experiment of the Gaussian dataset:

`python3 cotraining_logistic.py --device device_num --expert_num 5 --expertise 1 --case 1 --save s`



>device_num : The GPU number.
>
>s: 1=save the log, 0=unsave the log



- To run the experiment of the Dogs & Cats dataset (structural label):

`python3 cotraining_logistic.py --device device_num --expert_num 5 --expertise 1 --case 1 --save s --root-dir path`



> device_num : The GPU number.
>
> s: 1=save the log, 0=unsave the log
>
> path : path to the dataset



- To run the experiment of the Dogs & Cats dataset (non-structural label):

`python3 cotraining_logistic.py --device device_num --expert_num 5 --expertise 1 --case 3 --save s --root-dir path`



> device_num : The GPU number.
>
> s: 1=save the log, 0=unsave the log
>
> path : path to the dataset



