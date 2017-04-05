#!/bin/sh

train=$1
val=$2
rho=$3

echo 'training 1 step'
th main.lua -retrain resnet-50.t7 -resetClassifier -LR 0.01  -train_list $train  -val_list $val -RandomSizeCrop 
mv model_best.t7 save/model_step_1.t7

for((i=2;i<=$rho;i++)) 
do   
echo "training ""$i"" step"
th main.lua -retrain save/"model_step_""$((i-1))"".t7" -LR 0.001 -rho $i -learnStep -train_list $train  -val_list $val -freezeParam
mv model_best.t7 save/"model_step_""${i}"".t7"
done  

echo "training dynamic"
th main.lua -retrain save/"model_step_""${rho}"".t7"  -LR 0.001 -rho $rho -learnStep -dynamic -train_list $train  -val_list $val -freezeParam
mv model_best.t7 save/"model_step_""${rho}""_dynamic"".t7"
