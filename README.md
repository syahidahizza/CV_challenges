## Stanford Cars Dataset Classification using Dynamic Augmentation and Stitching Images

[PPT LINK for detail algorithm](https://docs.google.com/presentation/d/1mSd2WATNZr9X9gWaKbN3GbovtA3jijJhMFhMRHw2hf8/edit?usp=sharing)

In general, there are 2 training steps. First, fit network with data augmented by 25 chosen augmentation. Second, generate heatmap, and crop + stitch the initial image then fine-tune the network. We are using Inception V4 as base model for all the experiment.

Requirement :
1. Download cars data, and put it into cars/
2. Download both first and second training checkpoint and put them under models/

### First Training Step
The first Training step is straight-forward. The main different only on 25 random augmentation.
![Alt text](res/first_training.PNG?raw=true "First Training Step")


Train model using the following command
```
python3 train_first_model.py -data <data_path>
```

Then evaluate the model only use 4 of those augmentations (number 4,5,15,20). Existing checkpoint can be used for evaluation as follow:
```
python3 evaluate.py -data <data_path> --resume models/checkpoint_cars_1-e4_decay_real.pth.tar
```
| Scenario                                        | Accuracy      |
|-------------------------------------------------|---------------|
| After First Train                               | 93.906        |
|-------------------------------------------------|---------------|


### Second Training Step
The second training step can be seen as follow:
![Alt text](res/second_training.PNG?raw=true "Second Training Step")
Because the training is fine-tuning process, we choose lr 0.01.
```
python3 train_second_model.py -data <data_path> --resume <name_ckpt_init_train> --lr 0.01 --epoch 1000

```
Test saved model on stanford cars dataset using only output_1

```
python3 evaluate.py -data <data_path> --resume <ckpt_after_second_training> -pred_both False

```
| Scenario                                        | Accuracy      |
|-------------------------------------------------|---------------|
| After Second Train only output 1                | 94.145        |
|-------------------------------------------------|---------------|

Test saved model on stanford cars dataset using only output_1 and output_2
```
python3 evaluate.py -data <data_path> --resume <ckpt_after_second_training> -pred_both True

```
| Scenario                                        | Accuracy      |
|-------------------------------------------------|---------------|
| After Second Train output 1 and 2               | 94.266        |
|-------------------------------------------------|---------------|

