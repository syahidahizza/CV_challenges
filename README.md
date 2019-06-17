## Stanford Cars Dataset Classification using Dynamic Augmentation and Stitching Images

[PPT LINK for detail algorithm](https://docs.google.com/presentation/d/1mSd2WATNZr9X9gWaKbN3GbovtA3jijJhMFhMRHw2hf8/edit?usp=sharing)

In general, there are 2 training steps : 
1. Fit network with augmented data by 25 chosen augmentation. 
2. Generate a heatmap, then crop + stitch the initial image, and fine-tune the network. 

We are using Inception V4 as base model for all the experiment.

Requirement :
1. Download cars data, and put it into cars/
2. Download both [first](https://drive.google.com/file/d/1pOwBUhDfI1D9qXfC60X6ieJ2vdQwx8Mx/view?usp=sharing) and [second](https://drive.google.com/file/d/12DZrvDsxdKxcZyf9uQWmKFAMFU58dAV8/view?usp=sharing) training checkpoint and put them under models/

### First Training Step
The first Training step is straight-forward. The main different only on 25 random augmentations.
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
| After First Train                               | 93.906*       |


### Second Training Step
The second training step can be seen as follow:
![Alt text](res/second_train.PNG?raw=true "Second Training Step")
Because the training is fine-tuning process, we choose lr 0.01.
```
python3 train_second_model.py -data <data_path> --resume models/checkpoint_cars_1-e4_decay_real.pth.tar --lr 0.01 --epoch 1000

```
Test saved model on stanford cars dataset using only output_1

```
python3 evaluate.py -data <data_path> --resume models/ckpt_combined_network.pth.tar -pred_both False

```
| Scenario                                        | Accuracy      |
|-------------------------------------------------|---------------|
| After Second Train only output 1                | 94.145*       |

Test saved model on stanford cars dataset using only output_1 and output_2
```
python3 evaluate.py -data <data_path> --resume models/ckpt_combined_network.pth.tar -pred_both True

```
| Scenario                                        | Accuracy      |
|-------------------------------------------------|---------------|
| After Second Train output 1 and 2               | 94.266*       |

(*) : The accuracy may be different due to the probability factor in augmentation function. However, by performing multiple experimentations, we are confident that the difference is under 1%.
