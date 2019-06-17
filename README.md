## Stanford Cars Dataset Classification using Auto Augment and Stitching Images

https://docs.google.com/presentation/d/1mSd2WATNZr9X9gWaKbN3GbovtA3jijJhMFhMRHw2hf8/edit?usp=sharing

Train model on stanford cars dataset

```
python3 train_first_model.py -data <data_path>
python3 train_second_model.py -data <data_path> --resume <name_ckpt_init_train> --lr 0.01 --epoch 1000

```
Test saved model on stanford cars dataset

```
python3 evaluate.py -data <data_path> --resume <ckpt_after_second_training> -pred_both True

```

