# Speech-Emotion-Recognition

Introduction to Deep Learning in Speech \& Language Processing (WS23/24) Coding Task.

## Setup

*Prerequisite:*
- python 3.11
- cuda 12

*Install packages:*
```bash
  pip install -r requirements.txt
  ```

*Data Preparation:*
- Download the datasets ```train.json```, ```dev.json```, ```ser_test_1.json``` and ```ser_test_2.json``` to ```/dataset```.

## Data Preprocessing

Create a Dataset object from the raw datasets.

In ```/src```, run:
```bash
python preprocess.py -i ../dataset -o ./output
```

## Training

In ```/src```, run:
```bash
python train.py -i ./data_preprocessing/output --dataset_fn dataset.pt -o /model_ckpts --input_size 26 --num_heads 2 --num_encoder_layers 4 --num_classes 4 --pos_enc True -e 10 -bs 64 -lr 5e-2 -wd 0.01
```

- ```-i```, ```--input_dir```: Path to the preprocessed dataset.
- ```--dataset_fn```: File name of the preprocessed dataset.
- ```-o```, ```--output_dir```: Path to save the trained model.
- ```--input_size```: Dimension of the input features.
- ```--num_heads```: Number of attention heads.
- ```--num_encoder_layers```: Number of encoder layers in the Transformer encoder.
- ```--num_classes```: Number of classes.
- ```--pos_enc```: Whether to add positional encoding to the input features.
- ```-e```, ```--num_epochs```: Number of epochs.
- ```-bs```, ```--batch_size```: Batch size.
- ```-lr```, ```--learning_rate```: Learning rate.
- ```-wd```, ```--weight_decay```: Weight decay.

## Inference

In ```/src```, run:
```bash
python inference.py -i ./data_preprocessing/output -o <output_fn> -m <model_fn> --mode <mode>
```

- ```-i```, ```--input_dir```: Path to the model checkpoint.
- ```-o```, ```--output_fn```: Path + file name of the predictions.
- ```--mode```: Whether the inference is done on ser_test_1.json ('test') or ser_test_2.json ('final'), can be either "test" or "final".

## Reference
Ashkan Yazdani, Evangelos Skodras, Nikolaos Fakotakis, and Touradj Ebrahimi. Multimedia content analysis for emotional characterization of music video clips. *EURASIP Journal on Image and Video Processing*, 2013(1):1–10, 2013.
