# Deriving Biometrics for Orthostatic Hypotension

This repository is the official implementation of Bachelor Thesis Deriving Biometrics for Orthostatic Hypotension. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
If you have CUDA available on your laptop, download `torch` from the [PyTorch official site](https://pytorch.org/get-started/locally/),
choosing your specific settings. Otherwise, the CPU will be used.
For instance, using Windows with CUDA 11.8, execute:
```setup
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model]() trained on ___ using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### CNN+LSTM

| Metric                    | Score  |
|---------------------------|--------|
| Accuracy of Classification| 92.31% |
| F1 score of Classification| 92.41% |
| OHV1 MAE                  | 98.43  |
| OHV2 MAE                  | 20.84  |
| OTC MAE                   | 5.61   |
| POT MAE                   | 21.18  |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 


---
💡 README template from [here](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md).