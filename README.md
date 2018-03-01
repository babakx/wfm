Weighted Factorization Machines
===============================

This is an implementation of Weighted Factorization Machines (WFMs) with tensorflow based on [tffm](https://github.com/geffy/tffm).

## Documentation
To be completed.

## Usage
```
usage: wfm.py [-h] [--dataset ml1m/frappe/kassandra/msd/goodbooks]
              [--model bpr/fm/fmp/wfm/wfmp/wmf] [--load_model FOLDER_PATH]
              [--save_model FOLDER_PATH] [--epochs N] [--eval_freq N]
              [--eval_results FILE_PATH] [--order N] [--k K]
              [--hyper_params LR,REG,STD] [--batch_size N]
              [--has_context true/false] [--implicit true/false]
              [--weights all-one/all-diff/c-one/c-diff]
              [--all-conf true/false]

Weighted Factorization Machines

optional arguments:
  -h, --help            show this help message and exit
  --dataset ml1m/frappe/kassandra/msd/goodbooks
                        name of dataset (default: )
  --model bpr/fm/fmp/wfm/wfmp/wmf
                        name of the model (default: fmp)
  --load_model FOLDER_PATH
                        the path to the model tf model folder (default: None)
  --save_model FOLDER_PATH
                        folder path to save the tf model (default: None)
  --epochs N            number of epochs (default: 10)
  --eval_freq N         evaluate every N epochs (default: 10)
  --eval_results FILE_PATH
                        name of the file to save evaluation results to
                        (default: None)
  --order N             order of FM (default: 2)
  --k K                 number of latent factors (default: 10)
  --hyper_params LR,REG,STD
                        comma-separated list of hyper-parameters:
                        LearningRate, Regularization, InitStd. (default:
                        0.01,0.01,0.01)
  --batch_size N        number of samples in each mini-batch (default: 10000)
  --has_context true/false
                        Whether data has context or not (default: True)
  --implicit true/false
                        Whether dataset is implicit or explicit (default:
                        True)
  --weights all-one/all-diff/c-one/c-diff
                        how the weights should be initialized (default: c-dif)
  --all-conf true/false
                        Run experiments with pre-defined configurartions
                        (default: False)
```

