# MLM

## About

Multiple Languages and Modalities (MLM) is a dataset consisting of text in three languages (EN, DE, FR), images, location data, and triple classes.
The resource is designed to evaluate the strengths of multitask learning systems in generalising on diverse data. The paper (publication pending) defines a benchmark evaluation consisting of the following tasks:
- Cross-modal retrieval
- Location estimation

IR+LE is an architecture for a multitask learning system designed as a baseline for the above benchmark. The pipeline for cross-modal retrieval extends a system proposed by Marin et al:
http://im2recipe.csail.mit.edu/im2recipe-journal.pdf.

Multitask IR+LE Framework

![alt text](https://github.com/GOALCLEOPATRA/MLM/mlm_baseline/blob/ir+le.png?raw=true)

## IR+LE System and MLM Dataset 
### Requirements and Setup
Python version >= 3.7

PyTorch version >= 1.4.0

``` bash
# clone the repository
git clone https://github.com/GOALCLEOPATRA/MLM.git
cd MLM
pip install -r requirements.txt
```

### Download MLM dataset

Download the dataset hdf5 files from [here](https://zenodo.org/record/3885753) and place them under the [data](data) folder.

### Train tasks
Multi-task Learning (IR + LE)
``` bash
python train.py --task mtl
```

Cross-modal retrieval task
``` bash
python train.py --task ir
```

Location estimation task
``` bash
python train.py --task le
```

For setting other arguments (e.g. epochs, batch size, dropout), please check [args.py](args.py).

### Test tasks
Multi-task Learning (IR + LE)
``` bash
python test.py --task mtl
```

Information retrieval task
``` bash
python test.py --task ir
```

Location estimation task
``` bash
python test.py --task le
```

All logs and checkpoints will be saved under the experiments folder.

### Files for location and scene embeddings

https://tib.eu/cloud/s/SWNckCMgdxSMXR7

## License
The repository is under [MIT License](LICENSE).

## Cite
``` bash
Coming Soon!
```
