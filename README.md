# MLM: Multiple Languages and Modalities

## About

Multiple Languages and Modalities (MLM) is a dataset consisting of text in three languages (EN, FR, DE), images, location data, and triple classes.
The resource is designed to evaluate the strengths of multitask learning systems in generalising on diverse data. [The paper](http://cleopatra.ijs.si/goal-mlm/report) defines a benchmark evaluation consisting of the following tasks:
- Cross-modal retrieval
- Location estimation

Additional details on the resource and benchmark evaluation are available at the MLM website:
http://cleopatra.ijs.si/goal-mlm/
IR+LE is an architecture for a multitask learning system designed as a baseline for the above benchmark. The pipeline for cross-modal retrieval extends an approach proposed by Marin et al:
http://im2recipe.csail.mit.edu/im2recipe-journal.pdf.


**Multitask IR+LE Framework**

![system](/ir+le.png)

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
Multitask Learning (IR + LE)
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

Cross-modal retrieval task
``` bash
python test.py --task ir
```

Location estimation task
``` bash
python test.py --task le
```

All logs and checkpoints will be saved under the experiments folder.

## License
The repository is under [MIT License](LICENSE).

## Cite
``` bash
@inproceedings{armitage2020mlm,
  title={Mlm: a benchmark dataset for multitask learning with multiple languages and modalities},
  author={Armitage, Jason and Kacupaj, Endri and Tahmasebzadeh, Golsa and Maleshkova, Maria and Ewerth, Ralph and Lehmann, Jens},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={2967--2974},
  year={2020}
}
```
