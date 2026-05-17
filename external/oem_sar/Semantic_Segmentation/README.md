
## Usage
<div align="justify">

The repository structure consists of `dataset/train` and `dataset/test` for training and test data; a `pretrained/` directory for pretrained weights; and all source codes in the `source/`. The training and testing scripts, `train.py` and `test.py`, respectively, are at the root of the repo. The `docs` directory contains only GitHub page files.

To use the baseline code, first, clone the repository and change your directory into the `DFC2025-OEM-SAR-Baseline` folder. Then follow the steps below:</br>
1. Install all the requirements. `Python 3.8+` was used in our experiments. Install the list of packages in the `requirements.txt` file using `pip install -r requirements.txt`.
2. Download the dataset from [here](https://zenodo.org/records/14622048) into the respective directories: `dataset/train` and `dataset/test`
3. Download the pretrained weights from [here](https://drive.google.com/file/d/1Myd8b2KVFRuYVPyjB6EAv70OsNmjtgB9/view?usp=sharing) into the `pretrained` directory

Test the model with the pretrained weights by running the script `test.py` as:
```bash
python test.py
```
To train the model, run `train.py` as:
```bash
python train.py
```
</div>



## Acknowledgement
<div align="justify">

We are most grateful to the authors of [Semantic Segmentation PyTorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file) from which the baseline code is built on.
</div>
