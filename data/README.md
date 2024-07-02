# Dataset Struture Documentation

This folder gives an example of the dataset structure using fake data. 

Here, each training and validation example is assigned a unique id called "core_id". 

The dataset should include a metadata.csv file where each row corresponds to a core in the dataset, and should have, at minimum, columns for "core id" and "grade" as shown. 

The main data (bmode ultrasound, (optional) rf ultrasound, prostate and needle region masks) should be stored in an HDF5 file (`.h5` extension). It should contain groups for bmode, optionally rf, prostate mask and needle mask. For each type of data, the file should contain an entry for each core id in the metadata table. It should be accessible via, for example: 

```python
import h5py 

CORE_ID = 'test'

with h5py.File('data.h5', 'r') as f: 
    print(f['bmode'][CORE_ID])
    print(f['prostate_mask'][CORE_ID])
    print(f['rf'][CORE_ID]) # optional 
    print(f['needle_mask'][CORE_ID])
```

We also need a splits file which is a json-encoded file containing the list of cores for train, val, test, and ssl_train (cores used in self-supervised pretraining).

Finally, we also have a "Prompt table" which is a csv file containing a column for core_id and corresponding values for various prompt types. The prompt values should be either floating point or discrete integers. The prompts chosen for a given training run can be specified using the `--floating_point_prompts`, `--discrete_prompts`, and `--discrete_prompts_nvals` flags of the `train_prostnfound.py` script.

In this folder we have included fake data matching this structure for reference and illustration - you would export your data in this format and then pass paths to the extracted data in the training scripts. 
