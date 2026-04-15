# Introduction

Here you will find the necessary instructions for reproducibility of the
data-science results in the paper.

# Downloading the data

The three public datasets can be obtained from the following links

* [IIoT](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot?resource=download).
* [Wikipedia 2010](https://zenodo.org/records/3341488) and [wikipedia
  2011](https://zenodo.org/records/3342157).
* [IBM fraud](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data).

# Installing requirements

To install the requirements simply do

```
python3 -m venv .env
source .env/bin/activate
cd ds_scripts
pip install -r requirements.txt
```

Lastly, our library to simulate event filtering and other utils can be installed
with:

```
# From the head folder of this repository.
pip install -e .
```

# Pre-processing the data

The IIoT and wikipedia datasets require prior processing before usage. In the
folder `pre_processing_scripts` you will find one script to process all
datasets. Just replace the global variables with the proper paths in your
systems and run them.

If everything goes well you should now have a single `.parquet` file for each of
these datasets.

# Importing the data into clickhouse

Most filtering simulation run over clickhouse. As such, the parquet files should
be imported to clickhouse. In order to do that run the
`process_<dataset-name>.py` script.

These scripts require just two arguments:

```python
    parser.add_argument(
        '--migration_config_args',
        type=str,
        required=True,
        help=
        'Path to the yaml containing the arguments for the database migration.'
    )

    # For example "~/ibm.chdb"
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )
```

The first is the path to the migration config folder. The second is the path to
where the clickhouse database is going to be saved

The migration config files can be found under
`<dataset_name>_configs/<dataset_name>-migration-args.yaml`. On those files
there exists the following line that must be edited:

```yaml
file_path: PATH_TO_PARQUET_HERE
```

This line should point to yout `.parquet` file of the respective dataset.

Once this step is finished you should have three clickhouse databases saved to disk.

# Simulating filtering

# Training models
