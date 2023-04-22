# Change point detection in seismic data

Change point detection is extremely important as it helps us automatically evaluate the evolution of different developments throughout the globe. In this project, we evaluated different change point detection techniques from the *[ruptures][ruptures]* package. We applied these techniques to seismic data which can be downloaded using the *[obspy][obspy]* package. Our simple implementation aimed at comparing change point detection using different algorithms and data of different magnitude (in terms of displacement frequency and periods).

Our main results are presented in the *report/report.pdf* file and can be reproduced thanks to the *change_point_detection_rms.ipynb* notebook using the provided *data/displacement_from_2020_01_01_to_2022_01_01_FR.STR.00.BHZ.csv* data file.

Please refer to the following sections for more information about the package usage:

2. [Installation](#installation-instructions)
3. [Description](#package-description)
4. [Usage via command lines](#package-usage)
5. [Documentation](#documentation)

## Installation instructions

In order to use our package and run your own experiments, we advise you to set up a virtual environment.

You will need Python 3 and the *virtualenv* package:

    pip3 install virtualenv

Then, create your virtual environment and switch to it:

    python3 -m venv venv

    source venv/bin/activate (Linux)
    .\venv\Scripts\Activate.ps1 (Windows PowerShell)

Finally, install all the requirements:

    pip3 install -r requirements.txt (Linux)
    pip3 install -r .\requirements.txt

*Note*: Tested on Linux with Python 3.10.9.

## Package description

Below, we give a brief tree view of our package.

    .
    ├── data
    |   └── displacement...FR.STR.00.BHZ.csv  # to reproduce our results
    ├── doc  # contains a generated documentation of src/ in html
    ├── report  # contains our report in pdf format
    ├── src  # source code
    |   ├── models
    |   |   ├── __init__.py
    |   |   ├── hub.py  # wrapper around *ruptures* package
    |   |   ├── ppsd.py
    |   |   └── rms.py
    |   ├── routine
    |   |   ├── __init__.py
    |   |   ├── ppsd.py  # download seismic data and compute psd
    |   |   └── rms.py  # compute rms displacement data
    |   ├── utils
    |   |   ├── __init__.py
    |   |   ├── data.py
    |   |   ├── misc.py
    |   |   └── plotting.py
    |   ├── __init__.py
    |   └── data_preparation.py  # main file to prepare data
    ├── change_point_detection_ppsd.ipynb
    ├── change_point_detection_reproduction.ipynb
    ├── change_point_detection_rms.ipynb
    ├── download_seismic_data.ipynb
    ├── README.md    
    └── requirements.txt  # contains the required Python packages to run our files

## Package usage

Our implementation of change point detection can be found under the *src* folder.

### Notebooks

In order to use the notebooks, you will also need to install *jupyter*:

    pip3 install jupyter notebook ipykernel
    ipython kernel install --user --name=myvenv

There are four available notebooks:

- download_seismic_data.ipynb: run routines to download seismic data
- change_point_detection_rms.ipynb: try to detect the lockdowns and curfew dates in Strasbourg, France
- change_point_detection_ppsd.ipynb: same as above, but with the whole spectrogram (multivariate)
- change_point_detection_reproduction.ipynb: apply our functions to reproduce results from Maciel et al.

Reference: Susanne Taina Ramalho Maciel, Marcelo Peres Rocha, Martin Schimmel. *Urban seismic monitoring in Brası́lia, Brazil.* 2021.

### Reproduce our results

The results that are presented in our article can be reproduced easily by using the provided data and notebooks. The notebook to run is *change_point_detection_rms.ipynb*.

### Run your own experiments

First, you will need to download data from an available source. You can use the *src.data_preparation.py* file to do so:

```bash
python3 src/data_preparation.py [options]
```

- `--data-dir`: Name of the directory where data is stored. Default: "data".
- `--dataset-name`: Name of the dataset to use. Default: "experiment".
- `--save`: If True, will save the RMS data. Default: False.
- `--provider-name`: Name of the data provider. Default: "RESIF".
- `--network`: Name of the network. Default: "FR".
- `--station`: Name of the station. Default: "STR".
- `--location`: In some cases, a station might include multiple sensors. Location of the desired sensor. Default: "00".
- `--channel`: Name of the channel. Names can differ between networks. Default: "BHZ".
- `--start-date`: Start date to query data in the format "%Y-%m-%d". Default: "2020-03-01".
- `--end-date`: End date to query data in the format "%Y-%m-%d". Default: "2022-01-01".
- `--db-bins`: Specify the lower and upper boundary and the width of the db bins in the format lower_upper_width. Default: "-200_20_0.25".
- `--ppsd-length`: Length of data segments passed to psd in seconds. Default: 1800.
- `--overlap`: Overlap of segments passed to psd. Overlap may take values between 0 and 1. Default: 0.5.
- `--period-smoothing-width-octaves`: Determines over what period/frequency range the psd is smoothed around every central period/frequency. Default: 0.025.
- `--period-step-octaves`: Step length on frequency axis in fraction of octaves. Default: 0.0125.
- `--period-limits`: Set custom lower and upper end of period range in the format "lower_upper". Default: "0.008_50".
- `--freqs-pairs`: All pairs of frequencies to compute RMS values in the format "low1-up1 low2-up2 ...". Default: "0.1-1.0 1.0-20.0 4.0-14.0 14.0-20.0".
- `--output`: Choose which RMS data to compute. Default: "displacement".

You can also use the *download_seismic_data.ipynb* notebook to do the same thing.

Then, adapt the notebooks *change_point_detection_ppsd.ipynb* and *change_point_detection_rms.ipynb* to use the data you just collected and have fun!

## Documentation

A complete documentation is available in the *doc/src/* folder. If it is not
generated, you can run from the root folder:

```bash
pip3 install pdoc3
python3 -m pdoc -o doc/ --html --config latex_math=True --force src/
```

Then, open *doc/src/index.html* in your browser and follow the guide!

[//]: # (References)

[ruptures]: https://github.com/deepcharles/ruptures
[obspy]: https://github.com/obspy/obspy
