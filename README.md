This is a toolking for comprehensive semi-automated vectorisation of cadastral map series with a built-in feedback loop to iteratively improve prediction capabilities and reduce mending effort over time. Each feature is an individual binary segmentation model, and any combination of features can be selected and comipled from map patch predictions into a GeoPackage of full map vector layers.

Initial setup:

```python
# these are temporary instructions for now before final environments are subbed in.
conda env create -f envs/maptools.yml
conda activate maptools
```

Running through the pipeline:

```python
# from repo root in WSL, with geotools env active
rm -rf data/patches/images/Timberscombe data/patches/metadata/Timberscombe\\\_patches.csv

python "steps/01\\\_ patchify/patchify.py" --sheet Timberscombe --mask



python "steps/04\\\_ patchify/boundaries/predict.py" --sheet Timberscombe



```

