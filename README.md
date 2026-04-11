This is a toolking for comprehensive semi-automated vectorisation of cadastral map series with a built-in feedback loop to iteratively improve prediction capabilities and reduce mending effort over time. Each feature is an individual binary segmentation model, and any combination of features can be selected and comipled from map patch predictions into a GeoPackage of full map vector layers.

Initial setup:

```python
# these are temporary instructions for now before final environments are subbed in.
conda env create -f envs/maptools.yml
conda activate maptools
```

Running through the pipeline:

```python
# from repo root in WSL
conda activate maptools
python "steps/01_patchify/patchify.py" --sheet Timberscombe --mask # remove mask if no mask provided
python "steps/02_annotate/annotate.py"  --sheet Timberscombe
python "steps/02_annotate/export_masks.py"  --sheet Timberscombe
conda activate tf-gpu
python "steps/03_finetune/train.py" --sheet Timberscombe --name finetune_v1
# auto — picks most recent *_best.weights.h5
python "steps/04_predict/boundaries/predict.py" --sheet Timberscombe
conda activate maptools
python "steps/05_stitch/boundaries/stitch.py" --sheet Timberscombe
python "steps/06_vectorise/boundaries/vectorise.py" --sheet Timberscombe
        #step/07_feedback

```

