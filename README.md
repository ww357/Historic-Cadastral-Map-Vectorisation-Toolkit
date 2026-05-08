This is a toolking for comprehensive semi-automated vectorisation of cadastral map series with a built-in feedback loop to iteratively improve prediction capabilities and reduce mending effort over time. Each feature is an individual binary segmentation model, and any combination of features can be selected and comipled from map patch predictions into a GeoPackage of full map vector layers.

Initial setup:

```python
# these are temporary instructions for now before final environments are subbed in.
conda env create -f envs/maptools.yml
conda activate maptools
        #Not yet in envs: cond env create -f envs/tf-gpu.yml
        #Not yet in envs: cond env create -f envs/New-MapReader.yml
```

Running through the pipeline:
```python

## Step 01 - Patchify

conda run -n maptools python "steps/01_ patchify/draw_mask.py" --sheet MapSheetName
# run this to interactively make mask of map area on the document if necessary 
# to reduce patches for inference (or this mask can be made in another programme):
conda run -n maptools python "steps/01_ patchify/patchify.py" --sheet MapSheetName
# slice GeoTIFF into 512px patches (use --mask flag if no mask is auto-found)

## Step 02 - Annotate
conda run -n maptools python "steps/02_annotate/annotate.py" --sheet MapSheetName
# open labelme to draw boundary lines and feature polygons
conda run -n maptools python "steps/02_annotate/export_masks.py" --sheet MapSheetName
# convert labelme JSON to binary mask PNGs per feature label

## Step 03 - Fine-tune (Boundaries - U-Net)
conda run -n tf-gpu python "steps/03_finetune/boundaries/train.py" --sheet MapSheetName --name map_v1
# fine-tune boundary U-Net, checkpoints on path-F1

## Step 03 - Fine-tune (Features - MapSAM)
conda run -n MapSAM python "steps/03_finetune/MapSAM/train.py" --sheet MapSheetName --feature FeatureName --name map_v1 #CHECK THIS NAME FLAG STILL WORKS!
# fine-tune SAM DoRA weights for one feature class (repeat per feature)

## Step 04 - Predict
conda run -n tf-gpu python "steps/04_predict/boundaries/predict.py" --sheet MapSheetName
# run U-Net on all patches, skips manually annotated ones

conda run -n MapSAM python "steps/04_predict/MapSAM/predict.py" --sheet MapSheetName --feature FeatureName
# run MapSAM on all patches for one feature (repeat per feature)

## Step 05 - Vectorise
conda run -n maptools python "steps/05_vectorise/boundaries/vectorise.py" --sheet MapSheetName
# stitch boundary patches + skeletonise → polylines → GeoPackage
conda run -n maptools python "steps/05_vectorise/features/vectorise.py" --sheet MapSheetName --feature FeatureName
# stitch feature patches + polygonise → polygons → GeoPackage (repeat per feature)

## Step 06 - Text
conda activate New-MapReader
python "steps/06_text/predict.py" --sheet MapSheetName
conda activate maptools
python "steps/06_text/fix_text_layer.py" --sheet MapSheetName


# Output: data/outputs/MapSheetName.gpkg
```

## Acknowledgements

The boundary U-Net architecture is based on:

> Ran et al. (2022). Raster Map Line Element Extraction Method Based on Improved U-Net Network.
> *ISPRS International Journal of Geo-Information*, 11(8), 439.
> https://doi.org/10.3390/ijgi11080439
> GitHub: https://github.com/FutureuserR/Raster-Map

**MapSAM** — feature segmentation model used for buildings, water, and other polygon features.

> Xue Xia, Daiwei Zhang, Wenxuan Song, Wei Huang, Lorenz Hurni.
> *MapSAM: Adapting Segment Anything Model for Automated Feature Detection in Historical Maps.*
> https://github.com/xiaxue-ethz/MapSAM

**MapTextPipeline** — text detection and recognition pipeline used in step 06.

> Based on DNTextSpotter: Yu Xie et al.
> *DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training.*
> arXiv:2408.00355 (2024). https://github.com/yyyyyxie/DNTextSpotter
>
> MapText fork (maps-as-data): https://github.com/maps-as-data/MapTextPipeline

**MapReader** — map patch management and MapTextPipeline runner used in step 06.

> maps-as-data / The Alan Turing Institute.
> https://github.com/maps-as-data/MapReader
