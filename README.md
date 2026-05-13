This is a toolking for comprehensive semi-automated vectorisation of cadastral map series with a built-in feedback loop to iteratively improve prediction capabilities and reduce mending effort over time. Each feature is an individual binary segmentation model, and any combination of features can be selected and comipled from map patch predictions into a GeoPackage of full map vector layers.

Initial setup:

```python
conda env create -f envs/annotation.yml   #primaritly for annotation
conda env create -f envs/polygons.yml   #for running MapReader-wrapped text detection and MapSAM feature detection
conda env create -f envs/lines.yml      #for running boundary line detection U-Net
```

Running through the pipeline:
```python

## Step 01 - Patchify

conda activate maptools 
python "steps/01_ patchify/draw_mask.py" --sheet MapSheetName
# run this to interactively make mask of map area on the document if necessary 
# to reduce patches for inference (or this mask can be made in another programme):
python "steps/01_ patchify/patchify.py" --sheet MapSheetName
# slice GeoTIFF into 512px patches (use --mask flag if no mask is auto-found)

## Step 02 - Annotate
python "steps/02_annotate/annotate.py" --sheet MapSheetName
# open labelme to draw boundary lines and feature polygons
python "steps/02_annotate/export_masks.py" --sheet MapSheetName
# convert labelme JSON to binary mask PNGs per feature label

## Step 03 - Fine-tune (Boundaries - U-Net)
conda activate lines
python "steps/03_finetune/lines/train.py" --sheet MapSheetName --name map_v1
# fine-tune boundary U-Net, checkpoints on path-F1

## Step 03 - Fine-tune (Features - MapSAM)
conda activate polygons
python "steps/03_finetune/polygons/train.py" --sheet MapSheetName --feature FeatureName --name map_v1 #CHECK THIS NAME FLAG STILL WORKS!
# fine-tune SAM DoRA weights for one feature class (repeat per feature)

## Step 04 - Predict
conda activate lines
python "steps/04_predict/lines/predict.py" --sheet MapSheetName #CHANGE TO LINES
# run U-Net on all patches, skips manually annotated ones
conda activate polygons
python "steps/04_predict/polygons/predict.py" --sheet MapSheetName --feature FeatureName1 FeatureName2 FeatureName3 ...
# run MapSAM on all patches for listed features & run text prediction if "text" is specified

## Step 05 - Vectorise
conda activate maptools
python "steps/05_vectorise/lines/vectorise.py" --sheet MapSheetName # stitch boundary patches + skeletonise → polylines → GeoPackage
python "steps/05_vectorise/polygons/vectorise.py" --sheet MapSheetName --feature FeatureName
python "steps/05_vectorise/text/text_to_vector.py" --sheet MapSheetName # stitch feature patches + polygonise → polygons → GeoPackage (repeat per feature)


## Step 07 - Feedback loop
conda activate lines
python "steps/07_feedback/lines/feedback.py" --sheet MapSheetName

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
