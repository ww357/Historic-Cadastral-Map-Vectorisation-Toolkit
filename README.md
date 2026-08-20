
This is a toolkit for comprehensive semi-automated vectorisation of cadastral map series with a built-in feedback loop to iteratively improve prediction capabilities and reduce mending effort over time. Each feature is an individual binary segmentation model, and any combination of features can be selected and compiled from map patch predictions into a GeoPackage of full map vector layers. Land parcels are derived separately, by partitioning the boundary-line network using the recorded apportionment points as seeds (a point-seeded watershed), so no per-parcel model training is required.

The pipeline was initially developed for use with the [Tithe Maps of England and Wales]() to target:
- Solid boundary lines between land parcels with a modified version of [this improved U-Net architecture]() and model weights that have only been trained on tithe map training data (~300 map patches across 12(?) sample maps). Symbology such as building outlines and watercourses that are not land parcel boundaries, yet are morphologically identical, were also annotated to avoid bad prediction from the model's lack of additional semantic knowledge.
- Dashed bounary lines using a custom-made U-Net architecture with a pretrained EfficientNetB0 encoder that learns to redraw dashed boundary lines as if they were continuous using a Gaussian-blurred version of the linear annotation training set. 
- Water (watercourses and waterbodies), land cover symbology (all different symbology together in one class to account for inter-map sheet variability requiring post-prediction manual classification aka filling in an attribute field), and house footprints with [MapSAM](https://arxiv.org/abs/2411.06971). Derived from SAM and fine-tuned for the historic map domain, minimal few-shot fine-tuning is required to produce high-accuracy prediction for each class. 
- Text with the use of the [MapTextPipeline runner](https://github.com/maps-as-data/MapTextPipeline) with model weights built from David Rumsey historical map collection ([add link to these]()). 



Download's and Pre-requisites:
- This pipeline was built in Windows Subsystem for Linux (WSL) because of package and library agreements. 
- The anaconda environments are set up to work best with an NVIDIA GPU.
- Set up the following conda environments:
```python
conda env create -f envs/maptools.yml   # annotation, patchify, and vectorising
conda env create -f envs/polygons.yml   # used for both polygon (MapSAM) and text steps 
conda env create -f envs/lines.yml      # used for both solid and dashed line steps (U-Net models)
Running through the pipeline:
```
- Clone the necessary repositories into the `models/` folder
- Add model weights into the `models/base/` folder from [my huggingface](https://huggingface.co/ww357)

File structure:
``` bash
repo root/
│
├── data/
│   ├── raw/<SHEET>/<SHEET>.tif        ← [YOU] GeoTIFF map scan
│   ├── parcel_points/*.gpkg            ← [YOU] land parcel centroids
│   │
│   ├── map_area_masks/                 (drawn by step 00)
│   ├── patches/                        (created by step 01)
│   ├── annotations/                    (created by step 02)
│   ├── predictions/                    (created by steps 03-04)
│   └── outputs/<SHEET>.gpkg            (final GeoPackage created in step 05)
│
├── models/
│   ├── ImprovedLinearUNet/             ← [YOU] clone from GitHub (https://github.com/ww357/Improved-Linear-U-Net)
│   ├── MapSAM/                         ← [YOU] clone from GitHub (https://github.com/ww357/MapSAM)
│   ├── MapTextPipeline/                ← [YOU] clone from GitHub (https://github.com/maps-as-data/MapTextPipeline)
│   │
│   ├── base/
│   │   ├── unet/model_weights.weights.h5
│   │   ├── MapSAM/origional_weights/sam_vit_b_01ec64.pth  
│   │   └── MapTextPipeline/rumsey-finetune.pth      
│   │
│   └── finetuned/                      (generated model weights will land here in steps 03_finetune and 06_feedback)
│
├── steps/                              (pipeline scripts — do not edit)
├── envs/maptools.yml                   ← use to build the maptools conda env
└── config.yaml                         ← pipeline parameters
```

Running through the pipeline:
```python
## Step 01 - Patchify
conda activate maptools 
python "steps/01_patchify/draw_mask.py" --sheet MapSheetName
# run this to interactively make mask of map area on the document if necessary 
# to reduce patches for inference (or this mask can be made in another programme):
python "steps/01_patchify/patchify.py" --sheet MapSheetName
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
python "steps/03_finetune/polygons/train.py" --sheet MapSheetName --feature FeatureName --name map_v1
# fine-tune SAM DoRA weights for one feature class (repeat per feature)

## Step 04 - Predict
conda activate lines
python "steps/04_predict/lines/predict.py" --sheet MapSheetName
# run U-Net on all patches, skips manually annotated ones
conda activate polygons
python "steps/04_predict/polygons/predict.py" --sheet MapSheetName --feature FeatureName1 FeatureName2 FeatureName3 ...
# run MapSAM on all patches for listed features & run text prediction if "text" is specified

## Step 05 - Vectorise
conda activate maptools
python "steps/05_vectorise/lines/vectorise.py" --sheet MapSheetName # stitch and vectorise boundary lines
python "steps/05_vectorise/polygons/vectorise.py" --sheet MapSheetName # stitch and vectorise polygons
python "steps/05_vectorise/text/text_to_vector.py" --sheet MapSheetName # stitch and vectorise text

## Parcels - point-seeded watershed (run after the boundary lines are vectorised)
conda activate polygons
python "steps/04_predict/parcels/parcel_segment.py" --sheet MapSheetName
# partition the stitched boundary raster into parcels, seeded by apportionment points
python "steps/05_vectorise/parcels/parcel_vectorise.py" --sheet MapSheetName
# join apportionment attributes and write the 'parcels' layer to the GeoPackage

## Step 06 - Feedback loop
conda activate lines
python "steps/06_feedback/lines/feedback.py" --sheet MapSheetName

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