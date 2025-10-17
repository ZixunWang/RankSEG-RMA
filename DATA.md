This guidance mainly borrows from this [repo](https://raw.githubusercontent.com/zifuwanggg/JDTLosses/refs/heads/master/DATA.md).

The prepared data structure is as follows:
```
data_dir
|—— VOCdevkit
|   |—— VOC2010
|   |   |—— JPEGImages
|   |   |—— SegmentationClassContext
|   |—— VOC2012
|   |   |—— JPEGImages
|   |   |—— SegmentationClass
|   |   |—— SegmentationClassAug
|   |   |—— SegmentationClassTrainAug
|   |—— VOCaug
|—— cityscapes
|   |—— leftImg8bit
|   |   |—— train
|   |   |—— val
|   |—— gtFine
|   |   |—— train
|   |   |—— val
|—— ade
|   |—— ADEChallengeData2016
|   |   |—— images
|   |   |   |—— training
|   |   |   |—— validation
|   |   |—— annotations
|   |   |   |—— training
|   |   |   |—— validation
|—— lits
|   |—— train
|—— kits
|   |—— train
```

## PASCAL VOC
* Step 1: Download PASCAL VOC 2012 from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and extra data from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

* Step 2: Run the following from `MMSegmentation`
  ```
  python tools/dataset_converters/voc_aug.py \
    /path/to/data_dir/VOCdevkit \
    /path/to/data_dir/VOCdevkit/VOCaug
  ```

* Step 3: Run the following
  ```
  python datas/process_pascal_voc.py path/to/data_dir
  ```

## Cityscapes
* Step 1: Download the dataset from [here](https://www.cityscapes-dataset.com)
* Step 2: Run the following from `MMSegmentation`

  ```
  python tools/dataset_converters/cityscapes.py data/cityscapes
  ```
  
## ADE20K
* Download the dataset from [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)


## LiTS
* Run the following
  ```
  datas/prepare_lits_kits.py path/to/data_dir lits
  ```

## KiTS
* Run the following
  ```
  datas/prepare_lits_kits.py path/to/data_dir kits
  ```