# Semi-automated deepsea fauna detection
This repository contains the source codes used for semi-automatic detection of potential benthic megafauna on the seafloor. The workflow is based on the processing of a sequence of underwater optical images, and operates on the observation that fauna occurs infrequently in most photos. On the basis of this observation, we implemented the deepsea fauna detection workflow as follows:

<img src="https://cloud.geomar.de/s/jf9MmmTA63EJqJr/preview">

## Training

1. Randomly sample a subset of 'background' images (e.g 400) from the entire image dataset collected during the dive/deployment. These will be used for training the feature extractor and outlier detection algorithm.
```
python randomly_sample_images_for_training_VAE_and_outlier_detector.py
```

2. Segment the sampled 'background' images into homogenous background image patches whose pixels are characteristically similar to each other. 
```
python segment_background_images.py
```

3. Train a Convolution Variational Auto Encoder(CVAE) using the background image patches, and use the trained CVAE to extract feature vectors from the patches.
```
python train_CVAE_and_extract_features_from_background_patches.py
```

4. Train Isolation Forest algorithm on the background feature vectors to detect anomalous image patches.
```
python train_Isolation_Forest_outlier_detector.py
```

## Inference

1. Each image is segmented to extract image patches whose pixels are characteristically similar to each other.

2. The trained CVAE is then used to extract features from the image patches.

3. The trained Isolation Forest algorithm is used to detect any anomalous image patches.

4. Each detected anomalous patch is saved to disk ordered by the anomalous score.

5. A csv file is created for each dive which records details about all the detected anomalous patches. These include the parent image name and the bounding box coordinates marking the location of the patch within the parent image.
```
python perform_inference_to_detect_outlier_patches.py
```


## Fauna/non-fauna classification

1. A few examples of fauna and non-fauna image patches are retrieved from the saved anomalous patches.

2. The retrieved patches are used to train a CNN to distinguish between fauna (true positives) and non-fauna (false positives).

3. The trained CNN classifies all the anomalous image patches, and automatically sorts them into either fauna or non-fauna folder

## Semantic Annotation

1. The fauna image patched are imported into the custom image viewer tool.

2. Expert human annotator assigns each a semantic species-level name.

3. An updated csv file is created which records details about all the annotated fauna patches. These include the parent image name, bounding box coordinates and species name.

## Species distribution mapping

1. Each annotated patch is assigned the georeferenced coordinates of its parent image.

2. These coordinates are used to map the spatial distribution of the detected fauna color coded by the species name.

3. Seafloor bathymetry and its derivatives are also incorporated into the mega-fauna species distribution maps.
