# Semi-automated deepsea fauna detection
This repository contains the source codes used for semi-automatic detection of potential benthic megafauna on the seafloor. The workflow is based on the processing of a sequence of underwater optical images, and operates on the observation that fauna occurs infrequently in most photos. On the basis of this observation, we implemented the deepsea fauna detection workflow as follows:

<img src="https://cloud.geomar.de/s/a8xCSFffQoAbEDK/download/Fauna_detection_workflow.svg">

## Training

1. A subset of 400 images is sampled randomly from the entire image dataset collected during the dive/deployment.

2. The images are segmented to extract image patches whose pixels are characteristically similar to each other. 

3. A Convolution Variational Auto Encoder(CVAE) is then trained and used to extract features from these image patches.

4. Isolation Forest algorithm is then trained to detect anomalous image patches based on the extracted features.

## Inference

1. Each image is segmented to extract image patches whose pixels are characteristically similar to each other.

2. The trained CVAE is then used to extract features from the image patches.

3. The trained Isolation Forest algorithm is used to detect any anomalous image patches.

4. Each detected anomalous patch is saved to disk ordered by the anomalous score.

5. A csv file is created for each dive which records details about all the detected anomalous patches. These include the parent image name and the bounding box coordinates marking the location of the patch within the parent image.

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
