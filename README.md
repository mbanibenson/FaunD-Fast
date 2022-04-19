# deepsea-fauna-detection
This repository contains the source codes used for semi-automatic detection of potential benthic megafauna on the seafloor. The workflow is based on the processing of a sequence of underwater optical images, and operates on the observation that fauna occurs infrequently in most photos. On the basis of this observation, we implemented the deepsea fauna detection workflow as follows:

## Training

1. A subset of 400 images is sampled randomly from the entire image dataset collected during the dive/deployment.

2. The images are segmented to extract image patches whose pixels are characteristically similar to each other. 

3. A Convolution Variational Auto Encoder(CVAE) is then trained and used to extract features from these image patches.

4. Isolation Forest algorithm is then trained to detect anomalous image patches based on the extracted features.

## Inference

1. Each image is segmented to extract image patches whose pixels are characteristically similar to each other.

2. The trained CVAE is then used to extract features from the image patches.

3. The trained Isolation Forest algorithm is used to detect any anomalous image patches.

4. Each detected anomalous patche is saved to disk, while its details are saved in a seperate csv file. Details include the parent image name and the bounding box coordinates marking their location within the parent image.
