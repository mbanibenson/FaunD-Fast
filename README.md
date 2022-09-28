# Semi-automated deepsea fauna detection
This repository contains the source codes used for semi-automatic detection of potential benthic megafauna on the seafloor. The workflow is based on the processing of a sequence of underwater optical images, and operates on the observation that fauna occurs infrequently in most photos. On the basis of this observation, we implemented the deepsea fauna detection workflow as follows:

<img src="https://cloud.geomar.de/s/naRyyAdqMsocW4r/preview">

## Locate the anomalous superpixels

This function operates dive-by-dive and performs the following operations:

1. Randomly samples of a subset of images to reduce computation cost

2. Segments the sampled images and crops out the superpixels into square image patches

3. Trains a variational auto encoder model (VAE)

4. Extracts the features of superpixels using the trained VAE

5. Detects anomalous superpixels with unusual visual properties

6. Visualize the superpixels in VAE embedded feature space (after PCA)

```
python extract_superpixel_features_and_detect_outliers.py
```



## Generate weak annotations from post-processed anomalous superpixels (true positives)

The image patches flagged as anomalous contain a huge number of false positives (i.e image patches flagged as anomalous which are infact seafloor). This was by design since we would rather retrieve all anomalous patches together with a some seafloor patches instead of detecting a few pathces that are obviously anomalous while missing those which are only subtly anomalous.

Therefore, a cnn classifier was configured to help sort the retrieved anomalous image patches into fauna (true positives) and non-fauna(false positives). In order to train this classifier:

1. A few training examples of each class should be selected from the directory containing detected anomalous patches and copied to the respective folder in data/supervised_fauna_non_fauna. You can use e.g xnview to browse the images.

2. The CNN is trained using the training examples, after which it is used to classify all the detected anomalous patches. On the basis of their assigned class, each image patch is automatically sorted into either fauna or non-fauna folder. The classification results can be found in data/supervised_fauna_non_fauna/predictions.
```
python auto_sort_anomalous_patches_into_fauna_non_fauna.py
```

## Assign semantic morphotype labels to the weak annotations

After classification above, the pure fauna image patches will be located in data/supervised_fauna_non_fauna/predictions/fauna. All we have to do is semantically annotate each of them to a specific morphotype class. For this task, we have developed a simple web based tool for rapid annotation.

1. Start by importing the the fauna image patches into the custom image viewer/annotation tool.
```
python copy_files_to_image_viewer.py
```
2. Click on each image patch and annotate with a semantic morphotype class.

3. After annotation, a csv file is created with details about all the annotated fauna patches. These include the parent image name, bounding box coordinates and the annotated morphotype class name.



## Train a Faster R-CNN object detection model using the Tensorflor object detection API
We now have sufficient annotations to train a state-of-the art mask R-CNN object detector as follows:

1. Repurpose the annotations csv file to a format that can be ingested by tensorflow tfrecords utilities
```
python generate_csv_for_loading_training_data_into_tfrecords.py
```

2. Use the repurposed csv to convert all input datasets and annotations to tensorflow tfrecord format
```
python generate_tfrecords_and_input_files_for_object_detection.py
```

3. Train the state-of-the art faster R-CNN object detector. Remember to upgrade the config file first.
```
python train_mask_rcnn_object_detector.py
```

4. Export the trained faster R-CNN to a saved model to be loaded and used during inference/detection.
```
python export_trained_mask_rcnn_to_saved_tensorflow_model.py
```

4. Use the exported faster R-CNN object detector to detect objects from all your images.
```
python detect_objects_in_image.py
```

## Evaluate the performance of the trained object detector

1. Evaluate the trained object detector on the ground truth annotations
```
python evaluate_trained_faster_rnn_model.py
```

## Estimate abundance, diversity and spatial distribution of morphotypes

1. Each annotated patch is assigned the georeferenced coordinates of its parent image.

2. These coordinates are used to map the spatial distribution of the detected fauna color coded by the species name.
