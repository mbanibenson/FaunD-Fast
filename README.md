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

Anomalous image patches alongside their bounding box coordinates are detected and extracted from each test image as follows:

1. The test image is segmented to extract image patches whose pixels are characteristically similar to each other.

2. The trained CVAE is used to extract features from the extracted image patches.

3. The trained Isolation Forest algorithm is used to detect any anomalous image patches.

4. Each detected anomalous patch is ordered by its anomalous score and saved to the directory data/unsupervised/dive/predictions/ordered_patches.

5. A csv file is created for each dive which records details about all the detected anomalous patches. These include the parent image name and the bounding box coordinates marking the location of the patch within the parent image.
```
python perform_inference_to_detect_outlier_patches.py
```


## Fauna/non-fauna classification

The image patches flagged as anomalous contain a huge number of false positives (i.e image patches flagged as anomalous which are infact seafloor). This was by design since we would rather retrieve all anomalous patches together with a some seafloor patches instead of detecting a few pathces that are obviously anomalous while missing those which are only subtly anomalous.

Therefore, a cnn classifier was configured to help sort the retrieved anomalous image patches into fauna (true positives) and non-fauna(false positives). In order to train this classifier:

1. A few training examples of each class should be selected from the directory containing detected anomalous patches and copied to the respective folder in data/supervised_fauna_non_fauna. You can use e.g xnview to browse the images.

2. The CNN is trained using the training examples, after which it is used to classify all the detected anomalous patches. On the basis of their assigned class, each image patch is automatically sorted into either fauna or non-fauna folder. The classification results can be found in data/supervised_fauna_non_fauna/predictions.
```
python auto_sort_anomalous_patches_into_fauna_non_fauna.py
```

## Semantic Annotation

After classification above, the pure fauna image patches will be located in data/supervised_fauna_non_fauna/predictions/fauna. All we have to do is semantically annotate each of them to a specific morphotype class. For this task, we have developed a simple web based tool for rapid annotation.

1. Start by importing the the fauna image patches into the custom image viewer/annotation tool.
```
python copy_files_to_image_viewer.py
```
2. Click on each image patch and annotate with a semantic morphotype class.

3. After annotation, a csv file is created with details about all the annotated fauna patches. These include the parent image name, bounding box coordinates and the annotated morphotype class name.



## Training an object detector using the annotations
We now have sufficient annotations to train a state-of-the art mask R-CNN object detector as follows:

1. Repurpose the annotations csv file to a format that can be ingested by tensorflow tfrecords utilities
```
python generate_csv_for_loading_training_data_into_tfrecords.py
```

2. Use the repurposed csv to convert all input datasets and annotations to tensorflow tfrecord format
```
python generate_tfrecords_and_input_files_for_object_detection.py
```

3. Train the state-of-the art mask R-CNN object detector
```
python train_mask_rcnn_object_detector.py
```

4. Use the trained mask R-CNN object detector to detect objects from all your images.
```
python detect_objects_in_image.py
```

## Species distribution mapping

1. Each annotated patch is assigned the georeferenced coordinates of its parent image.

2. These coordinates are used to map the spatial distribution of the detected fauna color coded by the species name.

3. Seafloor bathymetry and its derivatives are also incorporated into the mega-fauna species distribution maps.
