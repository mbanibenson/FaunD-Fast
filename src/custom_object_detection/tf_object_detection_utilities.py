import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from pathlib import Path
import base64
from PIL import Image
import random
import shutil
import requests
import io
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def pre_process_csv_with_labels_to_have_one_row_per_image(path_to_csv_with_labels):
    '''
    Parse the input csv to prepare it for processing to tf example
    
    '''
    path_to_csv_with_labels = Path(path_to_csv_with_labels)
    
    df = pd.read_csv(path_to_csv_with_labels)
    
    def process_group(df_group):
        '''
        Process dataframe group to format reqired by object detector

        '''
        image_path = df_group.image_path.iat[0]
        image_name = df_group.image_name.iat[0]
        object_name = df_group.object_name.tolist()
        object_index = df_group.object_index.tolist()
        min_y = df_group.min_y.tolist()
        min_x = df_group.min_x.tolist()
        max_y = df_group.max_y.tolist()
        max_x = df_group.max_x.tolist()


        processed_df = pd.DataFrame({'image_path':image_path,
                                     'image_name':image_name,
                                     'object_name':[object_name],
                                     'object_index':[object_index],
                                     'min_y':[min_y],
                                     'min_x':[min_x],
                                     'max_y':[max_y],
                                     'max_x':[max_x]})

        return processed_df
    
    df_processed = df.groupby('image_name', as_index=False).apply(process_group).reset_index().iloc[:,2:]
    
    return df_processed


def encode_image_to_jpeg(image_file_path):
    '''
    Load an image and encode it to jpeg
    
    '''
    img = tf.keras.preprocessing.image.load_img(image_file_path)

    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    
    encoded_image_data = tf.io.encode_jpeg(
        input_arr,
        format='',
        quality=95,
        progressive=False,
        optimize_size=False,
        chroma_downsampling=True,
        density_unit='in',
        x_density=300,
        y_density=300,
        xmp_metadata='',
        name=None
    )
    
    return encoded_image_data.numpy()


def create_tf_example_from_a_row_of_csv_with_labels(row_of_csv_as_named_tuple):
    '''
    Given a row of csv with labels, generate a tf example. Use df.itertuples to generate the rows
    
    '''
    height = 1120
    width = 1680
    file_path = row_of_csv_as_named_tuple.image_path

    encoded_image_data_as_jpeg  = encode_image_to_jpeg(file_path)
    
    encoded_image_data = encoded_image_data_as_jpeg#base64.b64encode(encoded_image_data_as_jpeg)
        
    image_format = b'jpeg'
    
    xmins = row_of_csv_as_named_tuple.min_x
    xmaxs = row_of_csv_as_named_tuple.max_x
    
    ymins = row_of_csv_as_named_tuple.min_y
    ymaxs = row_of_csv_as_named_tuple.max_y
    
    classes_text = row_of_csv_as_named_tuple.object_name
    classes_text = [name.encode('utf8') for name in classes_text]
    classes = row_of_csv_as_named_tuple.object_index
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(file_path.encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(file_path.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(encoded_image_data),
          'image/format': dataset_util.bytes_feature(image_format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))
    
    return tf_example


# def generate_tfrecords_from_csv_with_labels(path_to_csv_with_labels):
#     '''
#     Given a csv with labels from image viewer, process its columns to generate trecord files
    
#     '''
#     processed_df = pre_process_csv_with_labels_to_have_one_row_per_image(path_to_csv_with_labels)
    
#     rows_of_csv_as_named_tuple = processed_df.itertuples()
    
#     tf_examples = [create_tf_example_from_a_row_of_csv_with_labels(row_of_csv) for row_of_csv in rows_of_csv_as_named_tuple]

#     random.shuffle(tf_examples)
    
#     number_of_train_examples = int(0.9 * len(tf_examples))
    
#     tf_train_examples = tf_examples[:number_of_train_examples]
    
#     tf_val_examples = tf_examples[number_of_train_examples:]
    
#     return tf_train_examples, tf_val_examples


def generate_tfrecords_from_csv_with_labels(path_to_csv_with_labels):
    '''
    Given a csv with labels from image viewer, process its columns to generate trecord files
    
    '''
    processed_df = pre_process_csv_with_labels_to_have_one_row_per_image(path_to_csv_with_labels)
    
    rows_of_csv_as_named_tuple = processed_df.itertuples()
    
    tf_examples = [create_tf_example_from_a_row_of_csv_with_labels(row_of_csv) for row_of_csv in rows_of_csv_as_named_tuple]
    
    return tf_examples


def download_config_file(directory_to_save_config_file, config_file_source_url):
    '''
    Download the configuration file for training
    
    '''
    directory_to_save_config_file = Path(directory_to_save_config_file)
        
    url = config_file_source_url
    
    config_file_name = directory_to_save_config_file / Path(url).name
    
    config_file = requests.get(url).text
    
    with open(config_file_name, 'w') as file:
        
        file.write(config_file)
        
        
def download_checkpoint_for_pretraining(detection_checkpoint_url, directory_to_save_checkpoint):
    '''
    Download a detection point for pretraining
    
    '''
    
    file_name = Path(detection_checkpoint_url).name
    
    file_save_path = directory_to_save_checkpoint / file_name
    
    directory_to_unpack = directory_to_save_checkpoint #/ 'pretrained_model'
    
    downloaded_file = tf.keras.utils.get_file(file_save_path, origin=detection_checkpoint_url)
    
    shutil.unpack_archive(filename=file_save_path, extract_dir=directory_to_unpack)
    
    unpacked_directory_name = Path(file_save_path).name.split('.')[0]
    
    pipeline_config_file = directory_to_save_checkpoint / f'{unpacked_directory_name}/pipeline.config'
    
    if pipeline_config_file.exists():
    
        shutil.copy2(pipeline_config_file, directory_to_save_checkpoint/'my_model_template.config')
    
    Path(file_save_path).unlink()
    
    return



def create_train_val_input_tfrecords(path_to_train_csv_with_labels, path_to_val_csv_with_labels, path_to_label_map, path_to_train_tfrecord_file, path_to_validation_tfrecord_file):
       
    train_writer = tf.io.TFRecordWriter(str(path_to_train_tfrecord_file))
    
    validation_writer = tf.io.TFRecordWriter(str(path_to_validation_tfrecord_file))
    
    tf_train_examples = generate_tfrecords_from_csv_with_labels(path_to_train_csv_with_labels)
    
    tf_val_examples = generate_tfrecords_from_csv_with_labels(path_to_val_csv_with_labels)
    
    print(f'Writing {len(tf_train_examples)} training examples ...')
    for tf_example in tf_train_examples:
        
        train_writer.write(tf_example.SerializeToString())

    train_writer.close()
    
    print(f'Writing {len(tf_val_examples)} validation examples ...')
    for tf_example in tf_val_examples:
        
        validation_writer.write(tf_example.SerializeToString())

    validation_writer.close()
    
######################## INFERENCE UTILS ######################################
def load_saved_model_for_inference(path_to_saved_model):
    '''
    Load the trained object detection model to be used for inference
    
    '''
    detection_model = tf.saved_model.load(path_to_saved_model)
        
    return detection_model


def load_label_map_info(label_map_path):
    '''
    Load label map data fo plotting
    
    '''
    label_map = label_map_util.load_labelmap(label_map_path)
    
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
    
    return category_index, label_map_dict


def read_image_into_a_tensor(image_file_path):
    '''
    Load an image into a numpy array
    
    '''
    image_as_ndarray = np.array(Image.open(image_file_path))
    
    image_tensor = tf.convert_to_tensor(image_as_ndarray)[tf.newaxis, ...]
    
    return image_tensor


def detect_objects_in_image(detection_model, image_tensor):
    '''
    Load image from file path and detect objects present within it.
    
    '''
    detections = detection_model(image_tensor)
    
    return detections


def visualize_detections(image_tensor, detections, category_index, score_threshold, directory_to_save_detection_figures,figname):
    '''
    Plot the detections for visualization
    
    '''
    image_np_with_detections = image_tensor.numpy()[0].copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=score_threshold,
      agnostic_mode=False)
    
    fig, ax = plt.subplots(figsize=(12,16))
    ax.imshow(image_np_with_detections)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(Path(directory_to_save_detection_figures) / f'{figname}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
def generate_matrix_of_detections(detections, score_threshold, file_path):
    '''
    Given a detection instance, process it to generate matrix of detection results
    
    '''
    
    scores  = detections['detection_scores'][0].numpy().reshape(-1,1)
    classes = detections['detection_classes'][0].numpy().reshape(-1,1)
    boxes = detections['detection_boxes'][0].numpy() * [1120, 1680, 1120, 1680]
    file_paths = np.array([str(Path(file_path))]*len(scores)).reshape(-1,1)
    file_names = np.array([str(Path(file_path).stem)]*len(scores)).reshape(-1,1)

    detection_matrix = np.hstack([file_names, scores, classes, boxes, file_paths])

    selector = (scores > score_threshold).squeeze()

    selected_detection_matrix = np.compress(selector,detection_matrix, axis=0)
    
    return selected_detection_matrix



def append_geospatial_coordinates_to_master_detection_summary_table(master_dataframe_summarizing_detections):
    '''
    Append geospatial coordinates to master dataframe summarizing detections before saving to disk
    
    '''
    path_to_table_with_georeferenced_coords_for_all_photos = Path.cwd().parents[0]/'reports/auxilliary_data/georeferenced_photo_coordinates.zip'
    
    with ZipFile(path_to_table_with_georeferenced_coords_for_all_photos) as myzip:
        
        with myzip.open('georeferenced_photo_coordinates.csv') as myfile:
            
            table_with_georeferenced_coords_for_all_photos = pd.read_csv(myfile)
            
    table_with_georeferenced_coords_for_all_photos['parent_image_name'] = table_with_georeferenced_coords_for_all_photos.Name.map(lambda x: x.split('.')[0])
            
    georeferenced_master_dataframe_summarizing_detections = pd.merge(master_dataframe_summarizing_detections, table_with_georeferenced_coords_for_all_photos, how='left', on='parent_image_name')
    
    return georeferenced_master_dataframe_summarizing_detections


def save_georeferenced_detection_results_as_csv(complete_detection_matrix, label_map_path, detection_results_csv_file_name):
    '''
    Georeference detection results and save to disk
    
    '''
    columns = ['parent_image_name', 'score', 'object_class_id', 'ymin', 'xmin', 'ymax', 'xmax', 'parent_image_path']

    detection_matrix_dataframe = pd.DataFrame(complete_detection_matrix, columns=columns)
    
    georeferenced_detection_matrix_dataframe = append_geospatial_coordinates_to_master_detection_summary_table(detection_matrix_dataframe)
    
    label_map = label_map_util.load_labelmap(label_map_path)

    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=False)
    
    label_map_dict_reversed = {v:k for k,v in label_map_dict.items()}
    
    object_class_name = georeferenced_detection_matrix_dataframe.object_class_id.map(lambda x:label_map_dict_reversed[int(float(x))])
    
    georeferenced_detection_matrix_dataframe['object_class_name'] = object_class_name
    
    georeferenced_detection_matrix_dataframe = georeferenced_detection_matrix_dataframe.loc[:,['parent_image_name', 'object_class_name', 'object_class_id', 'score', 'ymin', 'xmin', 'ymax', 'xmax', 'dive', 'time_stamp', 'Latitude', 'Longitude', 'recoded_classification', 'license_area', 'activity', 'parent_image_path']].rename(columns={'recoded_classification':'seafloor_classification'})

    georeferenced_detection_matrix_dataframe.to_csv(detection_results_csv_file_name, index=False)
    
    
    
def sample_ground_truth_images_for_annotation(directory_with_all_parent_images, directory_to_save_samples, path_to_annotations_table, n_samples):
    '''
    Sample images with the most detections for manual annotation and error reporting
    
    '''
    directory_to_save_samples = Path(directory_to_save_samples)
    
    path_to_annotations_table = Path(path_to_annotations_table)
    
    directory_with_all_parent_images = Path(directory_with_all_parent_images)
    
    df = pd.read_csv(path_to_annotations_table)
    
    number_of_detections_df = df.groupby('image_name').size().to_frame(name='number_of_detections').reset_index()
    
    sorted_number_of_detections_df = number_of_detections_df.sort_values(by='number_of_detections', ascending=False, ignore_index=True)
    
    file_names_to_sample = sorted_number_of_detections_df.loc[:n_samples, 'image_name']
    
    file_paths_to_sample = [(directory_with_all_parent_images / name) for name in file_names_to_sample]
    
    with ThreadPoolExecutor() as executor:
        
        [executor.submit(shutil.copy2, fp, directory_to_save_samples) for fp in file_paths_to_sample]

    
    


    
    