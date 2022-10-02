import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil


def save_copies_of_detected_patches_ordered_by_anomaly_score(data_directory):
    '''
    Save copies of detected outliers in order of anomalousness
    
    '''    
    for dive in data_directory.iterdir():

        dive_output_directory = dive / 'detection_outputs'

        if dive_output_directory.exists():
            
            dive_name = dive.name
            
            print(f'Ordering patches in dive {dive_name} based on outlierness ...')

            df = pd.read_csv(dive_output_directory/'detections_summary_table.csv')

            sorted_df = df.loc[:,['patch_name','parent_image_name', 'anomaly_score']].sort_values(by='anomaly_score', ascending=True)


            images_directory = dive_output_directory/'patches'

            sorted_images_directory = dive_output_directory/'sorted_patches'

            shutil.rmtree(sorted_images_directory, ignore_errors=True)
            sorted_images_directory.mkdir()

            anomalous_files = sorted_df.patch_name #pd.unique(sorted_df.parent_image_name).tolist()

            anomalous_files = [str(images_directory/f'{name}.png') for name in anomalous_files]

            anomalous_files_sortable_names = [str(sorted_images_directory/f'{dive_name}_patch_{i}.png') for i, name in enumerate(anomalous_files, start=1)]

            with ThreadPoolExecutor() as executor:

                [executor.submit(shutil.copy2, source_file, dest_file) for source_file, dest_file in zip(anomalous_files, anomalous_files_sortable_names)]

#             anomalous_files_as_string = '\n'.join(anomalous_files)

#             with open(dive/'outliers.txt', 'w') as file:

#                 file.write(anomalous_files_as_string)

        else:

            continue