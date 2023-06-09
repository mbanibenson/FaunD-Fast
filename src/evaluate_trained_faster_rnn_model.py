import subprocess
from pathlib import Path

model_directory = Path.cwd().parents[0] / 'fauna_detection_with_tensorflow_object_detection_api/my_model_dir/'

config_file_path = model_directory / 'my_model.config'

checkpoint_directory = model_directory

subprocess.run(f'bash -c "source activate tf-object-detection-api; python custom_object_detection/model_main_tf2.py --pipeline_config_path={str(config_file_path)} --model_dir={str(model_directory)} --checkpoint_dir={str(checkpoint_directory)} --alsologtostderr"', shell=True)