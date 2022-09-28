import subprocess
from pathlib import Path

model_directory = Path.cwd().parents[0] / 'fauna_detection_with_tensorflow_object_detection_api/my_model_dir/'

config_file_path = model_directory / 'my_model.config'

output_directory = model_directory / 'exported_model_dir'

subprocess.run(f'bash -c "source activate tf-object-detection-api; python custom_object_detection/exporter_main_v2.py --input_type=image_tensor --pipeline_config_path={str(config_file_path)} --trained_checkpoint_dir={str(model_directory)} --output_directory={str(output_directory)}"', shell=True)