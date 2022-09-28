import subprocess
from pathlib import Path

model_directory = Path.cwd().parents[0] / 'fauna_detection_with_tensorflow_object_detection_api/'

#center_net_detector/


# subprocess.run(f'bash -c "source activate tf-object-detection-api; python custom_object_detection/model_main_tf2.py --pipeline_config_path={str(config_file_path)} --model_dir={str(model_directory)} --checkpoint_dir={str(checkpoint_directory)} --alsologtostderr"', shell=True)

subprocess.run(f'bash -c "tensorboard --logdir={model_directory}"', shell=True)