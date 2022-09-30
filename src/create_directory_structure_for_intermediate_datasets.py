import shutil
from pathlib import Path

if __name__ == '__main__':
    
    print('Creating data directory structure ...')
    
    template_directory = Path.cwd().parent
    
    directory_template_path = template_directory / 'data.zip'
    
    shutil.unpack_archive(directory_template_path, template_directory)
    
    print('Done.')