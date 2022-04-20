from pathlib import Path
from visualization.sort_patches_by_outlier_scores import save_copies_of_detected_patches_ordered_by_anomaly_score

if __name__ == '__main__':
    
    data_directory = Path.cwd().parents[0] / 'data'
    
    save_copies_of_detected_patches_ordered_by_anomaly_score(data_directory)