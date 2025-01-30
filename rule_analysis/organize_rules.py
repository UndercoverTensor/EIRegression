import os
import shutil
import re


def organize_files_by_buckets(folder_path):
    pattern = re.compile(r'rule_results_(\d+)_buckets_\d+_iterations.*')

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            bucket_value = int(match.group(1))
            if 2 <= bucket_value <= 13:
                target_folder = f"{bucket_value}_buckets"
                target_path = os.path.join(folder_path, target_folder)

                os.makedirs(target_path, exist_ok=True)

                shutil.move(os.path.join(folder_path, filename), os.path.join(target_path, filename))
            else:
                os.remove(os.path.join(folder_path, filename))
