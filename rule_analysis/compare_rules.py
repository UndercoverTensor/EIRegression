import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from .organize_rules import organize_files_by_buckets


def compute_iou(rules1, rules2, top_k):
    top_rules1 = set(rules1[:top_k])
    top_rules2 = set(rules2[:top_k])
    if not top_rules1 and not top_rules2:
        return 1.0

    intersection = len(top_rules1 & top_rules2)
    union = top_k
    return intersection / union if union != 0 else 0


def parse_rules(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    rules_per_class = {}
    current_class = None

    for line in lines:
        line = line.strip()
        if line.startswith('---') and line.endswith('---'):
            try:
                current_class = int(line.strip('-'))
                rules_per_class[current_class] = []
            except ValueError:
                current_class = None
        elif line.startswith('rule') and current_class is not None:
            rule_name = line.split(':')[0].strip()
            rules_per_class[current_class].append(rule_name)

    return rules_per_class


def calculate_matrix(file_path, top_k):
    rules_per_class = parse_rules(file_path)
    classes = list(rules_per_class.keys())
    if not classes:
        return None
    iou_matrix = np.zeros((len(classes), len(classes)))

    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i == j:
                iou_matrix[i, j] = 1.0
            else:
                iou_matrix[i, j] = compute_iou(rules_per_class[class1], rules_per_class[class2], top_k)

    return iou_matrix, classes

def setup_directories(out_base):
    if not os.path.exists(out_base):
        os.makedirs(out_base)
    matrices_dir = os.path.join(out_base, "matrices")
    images_dir = os.path.join(out_base, "images")
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    return matrices_dir, images_dir


def process_bucket_directory(bucket_directory_path, top_k):
    matrices = []
    class_labels = None
    for filename in os.listdir(bucket_directory_path):
        file_path = os.path.join(bucket_directory_path, filename)
        if os.path.isfile(file_path):
            matrix, classes = calculate_matrix(file_path, top_k)
            if matrix is not None:
                matrices.append(matrix)
                if class_labels is None:
                    class_labels = classes
    return matrices, class_labels


def save_matrix(matrix, output_path):
    np.save(output_path, matrix)
    # print(f"Matrix saved to {output_path}")


def plot_heatmap(matrix, class_labels, title, image_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels, cmap='viridis')
    plt.title(title)
    plt.xlabel("Buckets/Classes")
    plt.ylabel("Buckets/Classes")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    # print(f"Heatmap saved to {image_path}")


def process_rule_similarities(rules_dir, dataset_name, out_base, top_k=5):
    # Setup output directories
    matrices_dir, images_dir = setup_directories(out_base)

    # Organize files by buckets
    organize_files_by_buckets(rules_dir)

    # Process each bucket directory
    for i in range(2, len(os.listdir(rules_dir)) + 2):
        bucket_directory_path = os.path.join(rules_dir, f"{i}_buckets")
        if os.path.exists(bucket_directory_path):
            matrices, class_labels = process_bucket_directory(bucket_directory_path, top_k)
            
            if matrices:
                avg_matrix = np.mean(matrices, axis=0)
                title = f"Average Rule Similarity (Top {top_k} rules) - on {dataset_name} ({i} Buckets)"
                matrix_filename = f"{dataset_name}_{i}_buckets.npy"
                image_filename = f"{dataset_name}_{i}_buckets.png"
                
                # Save the matrix
                save_matrix(avg_matrix, os.path.join(matrices_dir, matrix_filename))
                
                # Plot and save the heatmap
                plot_heatmap(avg_matrix, class_labels, title, os.path.join(images_dir, image_filename))


def main(directory_path, top_k, title, out):
    matrices = []
    class_labels = None

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            matrix, classes = calculate_matrix(file_path, top_k)
            if matrix is not None:
                matrices.append(matrix)
                if class_labels is None:
                    class_labels = classes

    if not matrices:
        print("No valid matrices found.")
        return

    avg_matrix = np.mean(matrices, axis=0)

    head, tail = os.path.split(out)
    matrice_path = os.path.join(head, 'matrices', tail)

    np.save(matrice_path, avg_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_matrix, annot=True, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels, cmap='viridis')
    plt.title(title)
    plt.xlabel("Buckets/Classes")
    plt.ylabel("Buckets/Classes")
    image_path = os.path.join(head, 'images', tail)
    plt.savefig(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse parameters for rule similarity experiments.")

    parser.add_argument('--rules_dir', type=str, help="Path to the rules directory")
    parser.add_argument('--top_k', type=int, default=5, help="Top K rules to select")
    parser.add_argument('--method', type=str, help="Regressor name")
    parser.add_argument('--dataset', type=str, help="Name of the dataset")
    parser.add_argument('--out_base', type=str, help="Output base directory")

    args = parser.parse_args()

    if not os.path.exists(args.out_base):
        os.makedirs(args.out_base)
    matrices_dir = os.path.join(args.out_base, "matrices")
    images_dir = os.path.join(args.out_base, "images")
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    organize_files_by_buckets(args.rules_dir)
    for i in range(2, len(os.listdir(args.rules_dir)) + 2):
        bucket_directory_path = os.path.join(args.rules_dir, str(i) + "_buckets")
        if os.path.exists(bucket_directory_path):
            bucket_title = args.method + "_" + args.dataset + "_" + str(i) + "_buckets"
            out = os.path.join(args.out_base, bucket_title)
            title = (f"Average Rule Similarity (Portion of rules present in both buckets)\n"
                     f"in Top {args.top_k} rules, {args.method} on {args.dataset} Dataset {i} Buckets")
            main(bucket_directory_path, args.top_k, title, out)
