import os
import math
import random

import torch.multiprocessing as mp

import dgl
from dgl.data import CoraFullDataset, FraudAmazonDataset, FraudYelpDataset
from sklearn import tree
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.model_selection import train_test_split

from collections import Counter, deque, defaultdict

from statistics import median

import networkx as nx
from networkx.algorithms.isomorphism import MultiDiGraphMatcher

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import time
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor, AutoImageProcessor, AutoModel
from torchvision.ops import sigmoid_focal_loss

device0 = torch.device('cpu')  # Rule attention
device1 = torch.device('cpu')  # Structural attention
device2 = torch.device('cpu')  # GAT layer
device_vit = torch.device("cpu")
device_org = torch.device("cpu")

# Load model and processor
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', use_fast=True)
vit_model = AutoModel.from_pretrained('google/vit-base-patch16-224').to(device_vit)

min_class_label = 1
maj_class_label = 0


def getClassificationMetrics(metrics, file):
    with open(file, "a") as f:
        for m in metrics:
            print('Precision (0): ' + str(m['0']['precision']) + "\n")
            print('Recall (0): ' + str(m['0']['recall']) + "\n")
            print('F1-Score (0): ' + str(m['0']['f1-score']) + "\n")
            print('Support (0): ' + str(m['0']['support']) + "\n")

            print('Precision (1): ' + str(m['1']['precision']) + "\n")
            print('Recall (1): ' + str(m['1']['recall']) + "\n")
            print('F1-Score (1): ' + str(m['1']['f1-score']) + "\n")
            print('Support (1): ' + str(m['1']['support']) + "\n")

            print('Accuracy: ' + str(m['accuracy']) + "\n")

            print('Precision (macro avg): ' + str(m['macro avg']['precision']) + "\n")
            print('Recall (macro avg): ' + str(m['macro avg']['recall']) + "\n")
            print('F1-Score (macro avg): ' + str(m['macro avg']['f1-score']) + "\n")

            print('Precision (weighted avg): ' + str(m['weighted avg']['precision']) + "\n")
            print('Recall (weighted avg): ' + str(m['weighted avg']['recall']) + "\n")
            print('F1-Score (weighted avg): ' + str(m['weighted avg']['f1-score']) + "\n")

            f.write('Precision (0): ' + str(m['0']['precision']) + "\n")
            f.write('Recall (0): ' + str(m['0']['recall']) + "\n")
            f.write('F1-Score (0): ' + str(m['0']['f1-score']) + "\n")
            f.write('Support (0): ' + str(m['0']['support']) + "\n")

            f.write('Precision (1): ' + str(m['1']['precision']) + "\n")
            f.write('Recall (1): ' + str(m['1']['recall']) + "\n")
            f.write('F1-Score (1): ' + str(m['1']['f1-score']) + "\n")
            f.write('Support (1): ' + str(m['1']['support']) + "\n")

            f.write('Accuracy: ' + str(m['accuracy']) + "\n")

            f.write('Precision (macro avg): ' + str(m['macro avg']['precision']) + "\n")
            f.write('Recall (macro avg): ' + str(m['macro avg']['recall']) + "\n")
            f.write('F1-Score (macro avg): ' + str(m['macro avg']['f1-score']) + "\n")

            f.write('Precision (weighted avg): ' + str(m['weighted avg']['precision']) + "\n")
            f.write('Recall (weighted avg): ' + str(m['weighted avg']['recall']) + "\n")
            f.write('F1-Score (weighted avg): ' + str(m['weighted avg']['f1-score']) + "\n")


def find_transformer_head_size(patternvector, max=8):
    for n in range(1, 5):
        candidate = max - n
        if patternvector.size(1) % candidate == 0:
            return candidate
    return 1


def find_frequent_patterns(neighbor_labels):
    most_frequent = {}
    for label in neighbor_labels:
        if str(label) in most_frequent:
            most_frequent[str(label)] += 1
        else:
            most_frequent[str(label)] = 1

    for k in most_frequent.keys():
        most_frequent[k] = most_frequent[k] / len(neighbor_labels)

    return most_frequent


def combine_images_per_cluster(image_folder='Images/Cluster', images_per_row=4, max_dim=65500):
    def process_cluster(cluster_prefix, tag):
        clusters = {}
        for fname in os.listdir(image_folder):
            if fname.endswith('.jpg') and fname.startswith(cluster_prefix):
                parts = fname.split('_')
                cluster_id = parts[0].replace(cluster_prefix, '')
                clusters.setdefault(cluster_id, []).append(os.path.join(image_folder, fname))

        for cluster_id, files in clusters.items():
            images = [Image.open(f) for f in files]
            if not images:
                continue

            img_width, img_height = images[0].size
            num_images = len(images)
            images_per_col = max_dim // img_height
            max_images_per_page = images_per_row * images_per_col

            for page_num in range(math.ceil(num_images / max_images_per_page)):
                start_idx = page_num * max_images_per_page
                end_idx = min(start_idx + max_images_per_page, num_images)
                page_images = images[start_idx:end_idx]

                page_rows = math.ceil(len(page_images) / images_per_row)
                combined_img = Image.new(
                    'RGB',
                    (images_per_row * img_width, page_rows * img_height),
                    color='white'
                )

                for idx, img in enumerate(page_images):
                    row = idx // images_per_row
                    col = idx % images_per_row
                    combined_img.paste(img, (col * img_width, row * img_height))

                suffix = f"_p{page_num + 1}" if page_num > 0 else ""
                combined_path = os.path.join(
                    image_folder,
                    f"{tag}_Cluster{cluster_id}_combined{suffix}.jpg"
                )
                combined_img.save(combined_path)

    # Process majority clusters
    process_cluster('MajCluster', 'Maj')
    # Process minority clusters
    process_cluster('MinCluster', 'Min')

    print("Finished combining cluster images.")


def graph_to_image(graph, start_id, text='image.jpg', print_pic=False):
    """Converts a NetworkX (Multi)Graph to a PIL image in memory (optimized)."""
    node_colors = []
    for node, attributes in graph.nodes(data=True):
        if node == start_id:
            node_colors.append("gray")
        elif "train_label" in attributes:
            label = attributes["train_label"].item()
            if label == min_class_label:
                node_colors.append("red")
            elif label == maj_class_label:
                node_colors.append("green")
            else:
                node_colors.append("yellow")
        else:
            node_colors.append("yellow")

    plt.clf()

    pos = {}
    layer_map = defaultdict(list)  # layer_index -> list of node IDs

    if start_id in graph.nodes:
        try:
            # Breadth-first search to assign layers (hop distance from start_id)
            visited = set()
            queue = deque([(start_id, 0)])  # (node, layer)

            while queue:
                current_node, layer = queue.popleft()
                if current_node in visited:
                    continue
                visited.add(current_node)
                layer_map[layer].append(current_node)

                for succ in graph.successors(current_node):
                    if succ not in visited:
                        queue.append((succ, layer + 1))

                for pred in graph.predecessors(current_node):
                    if pred not in visited:
                        queue.append((pred, layer - 1))

            # Assign positions by layer
            for layer, nodes_in_layer in layer_map.items():
                for i, node in enumerate(nodes_in_layer):
                    pos[node] = (layer, i)

            # Assign default position to remaining unvisited nodes
            for node in graph.nodes:
                if node not in pos:
                    pos[node] = (100, len(pos))  # Push far right out of view

            for node, p in pos.items():
                if not np.isfinite(p).all():
                    raise ValueError("Invalid node position detected.")
        except Exception as e:
            print(f"Custom layout failed – falling back to spring_layout: {e}")
            pos = nx.spring_layout(graph)
    else:
        pos = nx.spring_layout(graph)

    # --- Drawing the graph ---
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1200)

    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_weights = {}

        for u, v, key in graph.edges(keys=True):
            u_int, v_int = int(u), int(v)
            edge_key = u_int, v_int
            edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1

        # Draw single edge per pair, with thickness proportional to edge count
        for (u, v), count in edge_weights.items():
            width = 1 + count
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v)],
                width=width,
                arrows=graph.is_directed(),
                arrowstyle='-|>' if graph.is_directed() else '-',
                arrowsize=40,
                min_source_margin=15,
                min_target_margin=15
            )
    else:
        nx.draw_networkx_edges(
            graph, pos,
            arrows=graph.is_directed(),
            arrowstyle='-|>' if graph.is_directed() else '-',
            arrowsize=40,
            min_source_margin=15,
            min_target_margin=15)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='jpg', bbox_inches='tight')
    img_buf.seek(0)
    img = Image.open(img_buf)

    if print_pic:
        plt.savefig(text, bbox_inches='tight')

    return img


def get_vit_embedding(image):
    """Generates a normalized ViT embedding for a PIL image."""
    global vit_model

    # Avoid keeping tensors on GPU longer than needed
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(vit_model.device, non_blocking=True) for k, v in inputs.items()}

        outputs = vit_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.cpu()


def run_dbscan(points, pic_dict, min_or_maj, max_cluster, no_cluster=20):
    index_to_pic = {i: pic_dict[i] for i in range(len(points))}
    points_array = np.array(points)
    output_folder = 'Images/Cluster'
    os.makedirs(output_folder, exist_ok=True)

    if min_or_maj == 'Min':
        min_cluster_size = 2
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster)
        labels = hdbscan.fit_predict(points_array)

        clusters = {}
        for idx, label in enumerate(labels):
            if label != -1:
                clusters.setdefault(label, []).append(idx)  # Store index instead of point

        cluster_mean_list = {}
        for cluster_id, cluster_indices in clusters.items():
            print(f"The {min_or_maj} cluster {cluster_id} contains {len(cluster_indices)}")

            for idx in cluster_indices:
                filename = (
                    f'Images/Cluster/{min_or_maj}Cluster{cluster_id}_{idx}.jpg'
                )
                index_to_pic[idx].save(filename)

            cluster_points = points_array[cluster_indices]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_mean_list[cluster_id] = cluster_mean

        print(f"\nWe found {len(cluster_mean_list)} clusters.")

        if len(cluster_mean_list) > 0:
            combine_images_per_cluster()

    elif min_or_maj == 'Maj':
        current_number_of_cluster = float('inf')
        if no_cluster < 5:
            no_cluster = 10

        i = 1
        step = 5

        while current_number_of_cluster > no_cluster and i < 100:
            min_cluster_size = i * step
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(points_array)

            clusters = {}
            for idx, label in enumerate(labels):
                if label != -1:
                    clusters.setdefault(label, []).append(idx)

            cluster_mean_list = {}  # Changed variable name
            for cluster_id, indices in clusters.items():
                # Calculate the mean of the points in the current cluster
                cluster_points = points_array[indices]
                cluster_mean = np.mean(cluster_points, axis=0)
                cluster_mean_list[cluster_id] = cluster_mean

            print(f"\nmin_cluster_size={min_cluster_size} → {len(cluster_mean_list)} clusters.")
            for cluster_id, indices in clusters.items():
                print(f"Cluster {cluster_id} has {len(indices)} items.")
                for idx in indices:
                    filename = f'Images/Cluster/{min_or_maj}Cluster{cluster_id}_{idx}.jpg'
                    index_to_pic[idx].save(filename)

            print(f"\nWe found {len(cluster_mean_list)} clusters.")

            if len(cluster_mean_list) > 0:
                combine_images_per_cluster()

            current_number_of_cluster = len(cluster_mean_list)
            i += 1

    return cluster_mean_list


def fine_tune_vit(maj_image_lib, min_image_lib, val_image_lib, val_label_lib, patience=10):
    import copy

    global vit_model

    # Define GPU device inside the function
    device_vit_training = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move global model to GPU temporarily
    vit_model.to(device_vit_training)

    # Create classification head
    classification_head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(vit_model.config.hidden_size, 2)
    ).to(device_vit_training)

    optimizer = torch.optim.AdamW(
        list(vit_model.parameters()) + list(classification_head.parameters()),
        lr=1e-5
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    vit_model.train()
    classification_head.train()

    dur = []
    num_epochs = 200
    batch_size = 64

    all_images = min_image_lib + maj_image_lib
    all_labels = [min_class_label] * len(min_image_lib) + [maj_class_label] * len(maj_image_lib)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        if epoch == 0:
            t0 = time.time()

        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        all_images[:], all_labels[:] = zip(*combined)

        for i in range(0, len(all_images), batch_size):
            batch_images = all_images[i:i + batch_size]
            batch_labels = all_labels[i:i + batch_size]
            if not batch_images:
                continue

            pix_img = processor(images=batch_images, return_tensors="pt").to(device_vit_training)
            labels = torch.tensor(batch_labels).to(device_vit_training)

            optimizer.zero_grad()
            outputs = vit_model(**pix_img)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            normalized_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
            logits = classification_head(normalized_embedding)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        vit_model.eval()
        classification_head.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_image_lib), batch_size):
                batch_images = val_image_lib[i:i + batch_size]
                batch_labels = val_label_lib[i:i + batch_size]
                if not batch_images:
                    continue

                pix_img = processor(images=batch_images, return_tensors="pt").to(device_vit_training)
                labels = torch.tensor(batch_labels).to(device_vit_training)

                outputs = vit_model(**pix_img)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                normalized_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
                logits = classification_head(normalized_embedding)

                loss = loss_fn(logits, labels)
                val_loss += loss.item()

        vit_model.train()
        classification_head.train()

        avg_train_loss = epoch_loss / (len(all_images) / batch_size)
        avg_val_loss = val_loss / (len(val_image_lib) / batch_size)

        if epoch % 10 == 0:
            dur.append(time.time() - t0)
            print(f"Epoch {epoch:05d} | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Time(s) {np.mean(dur):.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch:05d}.")
                break

    vit_model.eval()
    classification_head.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for i in range(0, len(val_image_lib), batch_size):
            batch_images = val_image_lib[i:i + batch_size]
            batch_labels = val_label_lib[i:i + batch_size]
            if not batch_images:
                continue

            pix_img = processor(images=batch_images, return_tensors="pt").to(device_vit_training)
            labels = torch.tensor(batch_labels).to(device_vit_training)

            outputs = vit_model(**pix_img)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            normalized_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
            logits = classification_head(normalized_embedding)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    print("Classification Report on Validation Set for visual transformer:")
    print(classification_report(all_true, all_preds))

    # Move model back to CPU
    vit_model.to("cpu")
    vit_model.eval()

    # Free up GPU memory
    del optimizer, loss_fn, classification_head
    torch.cuda.empty_cache()
    print("Temporary GPU usage finished; model and memory moved back to CPU.")


def special_img_patterns(neighbor_minority, neighbor_majority, img_dict, start_id_lib, max_cluster):
    output_folder = 'Images'
    os.makedirs(output_folder, exist_ok=True)
    min_folder = '/Minority'
    os.makedirs(output_folder + min_folder, exist_ok=True)
    maj_folder = '/Majority'
    os.makedirs(output_folder + maj_folder, exist_ok=True)

    special_min = []
    special_maj = []
    min_image_lib = []
    maj_image_lib = []

    ''''
    Graph image creation
    '''
    for coun, n in tqdm(enumerate(neighbor_minority), total=len(neighbor_minority),
                        desc="Minority class pictures"):
        if n not in neighbor_majority or neighbor_minority[n] > neighbor_majority[n]:
            special_min.append(n)
            image_path = f'Images/Minority/Minority{coun}.png'
            img = graph_to_image(img_dict[n], start_id_lib[n], text=image_path, print_pic=True)
            min_image_lib.append(img)

    for coun, n in tqdm(enumerate(neighbor_majority), total=len(neighbor_majority),
                        desc="Majority class pictures"):
        if n not in neighbor_minority:
            special_maj.append(n)
            image_path = f'Images/Majority/Majority{coun}.png'
            img = graph_to_image(img_dict[n], start_id_lib[n], text=image_path, print_pic=True)
            maj_image_lib.append(img)

    print(f'There are {len(min_image_lib)} patterns in the minority class, we consider for clustering')
    print(f'There are {len(maj_image_lib)} unique patterns in the majority class, we consider for clustering')

    ''''
    Finetune ViT
    '''
    # Labels
    min_labels = [min_class_label] * len(min_image_lib)
    maj_labels = [maj_class_label] * len(maj_image_lib)

    # Combine and split into train and validation sets
    all_images = min_image_lib + maj_image_lib
    all_labels = min_labels + maj_labels

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    # Separate training images by class (for fine_tune_vit input)
    train_min_image_lib = [img for img, lbl in zip(train_imgs, train_labels) if lbl == min_class_label]
    train_maj_image_lib = [img for img, lbl in zip(train_imgs, train_labels) if lbl == maj_class_label]

    # Call updated fine-tuning function with early stopping
    fine_tune_vit(train_maj_image_lib, train_min_image_lib, val_imgs, val_labels)

    pattern_library = {}

    ''''
    Graph embedding
    '''

    emb_min = get_vit_embedding(min_image_lib)
    emb_maj = get_vit_embedding(maj_image_lib)

    for wl_hash, emb in zip(special_min, emb_min):
        pattern_library[wl_hash] = emb

    for wl_hash, emb in zip(special_maj, emb_maj):
        pattern_library[wl_hash] = emb

    ''''
    Visual Embedding Clustering 
    '''

    emb_min = run_dbscan(emb_min, min_image_lib, min_or_maj='Min', max_cluster = max_cluster)
    emb_maj = run_dbscan(emb_maj, maj_image_lib, min_or_maj='Maj', max_cluster = max_cluster, no_cluster=int(len(emb_min)))

    return emb_min, emb_maj, pattern_library


def get_k_hop_subgraph(graph, start_node_id, k, device_subgraph=device_org):
    """
    Extracts a subgraph containing all neighbors within k hops (both incoming and outgoing)
    from a given start node in a DGL graph.

    Args:
        graph (dgl.DGLGraph): The input DGL graph.
        start_node_id (int): The ID of the starting node.
        k (int): The number of hops to consider.

    Returns:
        dgl.DGLGraph: A subgraph induced by the nodes within k hops of the start node.
    """
    if not isinstance(graph, dgl.DGLGraph):
        raise TypeError("Input 'graph' must be a dgl.DGLGraph object.")
    if not isinstance(start_node_id, int) or start_node_id < 0 or start_node_id >= graph.num_nodes():
        raise ValueError("Invalid 'start_node_id'.")
    if not isinstance(k, int) or k < 0:
        raise ValueError("'k' must be a non-negative integer.")

    visited_nodes = {start_node_id}
    queue = [(start_node_id, 0)]  # (node_id, distance)

    while queue:
        current_node, distance = queue.pop(0)

        if distance < k:
            # Get outgoing neighbors
            successors = graph.successors(current_node).tolist()
            for neighbor in successors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))

            # Get incoming neighbors
            predecessors = graph.predecessors(current_node).tolist()
            for neighbor in predecessors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))

    visited_nodes_tensor = torch.tensor(list(visited_nodes), dtype=torch.int64)
    subgraph = dgl.node_subgraph(graph, visited_nodes_tensor.to(device_subgraph))

    new_start_node_id = (subgraph.ndata[dgl.NID] == start_node_id).nonzero(as_tuple=True)[0].item()

    return subgraph, new_start_node_id


def print_occurances(dictionary, dict, start_id_lib, min_or_max):
    output_folder = 'Images/Common_Pattern'
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []

    for coun, n in tqdm(enumerate(dictionary.keys()), total=len(dictionary), desc="Print most frequent Pattern"):
        if dictionary[n] * len(dictionary.keys()) > 2:
            img_path = f'{output_folder}/{min_or_max}_{int(dictionary[n] * len(dictionary.keys()))}_{coun}.png'
            graph_to_image(dict[n], start_id_lib[n], img_path, print_pic=True)
            image_paths.append(img_path)

    # Combine all images into a raster after generating
    if image_paths:
        images = [Image.open(p) for p in image_paths]
        img_w, img_h = images[0].size

        cols = int(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)

        raster_img = Image.new('RGB', (cols * img_w, rows * img_h), color='white')

        for i, img in enumerate(images):
            x = (i % cols) * img_w
            y = (i // cols) * img_h
            raster_img.paste(img, (x, y))

        raster_path = f'{output_folder}/{min_or_max}_raster.jpg'
        raster_img.save(raster_path)
        print(f"Saved raster image to: {raster_path}")


def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def empty_relevant_folders():
    empty_folder('Images/Cluster')
    empty_folder('Images/Common_Pattern')
    empty_folder('Images/Majority')
    empty_folder('Images/Minority')


def freqsubgraphmining(graph, org_idx, train_nodes, hop, min_class_label, maj_class_label, max_cluster):
    """
    Args:
        g: DGL graph.
        train_nodes: List of training node indices.
        hop: Number of hops for subgraph extraction.
        min_class_label: Label of the minority class.
        maj_class_label: Label of the majority class.
        max_cluster: Maximum amount of points in a cluster.

    Returns:
        A tuple containing lists of unique subgraphs for minority and majority classes.
        A dictionary from WL hash to embedding
    """
    empty_relevant_folders()

    neighbor_minority = []
    neighbor_majority = []

    wl_lib = {}
    start_id_lib = {}

    counter_connect = 0
    for t in tqdm(train_nodes, desc="Extract patterns"):
        prev_subgraph = None
        for n in range(1, hop + 1):
            subgraph, start_id = get_k_hop_subgraph(graph, int(org_idx[t]), k=n)
            subgraph.ndata['train_label'][start_id] = -1

            if subgraph.num_nodes() > 1:
                counter_connect += 1

            subgraph_nx = nx.DiGraph(dgl.to_networkx(subgraph.to('cpu'), node_attrs=['train_label']))
            wl_hash = nx.weisfeiler_lehman_graph_hash(subgraph_nx, node_attr='train_label')

            if prev_subgraph is None or wl_hash != prev_subgraph:
                if graph.ndata['label'][t] == min_class_label:
                    neighbor_minority.append(wl_hash)
                elif graph.ndata['label'][t] == maj_class_label:
                    neighbor_majority.append(wl_hash)

            if wl_hash not in wl_lib:
                # For Images:
                wl_lib[wl_hash] = subgraph_nx
                # For GNN
                # wl_lib[wl_hash] = subgraph
                start_id_lib[wl_hash] = start_id

            prev_subgraph = wl_hash

    # Count occurrences of each label
    neighbor_minority = find_frequent_patterns(neighbor_minority)
    # print_occurances(neighbor_minority, wl_lib, start_id_lib, 'Minority')

    neighbor_majority = find_frequent_patterns(neighbor_majority)
    # print_occurances(neighbor_majority, wl_lib, start_id_lib, 'Majority')

    most_frequent_majority = dict(sorted(neighbor_majority.items(), key=lambda item: item[1], reverse=True))

    # spec_min, spec_maj, emb_min, emb_maj, gat_model = special_patterns_gat_dgl(neighbor_minority, most_frequent_majority, wl_lib)
    emb_min, emb_maj, wl_to_img = special_img_patterns(neighbor_minority, most_frequent_majority, wl_lib,
                                                       start_id_lib, max_cluster)

    return emb_min, emb_maj, wl_to_img  # , gat_model


def get_train_label_number(graph):
    dev = graph.device
    lab = []
    for node_index in graph.nodes():
        if graph.ndata['train_mask'][node_index]:
            lab.append(graph.ndata['label'][node_index].item())
        else:
            lab.append(2)
    return torch.tensor(lab).to(dev)


def add_img_structure(g, node_index, graph_emb_min, graph_emb_maj, img_dict, hop, selection):
    min_vec = []
    maj_vec = []
    for n in range(hop, hop + 1):
        subgraph, start_node = get_k_hop_subgraph(g, node_index, k=n)
        subgraph = subgraph.to('cpu')
        subgraph.ndata['train_label'][start_node] = -1

        # Convert to NetworkX graph
        subgraph_nx = nx.DiGraph(dgl.to_networkx(subgraph, node_attrs=['train_label']))
        wl_hash = nx.weisfeiler_lehman_graph_hash(subgraph_nx, node_attr='train_label')

        # Retrieve or compute embedding
        if wl_hash in img_dict:
            node_emb = img_dict[wl_hash]
        else:
            img = graph_to_image(subgraph_nx, start_node)
            node_emb = get_vit_embedding(img).cpu().detach().numpy()
            img_dict[wl_hash] = node_emb

        # Compute distances to min cluster means
        min_vec.extend([
            np.linalg.norm(np.asarray(node_emb) - np.asarray(cluster_mean))
            for cluster_mean in graph_emb_min.values()
        ])

        # Compute distances to maj cluster means if needed
        if selection == 'maj_and_min':
            maj_vec.extend([
                np.linalg.norm(np.asarray(node_emb) - np.asarray(cluster_mean))
                for cluster_mean in graph_emb_maj.values()
            ])

    return min_vec + maj_vec


def ellipticapproach(selection, max_cluster):
    hop = 1

    dataset = dgl.data.CSVDataset('./Data')
    g = dataset[0].to(device_org)
    metrics = []

    org_ind = g.nodes()
    features = g.ndata['feat']
    label = g.ndata['label']

    ''''
    data = FraudYelpDataset()
    g = data[0]
    g = g.edge_type_subgraph(['net_rur'])
    g = dgl.add_self_loop(g)

    metrics = []

    org_ind = g.nodes()
    features = g.ndata['feat']
    label = g.ndata['label']
    '''
    # Only include labeled nodes in train and test set
    train_test_set = (label == 0) | (label == 1)
    train_test_set = torch.nonzero(train_test_set, as_tuple=True)[0]

    # KFold with 10 random splits
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Train method for different folds
    for fold, (train_index, test_index) in enumerate(kf.split(train_test_set)):
        print(f"\n===== Split {fold + 1} =====")

        # Get actual indices of nodes for training/testing
        selected_train = train_test_set[train_index]
        selected_test = train_test_set[test_index]

        train_labels = label[selected_train].tolist()
        test_labels = label[selected_test].tolist()

        print("Train class distribution:")
        print(Counter(train_labels))
        print("Test class distribution:")
        print(Counter(test_labels))

        train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(features.shape[0], dtype=torch.bool)

        for index in selected_train:
            train_mask[index] = True
        for index in selected_test:
            test_mask[index] = True
            # test_mask_org[org_ind[index]] = True
            # train_mask_org[org_ind[index]] = False

        g.ndata['train_mask'] = train_mask.to(device2)
        g.ndata['test_mask'] = test_mask.to(device2)

        train_indices_list = [index for index, value in enumerate(train_mask) if value == 1]
        test_indices_list = [index for index, value in enumerate(test_mask) if value == 1]

        g.ndata['train_label'] = get_train_label_number(g)

        ''''
        Visual Graph Representation Learning
        '''

        emb_min, emb_maj, wl_to_img = freqsubgraphmining(g, org_ind, train_indices_list, hop, min_class_label, maj_class_label, max_cluster)

        ''''
        Cluster Distance Vector Creation
        '''

        patternvector = torch.stack([
            torch.Tensor(
                add_img_structure(g, int(node_index), emb_min, emb_maj, wl_to_img, hop, selection)).to(
                device1)
            for node_index in tqdm(g.nodes(), desc="Processing Nodes")
        ]).to(device1)

        min_vals = patternvector.min(dim=1, keepdim=True)[0]
        max_vals = patternvector.max(dim=1, keepdim=True)[0]
        mean_values = torch.mean(patternvector, dim=1, keepdim=True)

        normalized_patternvector = 1 - (patternvector - min_vals) / (max_vals - min_vals)

        new_features = []
        for f_idx, f in enumerate(features):
            pa_ru = normalized_patternvector[f_idx].to(device2)
            new_features.append(torch.cat((f.to(device2), pa_ru), dim=0))

        new_features = torch.stack(new_features)

        '''
        Node Classification
        '''

        clf = RandomForestClassifier()
        clf.fit(new_features[train_mask], label[train_mask])
        clf.predict(new_features[test_mask])

        cl_rep = classification_report(
            clf.predict(new_features[test_mask]),
            label[test_mask].cpu(),
            digits=6,
            output_dict=True,
            zero_division=1
        )
        print(cl_rep)
        metrics.append(cl_rep)

    getClassificationMetrics(metrics, file + '.txt')


if __name__ == '__main__':
    selection = 'maj_and_min'
    grid_max_cluster_size = [5,10, 15, 20, 25, 30, 35]

    data_source = 'Elliptic'
    # data_source = 'FraudYelp'

    if data_source == 'Elliptic':
        min_class_label = 0
        maj_class_label = 1
    elif data_source == 'FraudYelp':
        min_class_label = 1
        maj_class_label = 0

    mp.set_start_method("spawn", force=True)

    for m in grid_max_cluster_size:

        if data_source == 'Elliptic':
            file = f"Results/Elliptic/onlyRFforClassification{m}"
        if data_source == 'FraudYelp':
            file = "FraudYelp/onlyRF"

        ellipticapproach(selection, 20)

