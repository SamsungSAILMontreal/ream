# Copyright (c) 2023 UNITES Lab
# This source code is licensed under the license found in https://github.com/wazenmai/HC-SMoE.
# This code is copied from HC-SMoE largely as is.

import torch
import numpy as np


@torch.no_grad()
def hcsmoe(expert_logits: torch.Tensor,
           k: int,
           method: str='average'):
    """

    Perform hierarchical clustering using the specified linkage method.
    :param expert_logits: Tensor of shape (number of experts, n_features - model dimension)
    :param k: The number of clusters to form.
    :param method: Linkage method: 'single', 'complete', 'average'.
    :return: Cluster assignments and the ID of the expert closest to the group center
    '"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if expert_logits.dim() == 3:
        expert_logits = expert_logits.mean(1)  # (E, D)
    print("hierarchical clustering - {} to {} clusters, features = {}".format(method, k, expert_logits.shape))
    n_samples = expert_logits.shape[0]
    expert_logits = expert_logits.float().to(device)

    # Compute pairwise distances
    distances = pairwise_distances(expert_logits, method)
    pair_distances = distances.clone()

    # Initialize clusters
    clusters = torch.tensor([i for i in range(n_samples)])

    # Perform clustering
    while len(torch.unique(clusters)) > k:
        i, j, distances = linkage_step(distances, pair_distances, clusters, method, expert_logits)
        # print(f"clusters: {len(torch.unique(clusters))}, merge ({i}, {j})")
        cj = clusters[j]
        # Merge cluster j to cluster i
        clusters[clusters == cj] = clusters[i]

    # Reassign cluster IDs to be contiguous
    d = {}
    element_id = 0
    for i, idx in enumerate(clusters):
        if idx.item() not in d:
            d[idx.item()] = element_id
            element_id += 1
        clusters[i] = d[idx.item()]

    center_indices = []
    for k in range(k):
        cluster_members = expert_logits[clusters == k]
        cluster_center = cluster_members.mean(dim=0)
        distances = torch.cdist(cluster_members, cluster_center.unsqueeze(0), p=2)
        closest_expert_idx = torch.argmin(distances, dim=0).item()
        center_indices.append(torch.where(clusters == k)[0][closest_expert_idx].item())

    del distances
    clusters = clusters.cpu().numpy()
    center_indices = np.array(center_indices)
    return clusters, center_indices


@torch.no_grad()
def compute_distance(pair_distances, clusters, method='average', X=None):
    if method == 'average':
        # dist(cluster i, cluster j) = sum_{x in cluster i, y in cluster j} dist(x, y) / (|cluster i| * |cluster j|)
        cluster_labels = torch.unique(clusters)
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        # Iterate through all pairs of clusters (ci, cj)
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                dist = []
                # Iterate through all pairs of points (vi, vj) for vi in ci and vj in cj
                for vi in torch.where(clusters == ci)[0]:
                    for vj in torch.where(clusters == cj)[0]:
                        dist.append(pair_distances[vi, vj].item())
                new_dist = torch.sum(torch.tensor(dist)) / (torch.sum(clusters == ci) * torch.sum(clusters == cj))
                distances[i, j] = new_dist
                distances[j, i] = new_dist
        distances.fill_diagonal_(float('inf'))
        idx = torch.argmin(distances)
        final_i, final_j = cluster_labels[idx // distances.shape[0]], cluster_labels[idx % distances.shape[0]]
    elif method == 'ward':
        # 1. Compute the center of each cluster
        cluster_labels = torch.unique(clusters)
        cluster_centers = torch.zeros((len(cluster_labels), X.shape[1]))
        for i, cluster in enumerate(cluster_labels):
            cluster_centers[i] = X[clusters == cluster].mean(dim=0)

        # 2. Compute the distance between each pair of clusters
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                ni = torch.sum(clusters == ci)
                nj = torch.sum(clusters == cj)
                new_dist = (ni * nj) / (ni + nj) * torch.cdist(cluster_centers[i].unsqueeze(0),
                                                               cluster_centers[j].unsqueeze(0), p=2)
                distances[i, j] = new_dist
                distances[j, i] = new_dist
        distances.fill_diagonal_(float('inf'))
        idx = torch.argmin(distances)
        final_i, final_j = cluster_labels[idx // distances.shape[0]], cluster_labels[idx % distances.shape[0]]
    else:
        raise NotImplementedError("Unsupported linkage method: {}".format(method))

    return final_i, final_j


@torch.no_grad()
def pairwise_distances(X, method='single'):
    """Compute pairwise Euclidean distances between points."""
    dot_product = torch.mm(X, X.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0).sqrt()
    if method == 'single' or method == 'average':
        distances.fill_diagonal_(float('inf'))
    elif method == 'complete':
        distances.fill_diagonal_(0.0)
    return distances


@torch.no_grad()
def linkage_step(distances, pair_distances, clusters=None, method='single', X=None):
    """Perform a single step of hierarchical clustering using the specified linkage method."""
    """
    Single linkage: d(ci, cj) = min_{x in ci, y in cj} dist(x, y) -> the closest pair of points
    Complete linkage: d(ci, cj) = max_{x in ci, y in cj} dist(x, y) -> the farthest pair of points
    Average linkage: d(ci, cj) = sum_{x in ci, y in cj} dist(x, y) / (|ci| * |cj|) -> the average distance between all pairs
    Ward linkage: d(ci, cj) = (|ci| * |cj|) / (|ci| + |cj|) * dist(mu(ci), mu(cj)) -> the increase in variance when merging clusters
    """

    ### 1. Find the pair of clusters with the smallest distance
    if method == 'single':
        # d(ci, cj) = min_{x in ci, y in cj} dist(x, y)
        min_idx = torch.argmin(distances).item()
        i, j = min_idx // distances.shape[0], min_idx % distances.shape[0]
        # print(f"min_idx: {min_idx}, ({i}, {j})")
    elif method == 'complete':
        # d(ci, cj) = max_{x in ci, y in cj} dist(x, y)
        max_idx = torch.argmax(distances).item()
        i, j = max_idx // distances.shape[0], max_idx % distances.shape[0]
    else:
        i, j = compute_distance(pair_distances, clusters, method, X)

    if i > j:
        i, j = j, i

    if method == 'average' or method == 'ward':
        return i, j, distances

    ### 2. Update the distance matrix
    # We merge cluster j to cluster i, so other clusters to cluster j will be inf. (cluster j dissapears)
    # And the distance from cluster i to other clusters will be updated based on the linkage method.
    for k in range(distances.shape[0]):
        if k != i and k != j:  # skip the merged cluster
            if method == 'single':
                new_dist = torch.min(distances[i, k], distances[j, k])
            elif method == 'complete':
                new_dist = torch.max(distances[i, k], distances[j, k])
            distances[i, k] = new_dist
            distances[k, i] = new_dist

    if method == 'single':
        distances[i, i] = float('inf')
        distances[j, :] = float('inf')
        distances[:, j] = float('inf')
    elif method == 'complete':
        distances[i, i] = 0.0
        distances[j, :] = 0.0
        distances[:, j] = 0.0

    return i, j, distances
