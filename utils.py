import torch
import random
import numpy as np
import os

def ADE_FDE(y_, y, batch_first=False):
    """
    Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        y_: Predicted trajectories (shape: S x L x N x 2).
        y: Ground truth trajectories (shape: S x L x N x 2).
        batch_first: If True, the batch dimension is the first dimension.
    
    Returns:
        Tuple of ADE and FDE values.
    """
    if torch.is_tensor(y):
        err = (y_ - y).norm(dim=-1)  # S x L x N
    else:
        err = np.linalg.norm(np.subtract(y_, y), axis=-1)
        
    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)  # S x N

    return ade, fde

def kmeans(k, data, iters=None):
    """
    Perform K-means clustering.
    
    Args:
        k: Number of clusters.
        data: Data points (shape: N x D).
        iters: Maximum iterations (default is 100000).
    
    Returns:
        Array of centroids after clustering.
    """
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]

    if iters is None:
        iters = 100000
        
    for _ in range(iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        closest = np.argmin(distances, axis=0)
        centroids_ = []
        
        for k in range(len(centroids)):
            cand = data[closest == k]
            if len(cand) > 0:
                centroids_.append(cand.mean(axis=0))
            else:
                centroids_.append(data[np.random.randint(len(data))])
                
        centroids_ = np.array(centroids_)
        
        if np.linalg.norm(centroids_ - centroids) < 0.0001:
            break
            
        centroids = centroids_

    return centroids  # Returns k centroids after clustering

def seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value (if None, a random seed will be generated).
    """
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
        
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    """
    Get the random number generator state.
    
    Args:
        device: Device identifier.
    
    Returns:
        Tuple of states for different random number generators.
    """
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device) if torch.cuda.is_available() and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
    )

def set_rng_state(state, device):
    """
    Set the random number generator state.
    
    Args:
        state: State tuple obtained from get_rng_state.
        device: Device identifier.
    """
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
