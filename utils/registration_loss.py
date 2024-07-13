import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, generate_binary_structure

def compute_dice(y_true, y_pred):

    intersection = np.sum(y_true * y_pred)
    sum_true_pred = np.sum(y_true+y_pred)
    if sum_true_pred == 0:
        return 0.0
    return 2.0 * intersection / sum_true_pred

def compute_robust_dice(dices, percentile=68):
    threshold_index = int(len(dices) * percentile / 100)
    sorted_dices = np.sort(dices)[::-1]
    robust_dice = np.mean(sorted_dices[:threshold_index])
    return robust_dice

def compute_centroid(image):
    indices = np.argwhere(image > 0)
    if indices.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    centroid = np.mean(indices, axis=0)
    return centroid

def compute_centroid_mae(centroids_true, centroids_pred):
    mae = np.mean(np.abs(centroids_true - centroids_pred))
    return mae

def compute_hausdorff95(y_true, y_pred):
    contours_true = find_contours(y_true)
    contours_pred = find_contours(y_pred)
    if contours_true.size == 0 or contours_pred.size == 0:
        return np.nan
    distances = cdist(contours_true, contours_pred)
    hd95 = np.percentile(np.min(distances, axis=1), 95)
    return hd95

def find_contours(binary_image):
    binary_image = binary_image.astype(np.bool)
    structure = generate_binary_structure(binary_image.ndim, 1)
    eroded = binary_erosion(binary_image, structure)
    contours = binary_image & ~eroded
    return np.argwhere(contours)

def compute_tre(maes_array):
    tre = np.mean(np.sqrt(np.sum(maes_array ** 2, axis=0)))
    return tre

def compute_rtre(maes_array):
    if maes_array.shape[1] == 0:
        return np.nan
    threshold_index = int(maes_array.shape[1] * 0.68)
    sorted_maes = np.sort(maes_array, axis=1)
    rtre = np.mean(np.sqrt(np.sum(sorted_maes[:, :threshold_index] ** 2, axis=0)))
    return rtre

def compute_rts(maes_array):
    if maes_array.shape[1] == 0:
        return np.nan
    rts = np.mean(np.min(maes_array, axis=0))
    return rts

