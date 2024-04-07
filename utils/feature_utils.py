# base data sci libraries
import os
# !pip install PyWavelets
# import pywt
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from scipy.stats import moment
import cv2
from skimage.filters import gabor_kernel
from PIL import Image
from sklearn.cluster import KMeans
import pywt

def extract_log_features(image_array, sigma=0.7, scalar=True):
    """
    Extract Laplacian of Gaussian (LoG) features from an image array.
    
    Parameters:
        image_array (numpy.ndarray): The input image array.
        sigma (float): The sigma value for the Gaussian filter. Controls the amount of smoothing.
        
    Returns:
        numpy.ndarray: The LoG filtered image as a feature vector.
    """
    # apply Laplacian of Gaussian filter
    log_image = ndimage.gaussian_laplace(image_array, sigma=sigma)

    if scalar:
        
        # OPTION 1
        # feature scalar: the sum of absolute values in the LoG image (a simple measure of edginess)
        feature_scalar = np.sum(np.abs(log_image))

        return feature_scalar

    else:
        # OPTION 2
        # feature vector: flatten the LoG image to use as a feature vector directly
        feature_vector = log_image.flatten()

        return feature_vector

def extract_hog_features(image_array, orientation, pixels, scalar=True):
    """
    Extract HOG features from an image array.
    
    Parameters:
        image_array (numpy.ndarray): The input image array.
        saclar: if the output must be a scalar or a feature vector
        
    Returns:
        numpy.ndarray: The HOG as a list of scalars or a feature vector.
    """
    #preprocessing
    # convert to floating point image with intensity [0, 1]
    if np.max(image_array) > 1:
        img = image_array.astype(np.float32) / 255.0
    # convert to grayscale
    else:
        img = image_array
    if len(img.shape) > 2:
        img = rgb2gray(img)
    gray_img = img
    
    #HOG feature extraction
    #the orientation and pixels will increase the detail (more orientations and less pixels are more computationally expensive)
    feature_vector = hog(gray_img, orientations=orientation, pixels_per_cell=(pixels, pixels), visualize=False, feature_vector=True)  
    if scalar:
        feature_scalar_ls = []
        feature_scalar_ls.extend([np.mean(feature_vector), #np.mean is for averaging features
                               np.sum(feature_vector), # Overall "strength" or "intensity" of the features
                               np.var(feature_vector),  # Variance
                               moment(feature_vector, moment=3), # Skewness
                               moment(feature_vector, moment=4)]) # Kurtosis
                
        return feature_scalar_ls
    else:
        return feature_vector

def extract_normals_features(image_array, scalar=True):
    #preprocessing
    # convert to floating point image with intensity [0, 1]
    if np.max(image_array) > 1:
        img = image_array.astype(np.float32) / 255.0
    else:
        img = image_array
    # convert to grayscale
    if len(img.shape) > 2:
        img = rgb2gray(img)
    gray_img = img
    # Compute gradients using Sobel operator
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute normal vectors Nx, Ny, Nz
    norm = np.sqrt(sobel_x**2 + sobel_y**2 + 1e-6)
    nx = sobel_x / norm
    ny = sobel_y / norm
    nz = 1 / norm

    # Concatenate nx, ny, nz along a new axis, and flatten it to form a 1D feature vector
    feature_vector = np.stack((nx, ny, nz), axis=-1).reshape(-1)
    if scalar:
        feature_scalar = np.sum(feature_vector)
        return feature_scalar
    else:
        return feature_vector
    
def extract_gabor_features(image_array, frequency=0.6, theta=0, sigma=1.0, scalar=True):
    """
    Extract Gabor features from an image array.

    Parameters:
        image_array (numpy.ndarray): Input image array.
        frequency (float): Controls width of strips of Gabor function. Decreasing wavelength produces thinner stripes
        theta (float): Orientation of the Gabor kernel in radians.
        sigma (float): Standard deviation of kernel in both x and y directions (isotropic).

    Returns:
        numpy.ndarray: Gabor-filtered image as a feature (scalar or vector).
    """
    # Create Gabor kernel
    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)) # NOTE: Adjust sigma_x or sigma_y if want anisotropic

    # Apply Gabor kernel to image
    gabor_image = np.abs(ndimage.convolve(image_array, kernel, mode='wrap'))

    if scalar:
        # OPTION 1
        # Feature scalar: sum of values in Gabor filtered image
        feature_scalar = np.sum(gabor_image)
        return feature_scalar
    else:
        # OPTION 2
        # Feature vector: flatten Gabor filtered image to use as feature vector directly
        feature_vector = gabor_image.flatten()
        return feature_vector

def create_visual_vocab(training_folder_path, k=20):

    # create sift detector object
    detector = cv2.SIFT_create()
    all_descriptors = []

    for filename in os.listdir(training_folder_path):
        if filename.endswith(".png"):
            image = Image.open(os.path.join(training_folder_path, filename)).convert('L')
            img_array = np.array(image)
            # detect key points + generate descriptors (vectors describing key points, each with 128 elements)
            _, descriptors = detector.detectAndCompute(img_array, None) # descriptors will be an array with shape (n,128) where n is number of key points detected
            if descriptors is not None:
                all_descriptors.extend(descriptors)

    all_descriptors_stacked = np.vstack(all_descriptors)
    fitted_model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(all_descriptors_stacked)
    return fitted_model
    # # k = 20  # Example number of visual words
    # kmeans_trained = KMeans(n_clusters=k, random_state=0).fit(all_descriptors_stacked)
    # return kmeans_trained

def extract_bovw_features(image_array, kmeans_trained, k=20):    
    """
    Extract Bag of Visual Words (BoVW) features from an image array.

    Parameters:
    image_array (numpy.ndarray): Input image array.
    codebook (numpy.ndarray): Visual vocabulary for quantization.
    patch_size (tuple): Size of the image patches to extract (height, width).

    Returns:
    numpy.ndarray: BoVW representation as a feature (scalar or vector).
    """

    # Check if image array is empty or has incorrect dimensions
    if image_array is None or len(image_array.shape) != 2:
        raise ValueError("Invalid image array")

    # Convert to uint8 if necessary
    if image_array.dtype != 'uint8':
        image_array = image_array.astype('uint8')
    # Detect and compute local features w/ SIFT
    
    # create sift detector object
    detector = cv2.SIFT_create()
    # detect key points + generate descriptors (vectors describing key points, each with 128 elements)
    _, descriptors = detector.detectAndCompute(image_array, None) # descriptors will be an array with shape (n,128) where n is number of key points detected

    if descriptors is not None:
        # predict the nearest cluster each descriptor belongs to
        visual_word_ids = kmeans_trained.predict(descriptors)
        # this object is an array representing the frequency of each category
        histogram, _ = np.histogram(visual_word_ids, bins=range(k + 1), density=True)
        return histogram
    else:
        return np.zeros(k)

def extract_wavelet_features(image_array, wavelet='haar', level=1, scalar=True):
    """
    Extract wavelet transform features from image array.
    
    Parameters:
        image_array (numpy.ndarray): Input image array.
        wavelet (str): Type of wavelet to be used. 
            Doc: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
            And: https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html
        level (int): Level of decomposition in wavelet transform.
        
    Returns:
        numpy.ndarray: Wavelet feature as a vector or scalar.
    """
    # apply wavelet transform
    coeffs = pywt.wavedec2(image_array, wavelet, level=level)

    if scalar:
        # compute mean, variance, skewness, and kurtosis from wavelet coefficients
        cA, cD = coeffs[0], coeffs[1:]
        feature_scalar_ls = []
        feature_scalar_ls.extend([np.mean(cA), np.var(cA), np.mean(cD), np.var(cD)])
        

        return feature_scalar_ls
    else:
        # return wavelet coefficients as a flattened feature vector
        feature_vector = np.concatenate([np.array(c).flatten() for c in coeffs])
        return feature_vector