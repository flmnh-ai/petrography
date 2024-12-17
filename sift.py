# ! pip install numpy opencv-python tifffile scikit-learn
import numpy as np
import cv2
from pathlib import Path
import tifffile
from sklearn.decomposition import PCA

import numpy as np
import cv2
from pathlib import Path
import tifffile
from sklearn.decomposition import PCA

def normalize_for_visualization(pca_image):
    """
    Normalize PCA components for visualization/ML input.
    Uses robust scaling based on percentiles.
    """
    normalized = np.zeros_like(pca_image, dtype=np.uint8)
    
    # Scale each component independently
    for i in range(3):
        p1, p99 = np.percentile(pca_image[:,:,i], (1, 99))
        normalized[:,:,i] = np.clip(
            ((pca_image[:,:,i] - p1) / (p99 - p1) * 255),
            0, 255
        ).astype(np.uint8)
    
    return normalized

def register_and_analyze_thin_sections(ppl_path, xpl_path, output_tiff, output_pca_png):
    """
    Register PPL and XPL thin section images and perform PCA analysis.
    
    Parameters:
    ppl_path (str): Path to the PPL image
    xpl_path (str): Path to the XPL image
    output_tiff (str): Path for the registered stack TIFF
    output_pca_png (str): Path for normalized PCA visualization (PNG)
    """
    # Read images in original bit depth
    ppl = cv2.imread(ppl_path, cv2.IMREAD_UNCHANGED)
    xpl = cv2.imread(xpl_path, cv2.IMREAD_UNCHANGED)
    
    if ppl is None or xpl is None:
        print("Error: Could not read images")
        return False, None
    
    # Convert to float32 for processing
    ppl = ppl.astype(np.float32)
    xpl = xpl.astype(np.float32)
    
    # Convert to grayscale for feature detection
    ppl_gray = cv2.cvtColor(ppl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    xpl_gray = cv2.cvtColor(xpl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    # SIFT and registration
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(ppl_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(xpl_gray, None)
    
    if descriptors1 is None or descriptors2 is None:
        print("Error: No features detected")
        return False, None
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        print("Error: Not enough good matches found")
        return False, None
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Error: Could not find homography")
        return False, None
    
    # Apply transformation
    h, w = ppl.shape[:2]
    registered_xpl = cv2.warpPerspective(xpl, H, (w, h))
    
    # Create and save image stack
    ppl_rgb = cv2.cvtColor(ppl, cv2.COLOR_BGR2RGB)
    registered_xpl_rgb = cv2.cvtColor(registered_xpl, cv2.COLOR_BGR2RGB)
    stack = np.stack([ppl_rgb, registered_xpl_rgb])
    tifffile.imwrite(output_tiff, stack)
    
    # Prepare data for PCA
    n_pixels = h * w
    X = np.zeros((n_pixels, 6), dtype=np.float32)
    X[:, 0:3] = ppl_rgb.reshape(n_pixels, 3)
    X[:, 3:6] = registered_xpl_rgb.reshape(n_pixels, 3)
    
    # Standardize features for PCA
    X_centered = X - X.mean(axis=0)
    
    # Perform PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_centered)
    
    # Reshape components and normalize for visualization
    pca_image = components.reshape(h, w, 3)
    normalized_pca = normalize_for_visualization(pca_image)
    
    # Save normalized version for visualization/ML
    cv2.imwrite(output_pca_png, cv2.cvtColor(normalized_pca, cv2.COLOR_RGB2BGR))
    
    # Print explained variance ratios
    print("Explained variance ratios:", pca.explained_variance_ratio_)
    
    return True, normalized_pca

# Example usage
if __name__ == "__main__":
    ppl_image = "/Users/nick/UFL Dropbox/Nicolas Gauthier/WallisDatasetSlides/8LV2-391-2.792.jpg"
    xpl_image = "/Users/nick/UFL Dropbox/Nicolas Gauthier/WallisDatasetSlides/8LV2-391-2.8xp93.jpg"
    output_tiff = "registered_stack.tiff"
    output_pca_png = "pca_visualization.png"
    
    success, pca_vis = register_and_analyze_thin_sections(
        ppl_image, xpl_image, output_tiff, output_pca_png
    )
    
    if success:
        print(f"Successfully processed images:")
        print(f"- Registered stack saved to {output_tiff}")
        print(f"- PCA visualization saved to {output_pca_png}")
    else:
        print("Processing failed")
