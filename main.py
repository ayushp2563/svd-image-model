# 1. Setup
import numpy as np
import matplotlib.pyplot as plt
import cv2  
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA 
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
from IPython.display import display, clear_output
import os
import warnings

warnings.filterwarnings('ignore') 

# --- Helper Functions ---

def load_image(image_path, grayscale=False):
    """Loads an image using OpenCV."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Convert from BGR (OpenCV default) to RGB (Matplotlib default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_metrics(original, reconstructed):
    """Calculates PSNR and SSIM."""
    # Ensure images are float and in range [0, 1] or uint8 for metrics
    if original.dtype != np.uint8:
        original_norm = original.astype(np.float64) / 255.0
    else:
         original_norm = original.astype(np.float64)

    if reconstructed.dtype != np.uint8:
       reconstructed_norm = np.clip(reconstructed, 0, 255).astype(np.float64) / 255.0
    else:
        reconstructed_norm = reconstructed.astype(np.float64)


    # Determine data range based on dtype
    data_range = 255.0 if original.dtype == np.uint8 else 1.0
    
    # Use appropriate images for calculation based on data range expected by skimage
    img_orig_for_metric = original if data_range == 255.0 else original_norm
    img_recon_for_metric = np.clip(reconstructed, 0, 255).astype(img_orig_for_metric.dtype) if data_range == 255.0 else reconstructed_norm

    # Handle multichannel (color) for SSIM if needed
    multichannel = len(original.shape) == 3

    try:
        # PSNR requires data_range
        psnr_val = psnr(img_orig_for_metric, img_recon_for_metric, data_range=data_range)
    except ValueError as e:
        print(f"PSNR calculation error: {e}. Shapes: Original {original.shape}, Reconstructed {reconstructed.shape}")
        psnr_val = -1 # Indicate error

    try:
         # SSIM needs consistent data types and handles multichannel
         ssim_val = ssim(img_orig_for_metric, img_recon_for_metric,
                        data_range=data_range,
                        channel_axis=-1 if multichannel else None, # Updated parameter name
                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False) # Default ssim params
    except ValueError as e:
        print(f"SSIM calculation error: {e}. Shapes: Original {original.shape}, Reconstructed {reconstructed.shape}")
        ssim_val = -1 # Indicate error
    except TypeError as e:
         print(f"SSIM calculation error (TypeError, likely multichannel issue): {e}. Shapes: Original {original.shape}, Reconstructed {reconstructed.shape}")
         # Try forcing grayscale for SSIM if multichannel fails unexpectedly
         if multichannel:
              try:
                    gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if data_range == 255.0 else cv2.cvtColor(original_norm*255, cv2.COLOR_RGB2GRAY) / 255.0
                    gray_recon = cv2.cvtColor(reconstructed.astype(np.uint8), cv2.COLOR_RGB2GRAY) if data_range == 255.0 else cv2.cvtColor(reconstructed_norm*255, cv2.COLOR_RGB2GRAY) / 255.0
                    gray_orig_metric = gray_orig if data_range == 255.0 else gray_orig
                    gray_recon_metric = gray_recon if data_range == 255.0 else gray_recon
                    ssim_val = ssim(gray_orig_metric, gray_recon_metric, data_range=data_range)
              except Exception as inner_e:
                    print(f"Fallback grayscale SSIM failed: {inner_e}")
                    ssim_val = -1
         else:
              ssim_val = -1 # Indicate error


    return psnr_val, ssim_val

def calculate_compression_ratio(original_shape, k):
    """Estimates compression ratio for SVD."""
    m, n = original_shape[0], original_shape[1]
    original_size = m * n
    # Size of U_k (m*k), S_k (k), V_k (k*n)
    compressed_size = k * (m + n + 1)

    if len(original_shape) == 3: # Color image
        original_size *= 3
        compressed_size *= 3

    ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    storage_saved = 100 * (1 - (compressed_size / original_size)) if original_size > 0 and compressed_size > 0 else 0

    return ratio, storage_saved

# --- 2. Core SVD Functions ---

def svd_compress_reconstruct(image, k):
    """Performs SVD compression and reconstruction on a grayscale or color image."""
    if len(image.shape) == 3: # Color image
        reconstructed_channels = []
        # Process each channel separately
        for i in range(3):
            channel = image[:, :, i]
            U, S, Vt = np.linalg.svd(channel, full_matrices=False)
            # Truncate
            U_k = U[:, :k]
            S_k = np.diag(S[:k])
            Vt_k = Vt[:k, :]
            # Reconstruct channel
            reconstructed_channel = U_k @ S_k @ Vt_k
            reconstructed_channels.append(reconstructed_channel)
        # Stack channels back together
        reconstructed_image = np.stack(reconstructed_channels, axis=-1)

    else: # Grayscale image
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        # Truncate
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        # Reconstruct
        reconstructed_image = U_k @ S_k @ Vt_k

    # Clip values to valid range (e.g., 0-255 for uint8)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    return reconstructed_image.astype(image.dtype) # Return same dtype as input

def plot_singular_values(image):
    """Calculates and plots the singular values."""
    if len(image.shape) == 3: # Use luminance for color image singular values plot
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    U, S, Vt = np.linalg.svd(gray_image, full_matrices=False)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(S, 'b.')
    plt.title('Singular Values (Linear Scale)')
    plt.ylabel('Value')
    plt.xlabel('Index')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(S, 'b.') # Log scale often more informative
    plt.title('Singular Values (Log Scale)')
    plt.ylabel('Value (log)')
    plt.xlabel('Index')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return S # Return singular values


# --- 3. Interactive SVD Explorer ---

# Global variable to hold the image for the interactive widget
current_image_interactive = None
original_image_interactive = None
output_widget = widgets.Output()

def run_interactive_svd_explorer(image_path):
    """Sets up and runs the interactive SVD explorer for the given image."""
    global current_image_interactive, original_image_interactive, output_widget
    output_widget.clear_output() # Clear previous output

    original_image_interactive = load_image(image_path, grayscale=False) # Load as color first
    if original_image_interactive is None:
        with output_widget:
            print(f"Failed to load image: {image_path}")
        return

    # Decide whether to work in grayscale or color based on user choice or image properties
    # For simplicity here, let's offer a choice or default to color if available
    is_color = len(original_image_interactive.shape) == 3 and original_image_interactive.shape[2] == 3

    if not is_color:
         current_image_interactive = original_image_interactive # Already grayscale or loaded as such
    else:
         current_image_interactive = original_image_interactive # Keep color

    max_k = min(current_image_interactive.shape[0], current_image_interactive.shape[1])

    k_slider = widgets.IntSlider(
        value=min(50, max_k), # Sensible starting value
        min=1,
        max=max_k,
        step=1,
        description='Keep k Singular Values:',
        continuous_update=False, # Update only on release for performance
        layout=Layout(width='80%')
    )

    def update_interactive_image(k):
        """Function called by the slider interaction."""
        global current_image_interactive, original_image_interactive, output_widget
        with output_widget:
            clear_output(wait=True) # Clear previous plot/text in the output widget

            if current_image_interactive is None:
                print("Error: Image not loaded.")
                return

            reconstructed = svd_compress_reconstruct(current_image_interactive, k)
            psnr_val, ssim_val = calculate_metrics(original_image_interactive, reconstructed)
            comp_ratio, storage_saved = calculate_compression_ratio(original_image_interactive.shape, k)

            # Display Images
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_image_interactive)
            axes[0].set_title(f'Original ({original_image_interactive.shape[0]}x{original_image_interactive.shape[1]})')
            axes[0].axis('off')

            axes[1].imshow(reconstructed)
            axes[1].set_title(f'Reconstructed (k={k})')
            axes[1].axis('off')

            plt.suptitle('SVD Image Compression Explorer', fontsize=16)
            plt.show()

            # Display Metrics
            print(f"\n--- Metrics for k = {k} ---")
            print(f"PSNR: {psnr_val:.2f} dB")
            print(f"SSIM: {ssim_val:.4f}")
            print(f"Estimated Compression Ratio: {comp_ratio:.2f} : 1")
            print(f"Estimated Storage Saved: {storage_saved:.2f}%")
            print("------------------------------")


    # Initial display
    update_interactive_image(k_slider.value)

    # Link slider to update function and display
    interactive_plot = interactive(update_interactive_image, k=k_slider)
    display(interactive_plot)
    display(output_widget) # Display the output area where plots/text appear


# --- 4. SVD for Denoising ---

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    row, col, *ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, *ch))
    gauss = gauss.reshape(row, col, *ch)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(image.dtype)
    return noisy_image

def svd_denoise(noisy_image, k_ratio=0.1):
     """Denoises an image using SVD by keeping a ratio of singular values."""
     if len(noisy_image.shape) == 3:
        max_k_possible = min(noisy_image.shape[0], noisy_image.shape[1])
        k = max(1, int(max_k_possible * k_ratio)) # Keep top e.g. 10% singular values
        denoised_image = svd_compress_reconstruct(noisy_image, k)
     else: # Grayscale
        max_k_possible = min(noisy_image.shape[0], noisy_image.shape[1])
        k = max(1, int(max_k_possible * k_ratio))
        denoised_image = svd_compress_reconstruct(noisy_image, k)

     print(f"Denoising using k={k} singular values (approx {k_ratio*100:.1f}%).")
     return denoised_image

# --- 5. Eigenfaces using SVD ---

def run_eigenfaces_demo(n_components=100, n_faces_to_show=10, face_idx_to_reconstruct=0):
    """Loads face data, computes eigenfaces using SVD, and shows reconstruction."""
    print("\n--- Running Eigenfaces Demo ---")
    # Load data
    try:
        faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
        faces_images = faces_data.images # Shape (400, 64, 64)
        n_samples, h, w = faces_images.shape
        print(f"Loaded Olivetti faces dataset: {n_samples} samples, {h}x{w} pixels each.")
    except Exception as e:
        print(f"Error loading Olivetti dataset: {e}")
        print("Please ensure scikit-learn is installed and can download datasets.")
        return

    # Flatten images: Treat each image as a row vector (n_samples x n_features)
    X = faces_images.reshape(n_samples, h * w)
    n_features = h * w

    # Calculate mean face
    mean_face = np.mean(X, axis=0)

    # Center the data
    X_centered = X - mean_face

    # Perform SVD on the centered data matrix
    # U contains projections, S singular values, Vt contains principal components (eigenfaces)
    print("Performing SVD on centered face data...")
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    print("SVD complete.")

    # The rows of Vt are the principal components (eigenfaces)
    eigenfaces = Vt[:n_components, :] # Take the first n_components

    # --- Visualization ---
    print(f"\nVisualizing Mean Face and top {n_faces_to_show} Eigenfaces...")

    # Plot the mean face
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mean_face.reshape(h, w), cmap=plt.cm.gray)
    plt.title('Mean Face')
    plt.axis('off')

    # Plot a gallery of the top eigenfaces
    plt.subplot(1, 2, 2) # Placeholder, gallery will be separate figure
    plt.text(0.5, 0.5, 'Eigenfaces ->\n(See next plot)', ha='center', va='center')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    def plot_gallery(title, images, image_shape, n_col=5, n_row=2):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                       interpolation='nearest', vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.show()

    plot_gallery(f"Top {n_faces_to_show} Eigenfaces (Principal Components)",
                 eigenfaces[:n_faces_to_show], (h,w),
                 n_col=min(n_faces_to_show, 5), n_row = (n_faces_to_show + 4) // 5)


    # --- Reconstruction ---
    print(f"\nReconstructing Face #{face_idx_to_reconstruct} using {n_components} Eigenfaces...")

    # Get the original face and center it
    original_face_flat = X[face_idx_to_reconstruct, :]
    original_face_centered = original_face_flat - mean_face

    # Project onto the eigenface basis (using Vt directly)
    # Coefficients = X_centered @ V^T (where V = Vt.T)
    # So, coeffs = X_centered @ eigenfaces.T
    # For a single face: coeff_single = original_face_centered @ eigenfaces.T
    coeffs = original_face_centered @ eigenfaces.T # Shape: (n_components,)

    # Reconstruct from coefficients and eigenfaces
    # reconstructed_centered = coeffs @ eigenfaces
    reconstructed_centered = coeffs @ eigenfaces # Shape: (n_features,)

    # Add the mean face back
    reconstructed_face_flat = reconstructed_centered + mean_face

    # Visualize original vs reconstructed
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_face_flat.reshape(h, w), cmap=plt.cm.gray)
    plt.title(f'Original Face #{face_idx_to_reconstruct}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_face_flat.reshape(h, w), cmap=plt.cm.gray)
    plt.title(f'Reconstructed (k={n_components})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    psnr_rec, ssim_rec = calculate_metrics(original_face_flat.reshape(h,w)*255, # Scale to 0-255 uint8 range for metrics
                                           reconstructed_face_flat.reshape(h,w)*255)
    print(f"Reconstruction Metrics (Face #{face_idx_to_reconstruct}, k={n_components}): PSNR={psnr_rec:.2f} dB, SSIM={ssim_rec:.4f}")
    print("--- Eigenfaces Demo Complete ---")


# --- 6. Example Usage ---

if __name__ == "__main__":

    # --- Prerequisites ---
    # Create a dummy image file for testing if it doesn't exist
    # In a real scenario, replace 'test_image.png' with your image path
    IMAGE_PATH = 'test_image.png'
    if not os.path.exists(IMAGE_PATH):
        print(f"Creating a dummy test image: {IMAGE_PATH}")
        dummy_img = np.random.randint(0, 256, (256, 384, 3), dtype=np.uint8)
        # Add some structure
        dummy_img[50:150, 100:200, :] = np.random.randint(100, 200, (100,100,3))
        cv2.imwrite(IMAGE_PATH, cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR)) # Save as BGR

    # --- Part 1: Basic Compression Example ---
    print("--- Part 1: Basic SVD Compression ---")
    img_orig_color = load_image(IMAGE_PATH, grayscale=False)
    img_orig_gray = load_image(IMAGE_PATH, grayscale=True)

    if img_orig_color is not None and img_orig_gray is not None:
        # Plot singular values (using grayscale version)
        print("Plotting Singular Values...")
        singular_values = plot_singular_values(img_orig_gray)
        max_k_val = len(singular_values)
        print(f"Total singular values: {max_k_val}")

        # Compress with a specific k
        k_compress = 50 # Example k value
        print(f"\nCompressing Color Image with k = {k_compress}")
        img_compressed_color = svd_compress_reconstruct(img_orig_color, k_compress)
        psnr_c, ssim_c = calculate_metrics(img_orig_color, img_compressed_color)
        cr_c, ss_c = calculate_compression_ratio(img_orig_color.shape, k_compress)
        print(f"Color Compressed - PSNR: {psnr_c:.2f} dB, SSIM: {ssim_c:.4f}, Comp Ratio: {cr_c:.2f}:1 ({ss_c:.1f}% saved)")

        print(f"\nCompressing Grayscale Image with k = {k_compress}")
        img_compressed_gray = svd_compress_reconstruct(img_orig_gray, k_compress)
        psnr_g, ssim_g = calculate_metrics(img_orig_gray, img_compressed_gray)
        cr_g, ss_g = calculate_compression_ratio(img_orig_gray.shape, k_compress)
        print(f"Grayscale Compressed - PSNR: {psnr_g:.2f} dB, SSIM: {ssim_g:.4f}, Comp Ratio: {cr_g:.2f}:1 ({ss_g:.1f}% saved)")


        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(img_orig_color)
        axes[0, 0].set_title('Original Color')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(img_compressed_color)
        axes[0, 1].set_title(f'Color Compressed (k={k_compress})')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(img_orig_gray, cmap='gray')
        axes[1, 0].set_title('Original Grayscale')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(img_compressed_gray, cmap='gray')
        axes[1, 1].set_title(f'Grayscale Compressed (k={k_compress})')
        axes[1, 1].axis('off')
        plt.tight_layout()
        plt.show()

    else:
        print("Skipping basic compression due to image loading error.")

    # --- Part 2: Interactive Explorer ---
    # Note: This part requires ipywidgets and is best run in Jupyter
    print("\n--- Part 2: Interactive SVD Explorer ---")
    print("Run the following line in a Jupyter Notebook cell:")
    print(f">>> run_interactive_svd_explorer('{IMAGE_PATH}')")
    # Example direct call (will display widgets if in compatible environment)
    # run_interactive_svd_explorer(IMAGE_PATH) # Uncomment to try running directly

    # --- Part 3: Denoising Example ---
    print("\n--- Part 3: SVD Denoising ---")
    if img_orig_color is not None:
        print("Adding Gaussian noise...")
        noisy_img = add_gaussian_noise(img_orig_color, sigma=25) # Add noise with std dev 25

        print("Denoising using SVD (keeping top ~10% singular values)...")
        # Keep fewer singular values for denoising than for pure compression
        denoised_img_svd = svd_denoise(noisy_img, k_ratio=0.1)

        # Compare with a simple Gaussian blur for reference
        denoised_img_blur = cv2.GaussianBlur(noisy_img, (5,5), 0)

        psnr_noisy, ssim_noisy = calculate_metrics(img_orig_color, noisy_img)
        psnr_svd, ssim_svd = calculate_metrics(img_orig_color, denoised_img_svd)
        psnr_blur, ssim_blur = calculate_metrics(img_orig_color, denoised_img_blur)

        print(f"\nMetrics vs Original:")
        print(f"Noisy Image:      PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}")
        print(f"SVD Denoised:     PSNR={psnr_svd:.2f} dB, SSIM={ssim_svd:.4f}")
        print(f"Gaussian Blurred: PSNR={psnr_blur:.2f} dB, SSIM={ssim_blur:.4f}")

        # Visualize denoising results
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_orig_color)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(noisy_img)
        axes[1].set_title(f'Noisy\nPSNR:{psnr_noisy:.1f} SSIM:{ssim_noisy:.2f}')
        axes[1].axis('off')
        axes[2].imshow(denoised_img_svd)
        axes[2].set_title(f'SVD Denoised\nPSNR:{psnr_svd:.1f} SSIM:{ssim_svd:.2f}')
        axes[2].axis('off')
        axes[3].imshow(denoised_img_blur)
        axes[3].set_title(f'Gaussian Blur\nPSNR:{psnr_blur:.1f} SSIM:{ssim_blur:.2f}')
        axes[3].axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping denoising example due to image loading error.")


    # --- Part 4: Eigenfaces Demo ---
    # Run the full Eigenfaces demo defined earlier
    run_eigenfaces_demo(n_components=100, n_faces_to_show=10, face_idx_to_reconstruct=5)

    print("\n--- Project Implementation Complete ---")