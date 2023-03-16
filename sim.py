import cv2
import numpy as np
import math
import random

def create_gaussian_phosphene(phosphene_radius, sigma=None):
    diameter = 2 * phosphene_radius + 1
    x, y = np.meshgrid(np.arange(-phosphene_radius, phosphene_radius + 1), np.arange(-phosphene_radius, phosphene_radius + 1))

    if sigma is None:
        sigma = phosphene_radius / 2

    gaussian_phosphene = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian_phosphene[y ** 2 + x ** 2 > phosphene_radius ** 2] = 0
    return gaussian_phosphene

def hexagonal_grid_iterator(img_shape, radius):
    height, width = img_shape
    vertical_spacing = radius * np.sqrt(3)
    horizontal_spacing = radius * 3

    y_start = int(vertical_spacing / 2)
    x_start = int(horizontal_spacing / 2)

    for y in np.arange(y_start, height, vertical_spacing, dtype=int):
        for x in np.arange(x_start, width, horizontal_spacing, dtype=int):
            yield x, y
        x_start = int(x_start + horizontal_spacing / 2) % int(horizontal_spacing)

def simulate_phosphene_vision(image_path, h_res, w_res, enhace_contrast=False, 
                              upsample_ratio=None, downsample_ratio=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # enhance the text contrast in the image
    if enhace_contrast:
        img = cv2.equalizeHist(img)
    
    if downsample_ratio is not None:
        print("downsample_ratio: ", downsample_ratio)
        # downsample the image to reduce the computation time
        img = cv2.resize(img, (img.shape[1] // downsample_ratio, img.shape[0] // downsample_ratio), interpolation=cv2.INTER_AREA)
    elif upsample_ratio is not None:
        print("upsample_ratio: ", upsample_ratio)
        # upsample the image to increase the phosphene radius
        img = cv2.resize(img, (img.shape[1] * upsample_ratio, img.shape[0] * upsample_ratio), interpolation=cv2.INTER_AREA)

    # Calculate the phosphene radius adaptively
    height, width = img.shape
    phosphene_radius = max(height // h_res, width // w_res) // 2
    print("height, width: ", height, width)
    print("phosphene_radius: ", phosphene_radius)

    # Create the Gaussian phosphene
    gaussian_phosphene = create_gaussian_phosphene(phosphene_radius, phosphene_radius / 2)

    simulated_img = np.zeros_like(img, dtype=np.float32)

    for x, y in hexagonal_grid_iterator(img.shape, phosphene_radius):
        # Get the average grayscale value in the circular region of the input image
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), phosphene_radius, 255, -1)
        valid_mask_indices = np.where(mask == 255)
        
        if valid_mask_indices[0].size > 0:
            avg_gray_value = np.mean(img[valid_mask_indices])
            # quantize the average grayscale value into 3-bit
            avg_gray_value = int(avg_gray_value / 32) * 32

            # Apply the Gaussian intensity distribution to the circular region
            phosphene_intensity = gaussian_phosphene * avg_gray_value

            # Perform the addition operation
            y_min = max(0, y - phosphene_radius)
            y_max = min(img.shape[0], y + phosphene_radius + 1)
            x_min = max(0, x - phosphene_radius)
            x_max = min(img.shape[1], x + phosphene_radius + 1)
            y_range = np.arange(y_min, y_max)
            x_range = np.arange(x_min, x_max)
            
            phosphene_y_min = phosphene_radius - (y - y_min)
            phosphene_y_max = phosphene_radius + (y_max - y)
            phosphene_x_min = phosphene_radius - (x - x_min)
            phosphene_x_max = phosphene_radius + (x_max - x)
            
            phosphene_cropped = phosphene_intensity[phosphene_y_min:phosphene_y_max, phosphene_x_min:phosphene_x_max]

            simulated_img[y_min:y_max, x_min:x_max] += phosphene_cropped

    # Normalize the image
    simulated_img = simulated_img - simulated_img.min()
    simulated_img = simulated_img / simulated_img.max() * 255
    simulated_img = simulated_img.astype(np.uint8)

    return simulated_img

if __name__ == "__main__":
    image_path = "./data/input_image_crop.jpg"
    simulated_img = simulate_phosphene_vision(image_path, 
                                              h_res=30, w_res=40, 
                                              enhace_contrast=False,
                                              upsample_ratio=3,
                                              downsample_ratio=None)

    # Save the simulated image
    cv2.imwrite("simulated_image.jpg", simulated_img)

    # Display the original and simulated images
    cv2.imshow("Original", cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("Simulated", simulated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
