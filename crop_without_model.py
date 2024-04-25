import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image
import easygui
import os

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    return average_brightness

def plot_grayscale_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def remove_background_rembg(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_np = np.array(output_image)

    if output_np.shape[2] == 4 and np.all(output_np[:, :, 3] == 255):
        return cv2.imread(image_path)
    else:
        return output_np

def apply_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to exclude small contours
    min_contour_area = 1000
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Approximate contours to reduce the number of points
    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Draw contours on the mask
    cv2.drawContours(mask, [approx], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    result_with_contours = cv2.bitwise_and(image, image, mask=mask)

    return result_with_contours, [approx]

# Select the input image file using a file dialog
input_path = easygui.fileopenbox(title='Select image file')
input_folder, input_filename = os.path.split(input_path)

# Remove background using rembg
result_rembg = remove_background_rembg(input_path)

# Calculate brightness of the original and processed images
brightness_original = calculate_brightness(cv2.imread(input_path))
brightness_processed = calculate_brightness(result_rembg)

# Plot the grayscale histogram
plot_grayscale_histogram(result_rembg)

# Apply contours to the processed image
result_with_contours, contours = apply_contours(result_rembg)

# Display brightness information
print(f"Original Image Brightness: {brightness_original}")
print(f"Processed Image Brightness: {brightness_processed}")

# Save the processed image in the same folder as the input image
output_path = os.path.join(input_folder, 'processed_' + input_filename)
cv2.imwrite(output_path, cv2.cvtColor(result_rembg, cv2.COLOR_BGR2RGB))
print(f"Processed image saved at: {output_path}")

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(1, 4, 2), plt.imshow(cv2.cvtColor(result_rembg, cv2.COLOR_BGR2RGB)), plt.title('Background Removed')
plt.subplot(1, 4, 3), plt.imshow(result_with_contours), plt.title('Result with Contours')
plt.subplot(1, 4, 4), plt.imshow(cv2.drawContours(result_rembg.copy(), contours, -1, (0, 255, 0), 2)), plt.title('Contours Only')

plt.show()