from io import BytesIO
from PIL import Image
import csv
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def read_csv(directory, image):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    csv_file = list(filter(lambda file: image in file, csv_files))[0]
    data = []
    with open(os.path.join(directory, csv_file), 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def calculate_mean_sizes(csv1, csv2):
    mean_sizes = []
    for row1, row2 in zip(csv1, csv2):
        size1 = float(row1['compressed_size'])
        size2 = float(row2['compressed_size'])
        mean_size = (size1 + size2) / 2.0
        mean_sizes.append(mean_size)
    return mean_sizes

def compress_image(image, target_file_size_kb, max_quality=100, min_quality=0):
    input_image_path = f"test_images/{image}.png"
    im = Image.open(input_image_path)

    # Start with some initial quality value
    quality = (max_quality + min_quality) // 2
    buffer = BytesIO()

    # Binary search to find the appropriate quality value
    while True:
        # Save the image with the current quality setting
        im.save(buffer, "JPEG", quality=quality)

        # Measure the size of the compressed image
        compressed_size_kb = len(buffer.getvalue()) / 1024

        # Adjust the quality value based on the comparison with the target size
        if compressed_size_kb > target_file_size_kb:
            max_quality = quality
        else:
            min_quality = quality

        # If the difference is less than 1 KB, break the loop
        if abs(compressed_size_kb - target_file_size_kb) < 3:
            break

        # Update quality for the next iteration
        quality = (max_quality + min_quality) // 2
        buffer.seek(0)
        buffer.truncate()

    output_image_path = f"jpeg/{image}_{quality}_compressed.jpg"
    # Save the final compressed image
    im.save(output_image_path, "JPEG", quality=quality)

    print("Target file size:", target_file_size_kb, "KB")
    print("Final file size:", compressed_size_kb, "KB")
    print("Quality used:", quality)

    # Calculate SSIM and MSE
    compressed_im = cv2.imread(output_image_path)
    original_im = cv2.imread(input_image_path)
    mse_value = mse(original_im, compressed_im)
    ssim_value = ssim(original_im, compressed_im, channel_axis=2)

    print("MSE:", mse_value)
    print("SSIM:", ssim_value)

    return [quality, compressed_size_kb, mse_value, ssim_value]

if __name__ == "__main__":
    images = [os.path.splitext(file)[0] for file in os.listdir("test_images") if file.endswith('.png')]
    for image in images:
        print(f"{image =}")
        csv1_data = read_csv("bmshj2018-factorized-msssim", image)
        csv2_data = read_csv("mbt2018-mean-msssim", image)
        target_file_sizes_kb = calculate_mean_sizes(csv1_data, csv2_data)

        csv_data = [
            ["quality", "compressed_size", "mse", "ssim"]
        ]
        for target_size in target_file_sizes_kb:
            data = compress_image(image, target_size)
            csv_data.append(data)

        print(csv_data)
        csv_file_path = f"jpeg/{image}.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        print()
