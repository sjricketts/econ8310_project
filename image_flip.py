import cv2
import os

input_folder = "/Users/hugocorado/Documents/PROJECT_FRAMES/dusty_1/"

print(f"Processing folder: {input_folder}")

# Loop through all image files in the folder
for image_file in os.listdir(input_folder):

    # Construct full image path
    image_path = os.path.join(input_folder, image_file)

    # Read the image
    image = cv2.imread(image_path)

    # Rotate 90 degrees clockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # Overwrite the original image
    result = cv2.imwrite(image_path, rotated_image)

    if result:
        print(f"Rotated: {image_file}")
    else:
        print(f"Failed to save: {image_file}")

print("\nAll images rotated and saved to dir: " + input_folder)