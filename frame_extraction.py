import cv2
import os

video_dir = "C:/Users/Tech/OneDrive - University of Nebraska at Omaha/DataScience/BusinessForecasting-ECON8310/final-project/videos"
output = "C:/Users/Tech/OneDrive - University of Nebraska at Omaha/DataScience/BusinessForecasting-ECON8310/final-project/frames"

# Create output directory if it doesn't exist
os.makedirs(output, exist_ok=True)

# Loop through all .mov files in video_dir
for video_file in os.listdir(video_dir):

    # Construct full video pathsa
    video_path = os.path.join(video_dir, video_file)

    # Remove .mov extension
    video_name = os.path.splitext(video_file)[0]

    # Create a folder for this video's frames
    video_output_dir = os.path.join(output, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\nProcessing: {video_file}")

    vidcap = cv2.VideoCapture(video_path)

    print(f"Video opened successfully. Extracting frames to: {video_output_dir}")

    success, image = vidcap.read()
    count = 0
    while success:
        # Rotate image 90 degrees clockwise to make it vertical
        image_vertical = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_vertical = image
        filename = os.path.join(video_output_dir, f"{video_name}_frame{count}.jpg")
        result = cv2.imwrite(filename, image_vertical)
        if result:
            print(f'Saved frame {count}: {filename}')
        else:
            print(f'Failed to save frame {count}')
        success, image = vidcap.read()
        count += 1

    print(f"Total frames extracted from {video_file}: {count}")
    vidcap.release()

print("\nAll videos processed!")