import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import xml.etree.ElementTree as xmlET
from pathlib import Path
from PIL import ImageOps
from PIL import Image
from urllib.parse import quote
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BaseballData(Dataset):
    """
    PyTorch Dataset for baseball frames and annotations stored in a public GitHub repo.

    Expected folde structure:
        repo/
          ├─ annotations/
          │    ├─ imageid1.xml
          │    └─ imageid2.xml
          └─ frames/
               ├─ imageid1_frame0.jpg
               ├─ imageid1_frame1.jpg
               └─ imageid2_frame0.jpg

    Args:
        base_folder (str): Folder where annotation and framers images reside: C:/dir1/dir2
        image_size (tuple): Resize frames to this (default (28,28))
    """

    def __init__(self, base_folder, image_size=(28, 28)):
        super().__init__()
        self.base_folder = base_folder
        self.image_size = image_size

        print("Initializing dataset...")
        self.raw_data = self._consolidate_from_github_repo()
        if self.raw_data.empty:
            raise ValueError("No data found — check repo path or annotations.")
        print(f"Dataset loaded with {len(self.raw_data)} samples")

    # -------------------------------------------------------------------------
    #  Helper methods (previous standalone functions, now encapsulated)
    # -------------------------------------------------------------------------

    def _parse_cvat_xml_and_frames(self, xml_file, frames_base_folder):
        """Parse one XML and associate frames with their corresponding images (load each frame only once)."""
        
        df = pd.DataFrame()
        tree = xmlET.parse(xml_file)
        root = tree.getroot()
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
  
        tracked_items ={}
        frame_list =[]
        for track in root.findall("track"):
            
            track_id = int(track.get("id"))
            label = track.get("label")

            for box in track.findall("box"):
              frame = int(box.get("frame"))

              moving_attr = box.find("attribute[@name='moving']")
              if moving_attr is not None and moving_attr.text:
                moving = moving_attr.text.strip().lower()
              else:
                moving = "false"  
              trackid_info = {
                  "track_id": track_id,
                  "label": label,
                  "coorindates":{
                    "xtl": float(box.get("xtl")),
                    "ytl": float(box.get("ytl")),
                    "xbr": float(box.get("xbr")),
                    "ybr": float(box.get("ybr"))
                  },
                  "moving": moving
              }
              if frame not in tracked_items:
                tracked_items[frame] = [trackid_info]
              else:
                tracked_items[frame].append(trackid_info)


        for frame, objects in tracked_items.items():
          image_found = False
          for ext in [".jpg", ".jpeg", ".png"]:
              image_file = f"{frames_base_folder.rstrip('/')}/{quote(base_name)}/{quote(base_name)}_frame{frame}{ext}"
              if os.path.exists(image_file):
                # Convert to grayscale
                # img = Image.open(image_file).convert("L") 

                img = Image.open(image_file)
                img = img.convert("RGB")
                #img = img.rotate(90, expand=True)  # rotate 90° counter - clockwise
                
                original_size = img.size  # (width, height)
                img = img.resize(self.image_size)

                #flat_pixels = np.array(img).flatten().tolist()

                image_found = True
                frame_info = {
                    "image" : img,
                    #"image" : flat_pixels,
                    "image_file" : image_file,
                    "tracked_objects" : objects,
                    "original_size": original_size  # store original frame size
                }
                frame_list.append(frame_info)
                break

          if not image_found:
              print(f"Missing frame image for {base_name}_frame{frame}")
              continue

          # Append annotation record

        df = pd.DataFrame(frame_list)

        return df


    def _consolidate_from_github_repo(self):

        """Fetch all XML files from 'annotations' folder and process them."""

        base_folder = self.base_folder
        if os.path.isdir(base_folder):
            print(f"Base Folder - {base_folder} exists!")
        else:
            print(f"Base Folder - {base_folder} does not exist.")


        #specify the location of annotations and frames images
        #-------------------------------------------------------
        annotations_path = "annotations"
        frames_path = "frames"


        # API Calls to get associated frames
        #-----------------------------------------------------------------

        all_dfs = []
        frames_folder_loc = f"{base_folder}/{frames_path}"
        annotation_folder_loc = Path(f"{base_folder}/{annotations_path}")

        for xml_file in annotation_folder_loc.glob("*.xml"):
            print(f"Reading {xml_file} ...")
            try:
                df = self._parse_cvat_xml_and_frames(xml_file, frames_folder_loc)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error parsing {base_folder}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        # create a new continuous index (0, 1, 2, …) for the combined DataFrame
        return pd.concat(all_dfs, ignore_index=True)

    @staticmethod
    def collate_fn(batch):
        images = []
        labels = []
        coords = []

        for img, lbl, cor in batch:
            images.append(img)
            labels.append(lbl)
            coords.append(cor)

        # stack only the image tensors (if all same size)
        images = torch.stack(images, dim=0)

        # keep labels and coords as lists (different lengths)
        return images, labels, coords


    # -------------------------------------------------------------------------
    #  Required Dataset methods for PyTorch
    # -------------------------------------------------------------------------

    def __len__(self):
        """Total number of samples."""
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        """Return one sample (image tensor, label tensor)."""
        row = self.raw_data.iloc[idx]
        image_file = row['image_file']
        print(f"Image Name = {row['image_file']}")

        # Convert PIL image to a PyTorch Tensor
        # Tensor will have data in the order of height , weight and channel (H, W, C)
        image_np = np.array(row["image"], dtype=np.float32) / 255.0

        #Needed Format for PyTorch tensors  (C, H, W)
        image_np = np.transpose(image_np, (2, 0, 1))
       
        image = torch.tensor(image_np, dtype=torch.float32)

        # For Scaling Purpose
        #-------------------------
        orig_w, orig_h = row['original_size']
        new_w, new_h = self.image_size

        tracked_items = row["tracked_objects"]

        #Create tracked Items coordinates as list of [xtl, ytl, xbr, ybr]
        all_coordinates = []
        for item in tracked_items:
            coords_dict = item['coorindates']
            # Extract the values in a specific order: xtl, ytl, xbr, ybr

            # Scale it propotionally to our image size
            xtl = coords_dict['xtl'] * (new_w / orig_w)
            ytl = coords_dict['ytl'] * (new_h / orig_h)
            xbr = coords_dict['xbr'] * (new_w / orig_w)
            ybr = coords_dict['ybr'] * (new_h / orig_h)

            all_coordinates.append([xtl, ytl, xbr, ybr])

        #print(all_coordinates)
        coordinates_list_torch = torch.tensor(all_coordinates, dtype=torch.float32)

        # label from 'moving' attribute (0 or 1)
        label_list = [1 if item.lower() == "true" else 0 for item in [item['moving'] for item in tracked_items]]
        labels_list_torch = torch.tensor(label_list, dtype=torch.long)

        #print(label_list)
        #print(label_list_torch)
        return image, labels_list_torch, coordinates_list_torch




#### EXECUTE CODE ###############
"""
    Expected folder structure:
        repo/
          ├─ annotations/
          │    ├─ imageid1.xml
          │    └─ imageid2.xml
          └─ frames/
               ├─ imageid1_frame0.jpg
               ├─ imageid1_frame1.jpg
               └─ imageid2_frame0.jp
"""

base_folder = "C:/Users/Tech/OneDrive - University of Nebraska at Omaha/DataScience/BusinessForecasting-ECON8310/econ8310-assignment3"
base_folder = "C:/Users/Tech/OneDrive - University of Nebraska at Omaha/DataScience/BusinessForecasting-ECON8310/final-project"
traindata = BaseballData(base_folder, image_size=(224, 224))
loader = DataLoader(traindata, batch_size=8, shuffle=True,collate_fn=lambda x: traindata.collate_fn(x))

# Get one batch
images, labels, coords = next(iter(loader))
print("Batch shapes:", images.shape)
print("Labels for first image:", labels[0])
print("Coords for first image:", coords[0])

# Visualize first image
image_tensor = images[0]      

# remove channel dimension
# For Graysclae Image            
#image_np = image_tensor.squeeze().numpy()  
#plt.imshow(image_np, cmap="gray")

#Orders the tensor to (H,W,C) for Ploting
#---------------------------------------------
image_np = image_tensor.permute(1, 2, 0).numpy()


plt.clf()   # Clear the current figure
plt.imshow(image_np)

#draw bounding boxes
for box in coords[0]:
    xtl, ytl, xbr, ybr = box
    plt.gca().add_patch(
        plt.Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, 
                      edgecolor='red', facecolor='none', linewidth=1)
    )

plt.title(f"Moving labels: {labels[0].tolist()}")
plt.axis('off')
plt.show()