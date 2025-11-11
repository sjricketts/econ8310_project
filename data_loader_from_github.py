from numpy._core import records
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import requests
import xml.etree.ElementTree as xmlET
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import os
from urllib.parse import quote
import matplotlib.pyplot as plt


class BaseballData(Dataset):
    """
    PyTorch Dataset for baseball frames and annotations stored in a public GitHub repo.

    Expected GitHub structure:
        repo/
          ├─ annotations/
          │    ├─ game1.xml
          │    └─ game2.xml
          └─ frames/
               ├─ game1_frame0.jpg
               ├─ game1_frame1.jpg
               └─ game2_frame0.jpg

    Args:
        repo_url (str): GitHub repo base URL (e.g. 'https://github.com/user/repo/tree/main')
        image_size (tuple): Resize frames to this (default (28,28))
    """

    def __init__(self, repo_url, image_size=(28, 28)):
        super().__init__()
        self.repo_url = repo_url
        self.image_size = image_size

        print("Initializing dataset...")
        self.raw_data = self._consolidate_from_github_repo()
        if self.raw_data.empty:
            raise ValueError("No data found — check repo path or annotations.")
        print(f"Dataset loaded with {len(self.raw_data)} samples")

    # -------------------------------------------------------------------------
    #  Helper methods (previous standalone functions, now encapsulated)
    # -------------------------------------------------------------------------

    def _parse_cvat_xml_and_frames(self, xml_url, frames_base_url):
        """Parse one XML and associate frames with their corresponding images (load each frame only once)."""
        response = requests.get(xml_url)
        df = pd.DataFrame()


        """
        When you make a request (like response = requests.get(url)), the response object includes a status code — for example:
        200 = OK
        404 = Not Found
        500 = Server Error
        When you call response.raise_for_status():
        If the status code indicates success (200–299), nothing happens — the program continues.
        If the status code indicates an error (400 or higher), it raises an exception
        """
        response.raise_for_status()

        tree = xmlET.parse(BytesIO(response.content))
        root = tree.getroot()

        base_name = os.path.splitext(os.path.basename(xml_url))[0]

        tracked_items ={}
        frame_list =[]
        for track in root.findall("track"):
            track_id = int(track.get("id"))
            label = track.get("label")
            #image_found = False

            for box in track.findall("box"):
              frame = int(box.get("frame"))

              moving_attr = box.find("attribute[@name='moving']")
              moving = moving_attr.text.strip().lower() if moving_attr is not None else None
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
              image_url = f"{frames_base_url.rstrip('/')}/{quote(base_name)}/{quote(base_name)}_frame{frame}{ext}"
              img_resp = requests.get(image_url)
              if img_resp.status_code == 200:
                  img = Image.open(BytesIO(img_resp.content)).convert("L")
                  img = img.resize(self.image_size)
                  flat_pixels = np.array(img).flatten().tolist()
                  image_found = True
                  frame_info = {
                      "image" : img,
                      #"image" : flat_pixels,
                      "image_url" : image_url,
                      "tracked_objects" : objects
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

        #check for Valid Github URL
        #----------------------------
        parts = self.repo_url.replace("https://github.com/", "").split("/")
        if len(parts) < 4 or parts[2] != "tree":
            raise ValueError("Repo URL must be in format: https://github.com/<user>/<repo>/tree/<branch>")

        user, repo, _, branch = parts[:4]

        #specify the location of annotations and frames images
        #-------------------------------------------------------
        annotations_path = "annotations"
        frames_path = "frames"

        #API call get annotation XML Files
        #-------------------------------------------------------
        api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{annotations_path}?ref={branch}"
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch annotation folder: {response.status_code}")

        files = response.json()
        xml_files = [f for f in files if f["name"].endswith(".xml")]

        if not xml_files:
            print("No XML annotation files found.")
            return pd.DataFrame()

        # API Calls to get associated frames
        #-----------------------------------------------------------------

        all_dfs = []
        frames_base_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{frames_path}"

        for f in xml_files:
            xml_raw_url = f["download_url"]
            print(f"Reading {f['name']} ...")
            try:
                df = self._parse_cvat_xml_and_frames(xml_raw_url, frames_base_url)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error parsing {f['name']}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        # create a new continuous index (0, 1, 2, …) for the combined DataFrame
        return pd.concat(all_dfs, ignore_index=True)



    # def check_moving_labels(self):
    #     """
    #     Check the 'moving' attribute for each image and track.
    #     Returns a DataFrame with counts of True/False per (image_name, track_id).
    #     """
    #     if self.raw_data.empty:
    #         print("Dataset is empty.")
    #         return pd.DataFrame()

    #     # Normalize moving values to lowercase
    #     df = self.raw_data.copy()
    #     df["moving_norm"] = df["moving"].astype(str).str.lower()

    #     # Count true/false per image_name and track_id
    #     summary = df.groupby(["image_name", "track_id"])["moving_norm"].value_counts().unstack(fill_value=0)

    #     # Optional: rename columns
    #     summary = summary.rename(columns={"true": "True_count", "false": "False_count"})

    #     return summary



    # -------------------------------------------------------------------------
    #  Required Dataset methods for PyTorch
    # -------------------------------------------------------------------------

    def __len__(self):
        """Total number of samples."""
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        """Return one sample (image tensor, label tensor)."""
        row = self.raw_data.iloc[idx]



        # convert flattened pixel array
        # image_array = np.array(row["image"], dtype=np.float32)
        # side_len = int(np.sqrt(len(image_array)))
        # image = image_array.reshape(1, side_len, side_len)
        # image = torch.tensor(image, dtype=torch.float32)

        # # Convert PIL image to a PyTorch Tensor (grayscale, add batch dim if needed later)
        image_np = np.array(row["image"], dtype=np.float32) / 255.0

        # # Add channel dimension (C, H, W) for grayscale images
        image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

        tracked_items = row["tracked_objects"]
        # coordinates as tensor list of [xtl, ytl, xbr, ybr]
        all_coordinates = []
        for item in tracked_items:
            coords_dict = item['coorindates']
            # Extract the values in a specific order: xtl, ytl, xbr, ybr
            coords_list = [coords_dict['xtl'], coords_dict['ytl'], coords_dict['xbr'], coords_dict['ybr']]
            #all_coordinates.append(torch.tensor(coords_list, dtype=torch.float32)) # Append tensors directly
            all_coordinates.append(coords_list)
        #print(all_coordinates)

        # Return a list of coordinate tensors instead of stacking
        #coordinates_list = all_coordinates
        coordinates_list_torch = torch.tensor(all_coordinates, dtype=torch.float32)


        #print(coordinates_tensor)

        # label from 'moving' attribute (0 or 1)
        label_list = [1 if item.lower() == "true" else 0 for item in [item['moving'] for item in tracked_items]]
        #labels_list = [torch.tensor(label, dtype=torch.long) for label in label_list] # Append tensors directly
        labels_list_torch = torch.tensor(label_list, dtype=torch.long)
        #print(label_list)
        #print(labels_tensor)



        return image, labels_list_torch, coordinates_list_torch

repo_url = "https://github.com/khemkandel/research-public/tree/main"
traindata = BaseballData(repo_url, image_size=(28, 28))
loader = DataLoader(traindata, batch_size=1, shuffle=True)


# Get one batch
images, labels, coords = next(iter(loader))
print("Batch shapes:", images.shape, labels,coords)

# Visualize first image in the batch
image_tensor = images[0]            # [1, H, W]
label_tensor = labels[0]

# Convert to numpy for plotting
image_np = image_tensor.squeeze().numpy()  # remove channel dimension

plt.imshow(image_np, cmap="gray")
plt.axis('off')
plt.show()