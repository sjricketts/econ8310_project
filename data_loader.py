import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import requests
import xml.etree.ElementTree as ET
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
        response.raise_for_status()
        tree = ET.parse(BytesIO(response.content))
        root = tree.getroot()

        base_name = os.path.splitext(os.path.basename(xml_url))[0]
        records = []

        # Cache to store already loaded flattened frames
        frame_cache = {}

        for track in root.findall("track"):
            track_id = int(track.get("id"))
            label = track.get("label")

            for box in track.findall("box"):
                frame = int(box.get("frame"))

                # Load & flatten image only once per frame
                if frame not in frame_cache:
                    flat_pixels = None
                    image_found = False
                    for ext in [".jpg", ".jpeg", ".png"]:
                        image_url = f"{frames_base_url.rstrip('/')}/{quote(base_name)}/{quote(base_name)}_frame{frame}{ext}"
                        img_resp = requests.get(image_url)
                        if img_resp.status_code == 200:
                            img = Image.open(BytesIO(img_resp.content)).convert("L")
                            img = img.resize(self.image_size)
                            flat_pixels = np.array(img).flatten().tolist()
                            image_found = True
                            break

                    if not image_found:
                        print(f"Missing frame image for {base_name}_frame{frame}")
                        continue

                    frame_cache[frame] = flat_pixels
                else:
                    flat_pixels = frame_cache[frame]  # reuse cached pixels

                # Append annotation record
                moving_attr = box.find("attribute[@name='moving']")
                moving = moving_attr.text.strip().lower() if moving_attr is not None else None

                records.append({
                    "track_id": track_id,
                    "label": label,
                    "frame": frame,
                    "xtl": float(box.get("xtl")),
                    "ytl": float(box.get("ytl")),
                    "xbr": float(box.get("xbr")),
                    "ybr": float(box.get("ybr")),
                    "moving": moving,
                    "annotation_file": base_name,
                    "image_name": f"{base_name}_frame{frame}",
                    "image_url": image_url,
                    "pixels": flat_pixels
                })

        return pd.DataFrame(records)

    def _consolidate_from_github_repo(self):
        """Fetch all XML files from 'annotations' folder and process them."""
        parts = self.repo_url.replace("https://github.com/", "").split("/")
        if len(parts) < 4 or parts[2] != "tree":
            raise ValueError("Repo URL must be in format: https://github.com/<user>/<repo>/tree/<branch>")

        user, repo, _, branch = parts[:4]
        annotations_path = "annotations"
        frames_path = "frames"

        api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{annotations_path}?ref={branch}"
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch annotation folder: {response.status_code}")

        files = response.json()
        xml_files = [f for f in files if f["name"].endswith(".xml")]

        if not xml_files:
            print("No XML annotation files found.")
            return pd.DataFrame()

        all_dfs = []
        frames_base_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{frames_path}"

        for f in xml_files:
            xml_raw_url = f["download_url"]
            print(f"📄 Reading {f['name']} ...")
            try:
                df = self._parse_cvat_xml_and_frames(xml_raw_url, frames_base_url)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error parsing {f['name']}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)


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
        image_array = np.array(row["pixels"], dtype=np.float32)
        side_len = int(np.sqrt(len(image_array)))
        image = image_array.reshape(1, side_len, side_len)
        image = torch.tensor(image, dtype=torch.float32)

        # coordinates as tensor [xtl, ytl, xbr, ybr]
        coords = torch.tensor([row["xtl"], row["ytl"], row["xbr"], row["ybr"]], dtype=torch.float32)

        # label from 'moving' attribute (0 or 1)
        label = 1 if str(row["moving"]).lower() == "true" else 0
        label = torch.tensor(label, dtype=torch.long)

        return image, label, coords

repo_url = "https://github.com/khemkandel/research-public/tree/main"
traindata = BaseballData(repo_url, image_size=(28, 28))
loader = DataLoader(traindata, batch_size=8, shuffle=True)

# Get one batch
images, labels, coords = next(iter(loader))
print("Batch shapes:", images.shape, labels, coords)

# Visualize first image in the batch
image_tensor = images[0]            # [1, H, W]
label_tensor = labels[0]

# Convert to numpy for plotting
image_np = image_tensor.squeeze().numpy()  # remove channel dimension

plt.imshow(image_np, cmap="gray")
plt.title(f"Label (moving): {label_tensor.item()}")
plt.axis('off')
plt.show()
