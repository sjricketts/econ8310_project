# BaseballData PyTorch Dataset

A custom **PyTorch Dataset** designed to handle **baseball video frame extraction and XML-based CVAT annotations**, allowing you to load, preprocess, and visualize baseball tracking data directly from a local or GitHub repository.

---

## Expected Folder Structure

The dataset expects your project folder to be structured as follows:

```
repo/
├─ annotations/
│    ├─ video1.xml
│    └─ video2.xml
├─ videos/
│    ├─ video1.mov
│    └─ video2.mov
└─ frames/
     ├─ video1/
     │   ├─ video1_frame0.jpg
     │   ├─ video1_frame1.jpg
     │   └─ ...
     └─ video2/
         ├─ video2_frame0.jpg
         ├─ video2_frame1.jpg
         └─ ...
```

If `extract_videos=True`, the module automatically extracts frames from `.mov` videos and saves them under the `frames/` directory.

---

## Features

✅ Automatically extracts video frames  
✅ Parses **CVAT XML annotations** for bounding boxes and movement labels  
✅ Converts all images to PyTorch tensors  
✅ Handles **missing frames and invalid data gracefully**  
✅ Supports **batch visualization** with bounding boxes using Matplotlib  
✅ Scales bounding box coordinates to match resized images  

---

## Class Overview

```python
class BaseballData(Dataset):
    """
    PyTorch Dataset for baseball frames and annotations stored locally or on GitHub.
    """
```
### Arguments:
| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| `base_folder` | `str` | — | Base directory containing your dataset |
| `videofolder` | `str` | `'videos'` | Folder with video files |
| `annotation_folder` | `str` | `'annotations'` | Folder containing XML annotations |
| `extract_videos` | `bool` | `False` | Extracts frames from videos if `True` |
| `image_size` | `tuple` | `(28, 28)` | Target image resize size |

---



## Example Usage

```python
from torch.utils.data import DataLoader

base_folder = "C:/path/to/your/project"

# Initialize dataset
traindata = BaseballData(
    base_folder=base_folder,
    videofolder='videos',
    annotation_folder='annotations',
    extract_videos=False,
    image_size=(224, 224)
)

# Create DataLoader
loader = DataLoader(
    traindata,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda x: traindata.collate_fn(x)
)

# Visualize sample batch
traindata.visualize_batch(loader)
```

---

## Visualization

The `visualize_batch()` function displays a batch of images along with their **bounding boxes** and **movement labels**:


## Key Methods

| Method | Description |
|---------|-------------|
| `_frame_extractor()` | Extracts and saves frames from `.mov` videos |
| `_parse_cvat_xml_and_frames()` | Reads CVAT XML files and links them to frame images |
| `_consolidate_from_github_repo()` | Builds the final dataframe of all frames and annotations |
| `collate_fn()` | Custom collate function for PyTorch DataLoader |
| `visualize_batch()` | Displays image batches with bounding boxes and labels |
| `__getitem__()` | Returns image tensor, label tensor, and bounding box coordinates |

---

## Error Handling

The dataset includes several safety checks:
- Throws an error if **no data is found** in the provided folder.  
- Exits with a `FileNotFoundError` if an image file is missing.  
- Raises a `ValueError` if an image is corrupted or empty.  