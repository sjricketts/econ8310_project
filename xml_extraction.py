# Script to extract attributes needed for dataset from XML files
# track(id, label), box(frame, keyframe, outside, xtl, ytl, xbr, ybr), attribute(name, attribute true/false)

import os
from lxml import etree
import polars as pl
import pickle

annotations_dir = '/Users/hugocorado/Documents/Annotations/'

# Create lists to store data from all files
all_data = {
    "file_name": [],
    "track_id": [],
    "track_label": [],
    "box_frame": [],
    "box_keyframe": [],
    "box_outside": [],
    "box_xtl": [],
    "box_ytl": [],
    "box_xbr": [],
    "box_ybr": [],
    "attribute_value": []
}

# Iterate through all XML files in the directory
for filename in os.listdir(annotations_dir):
    if filename.endswith('.xml'):
        xml_path = os.path.join(annotations_dir, filename)

        tree = etree.parse(xml_path)
        root = tree.getroot()

        file_name = os.path.basename(xml_path).split('.')[0]

        # Iterate through each track
        for track in root.xpath('//track'):
            track_id = track.get('id')
            track_label = track.get('label')

            # Iterate through each box in the track
            for box in track.xpath('.//box'):
                all_data["file_name"].append(file_name)
                all_data["track_id"].append(track_id)
                all_data["track_label"].append(track_label)
                all_data["box_frame"].append(box.get('frame'))
                all_data["box_keyframe"].append(box.get('keyframe'))
                all_data["box_outside"].append(box.get('outside'))
                all_data["box_xtl"].append(box.get('xtl'))
                all_data["box_ytl"].append(box.get('ytl'))
                all_data["box_xbr"].append(box.get('xbr'))
                all_data["box_ybr"].append(box.get('ybr'))

                # Get the "moving" attribute value for this box
                moving_attr = box.xpath('.//attribute[@name="moving"]/text()')
                all_data["attribute_value"].append(moving_attr[0] if moving_attr else None)

# Create single DataFrame from all data
xml_df = pl.DataFrame(all_data)

print(f"Total rows: {len(xml_df)}")
print(xml_df)

# Save the Polars DataFrame to a pickle file
output_path = '/Users/hugocorado/Documents/GitHub/econ8310_project/annotations_data.pkl'

with open(output_path, "wb") as f:
    pickle.dump(xml_df, f)

print(f"Polars DataFrame saved to {output_path}")

# FOR LATER USE:
# To load the DataFrame back from the pickle file:
'''with open(file_path, "rb") as f:
    loaded_df = pickle.load(f)

print("\nLoaded Polars DataFrame:")
print(loaded_df)'''