"""Extract bounding box dimensions from metadata.json to CSV."""

import csv
import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Extract bounding box dimensions from metadata.json to CSV'
    )
    parser.add_argument('output_folder', help='Path to output folder containing metadata.json')
    args = parser.parse_args()

    metadata_path = os.path.join(args.output_folder, 'metadata.json')

    with open(metadata_path) as f:
        metadata = json.load(f)

    csv_path = os.path.join(args.output_folder, 'bounding_boxes.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'bbox_L', 'bbox_W', 'bbox_H'])

        for agg in metadata['agglomerates']:
            bbox = agg['bounding_box']
            writer.writerow([
                agg['filename'],
                bbox['length'],
                bbox['width'],
                bbox['height']
            ])

    print(f"Wrote {csv_path}")


if __name__ == '__main__':
    main()
