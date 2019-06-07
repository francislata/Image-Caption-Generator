from collections import defaultdict
from pycocotools.coco import COCO
import os
import argparse
import pandas as pd

def _parse_arguments():
    """Parses the command line arguments"""

    arg_parser = argparse.ArgumentParser(description='MSCOCO Caption Dataset Creation Tool')

    # Required arguments
    arg_parser.add_argument('train_images_file_path', type=str, help='the file path of the training images')
    arg_parser.add_argument('valid_images_file_path', type=str, help='the file path of the validation images')
    arg_parser.add_argument('train_captions_file_path', type=str, help='the file path of the training captions JSON file')
    arg_parser.add_argument('valid_captions_file_path', type=str, help='the file path of the validation captions JSON file')
    arg_parser.add_argument('train_csv_filename', type=str, help='the filename to use for the training dataset CSV file')
    arg_parser.add_argument('valid_csv_filename', type=str, help='the filename to use for the validation dataset CSV file')

    return arg_parser.parse_args()

def _create_dataset(images_file_path, captions_file_path, csv_filename):
    """Creates a dataset and saves it in a CSV file"""

    coco_captions = COCO(captions_file_path)
    image_captions_mapping = defaultdict(list)
    image_ids = []
    image_filenames = []
    captions = []

    for image_caption in coco_captions.loadAnns(coco_captions.getAnnIds()):
        image_captions_mapping[image_caption['image_id']].append(image_caption['caption'])

    for image_id, caption in image_captions_mapping.items():
        image_ids.append(image_id)
        captions.append(caption)

    for image_id in image_ids:
        for image_filename in os.listdir(images_file_path):
            if str(image_id) in image_filename:
                image_filenames.append(image_filename)
                break

    df = pd.DataFrame({'image_filenames': image_filenames, 'captions': captions})
    df.to_csv(csv_filename, index=False)

def main(args):
    print("Creating training and validation datasets...")

    _create_dataset(args.train_images_file_path, args.train_captions_file_path, args.train_csv_filename)
    _create_dataset(args.valid_images_file_path, args.valid_captions_file_path, args.valid_csv_filename)

    print("Done!")


if __name__ == '__main__':
    args = _parse_arguments()
    main(args)
