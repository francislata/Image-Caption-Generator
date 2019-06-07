from dataset.dataset import COCODataset
import argparse

def _parse_arguments():
    """Parses the command line arguments"""

    args_parser = argparse.ArgumentParser(description='Neural Image Caption Generator')

    # Required arguments
    args_parser.add_argument('train_images_file_path', type=str, help='the location of the training dataset images')
    args_parser.add_argument('valid_images_file_path', type=str, help='the location of the validation dataset images')
    args_parser.add_argument('train_csv_file_path', type=str, help='the location of the training CSV file mapping image filenames to captions')
    args_parser.add_argument('valid_csv_file_path', type=str, help='the location of the validation CSV file mapping image filenames to captions')

    return args_parser.parse_args()

def main(args):
    print("Creating training and validation datasets...")

    train_ds = COCODataset(args.train_images_file_path, args.train_csv_file_path)
    valid_ds = COCODataset(args.valid_images_file_path, args.valid_csv_file_path)
    
    print("Done!\n")
    

if __name__ == '__main__':
    args = _parse_arguments()
    main(args)
