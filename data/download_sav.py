import os
import csv
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", required=True,
        help='File with the links to download. The file should be a csv file with the columns "file_name" and "cdn_link".'
    )
    parser.add_argument(
        "--output-dir", required=True,
        help='Directory where the files will be saved.'
    )
    return parser.parse_args()


def download(input_file, output_dir):
    print("Downloading files...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(input_file, mode='r') as file:
        reader = csv.DictReader(file,
                                delimiter='\t')
        print(reader)
        for row in reader:
            file_name = row['file_name']
            cdn_link = row['cdn_link']
            print(f"Downloading {file_name} from {cdn_link}")
            os.system(f'wget -O {output_dir}/{file_name} "{cdn_link}"')

    print("All files have been downloaded."
            f"Check the files in {output_dir}")


def main():
    args = get_args()
    download(args.input_file, args.output_dir)

    
if __name__ == "__main__":
    main()