from os import listdir, makedirs
from os.path import join, exists
import numpy as np
import argparse


def merge_text_files(
        output_dir: str,
        name: str,
        in_dirs: list,
):
    print(f"Merge {name}")
    assert name.endswith(".txt")
    elements = []
    for x in in_dirs:
        with open(join(x, name), "r") as f:
            elements += f.readlines()

    elements = [x.replace("\n", "") for x in elements]
    print("    length:", len(elements))
    with open(join(output_dir, name), "w+") as f:
        f.writelines(elements)


def merge_npy_files(
        output_dir: str,
        name: str,
        in_dirs: list,
):
    print(f"Merge {name}")
    assert name.endswith(".npy")
    elements = []
    for i, x in enumerate(in_dirs):
        print(f"    {i}/{len(in_dirs)}", end="\r")
        elements.append(np.load(join(x, name)))

    elements = np.concatenate(elements, axis=0)
    print("    length:", len(elements))
    np.save(join(output_dir, name), elements)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--output_dir", required=True, type=str)
    args = args.parse_args()

    output_dir = args.output_dir

    in_dirs = [output_dir.replace("MERGED", str(x)) for x in range(0, 56)]

    for x in in_dirs:
        assert exists(x)

    if not exists(output_dir):
        makedirs(output_dir)

    # Merge the model files
    merge_text_files(output_dir, "model.txt", in_dirs)

    # Merge the image names
    merge_text_files(output_dir, "image_names.txt", in_dirs)

    # All numpy files
    for x in listdir(in_dirs[0]):
        if x.endswith(".npy"):
            merge_npy_files(output_dir, x, in_dirs)
