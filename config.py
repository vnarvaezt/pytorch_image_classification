import os

ROOT_DIR = os.path.abspath(os.curdir)
data_dir = "data/garbage_classification"
batch_size = 100
num_epochs = 1
do_train = False

STEPS = ["TRAIN", "EVALUATE"]

def make_dir(root_path, new_directory):
    try:
        output_path = os.path.join(root_path, new_directory)
        isExist = os.path.exists(output_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(output_path)
    except Exception as error:
        print(error)

make_dir(ROOT_DIR, "output/feature_extraction")
make_dir(ROOT_DIR, "output/freezed_weights")
