import csv
import logging
import os
import sys
import torch

# *******************************************************************************
# Storage utils
# *******************************************************************************

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_storage_dir():
    return "storage"

def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)

def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")

def get_vocab(model_dir, device="cpu"):
    return get_status(model_dir, device)["vocab"]

# *******************************************************************************
# Status utils
# *******************************************************************************

def get_status(model_dir, device="cpu"):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)

def get_model_state(model_dir, device="cpu"):
    return get_status(model_dir, device)["model_state"]

def save_status(status, model_dir, nsteps=None):
    create_folders_if_necessary(model_dir)
    if nsteps:
        path = os.path.join(model_dir, "models", f"{str(nsteps)}.pt")
    status_path = get_status_path(model_dir)
    
    # Save nsteps model inside models folder
    create_folders_if_necessary(path)
    torch.save(status, path)
    # Save current status model output models folder
    torch.save(status, status_path)


# *******************************************************************************
# Logging utils
# *******************************************************************************

def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)