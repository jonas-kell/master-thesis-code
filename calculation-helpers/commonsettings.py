import os

scaler_switch = 1e-4


def get_full_file_path(from_file_namne: str):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + from_file_namne + ".json",
    )
