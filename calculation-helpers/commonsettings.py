import os

scaler_switch = 1e2


def get_full_file_path(from_file_name: str):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + from_file_name + ".json",
    )


def get_newest_file_name():
    folder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/",
    )

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    if not files:
        return None
    newest_file = max(
        files, key=lambda f: os.path.getctime(os.path.join(folder_path, f))
    )
    return os.path.splitext(newest_file)[0]


def get_filenames_containing(test: str):
    folder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/",
    )

    res = []

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    if not files:
        return []
    for file in files:
        path = os.path.splitext(file)[0]
        if test in path:
            res.append(path)

    return res
