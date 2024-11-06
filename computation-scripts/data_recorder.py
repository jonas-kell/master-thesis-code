import matplotlib.pyplot as plt

data_store = []


def record_data(data):
    if not data_store or len(data_store[-1]) == len(data):
        data_store.append(data)
    else:
        raise ValueError("Each data point must contain the same number of series.")


def plot_data():
    time_points = range(len(data_store))
    transposed_data = list(zip(*data_store))
    for i, series_data in enumerate(transposed_data, start=1):
        plt.plot(time_points, series_data, label=f"Series {i}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    data_store.clear()
