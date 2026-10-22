import h5py
import logging

logger = logging.getLogger(__name__)


class HDF5Saver:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path: str = file_path
        self.encoding = encoding
        self.file = h5py.File(self.file_path, "a")

    def save(self, data: dict, flag: str | None = None):
        for dataset_name, dataset_value in data.items():
            if dataset_name in self.file:
                del self.file[dataset_name]

            if isinstance(dataset_value, str):
                dataset_value = dataset_value.encode(self.encoding)

            ds = self.file.create_dataset(dataset_name, data=dataset_value)
            if flag:
                ds.attrs["flag"] = flag

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class HDF5Loader:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path: str = file_path
        self.encoding: str = encoding
        self.file = h5py.File(self.file_path, "r")

    def load(self) -> dict:
        dataset: dict = {}
        for name in self.file:
            data = self.file[name]
            if data.ndim == 0:
                value = data[()]
                if isinstance(value, bytes):
                    dataset[name] = value.decode(self.encoding)
                else:
                    dataset[name] = value
            else:
                dataset[name] = data[:]
        return dataset

    def load_flags(self) -> dict:
        flags: dict = {}
        for name in self.file:
            data = self.file[name]
            if "flag" in data.attrs:
                flags[name] = data.attrs["flag"]
        return flags

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="hdf5_manager.log",
        filemode="a",
    )
    with HDF5Loader("./data/texts.h5", encoding="utf-8") as loader:
        data = loader.load()
        # print("Loaded data:", data)

    with HDF5Saver("./data/texts_eng.h5", encoding="utf-8") as saver:
        saver.save(data)
        print("Data saved successfully.")

    with HDF5Loader("./data/texts_eng.h5", encoding="utf-8") as loader:
        data = loader.load()
        print("Loaded data from texts_eng.h5:", data)