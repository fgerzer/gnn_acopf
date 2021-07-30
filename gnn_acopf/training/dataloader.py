from torch_geometric.data.dataloader import DataLoader

class Dataset:
    """
    Simple data class to contain train, val, and test dataset.

    These are set by subclasses, and are also accessible via their alias of
    train, val, and test.
    """
    train_dataset, val_dataset, test_dataset = None, None, None

    @property
    def train(self):
        return self.train_dataset

    @property
    def val(self):
        return self.val_dataset

    @property
    def test(self):
        return self.test_dataset

class GeometricDatasetLoaders:
    """
    Data class for the loaders of one dataset.

    This is essentially a thin wrapper around torch_geometric's DataLoader for three datasets.

    Usually, this will be called using DatasetLoaders.from_dataset.

    Parameters
    ----------
    train_loader, val_loader, test_loader : DataLoader
        The dataloaders to use.
    """
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    @property
    def train(self):
        return self.train_loader

    @property
    def val(self):
        return self.val_loader

    @property
    def test(self):
        return self.test_loader

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """
        Creates a new DatasetLoader from a dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to use.
        kwargs :
            kwargs will be directly piped through the DataLoader class.

        Returns
        -------
        dataset_loaders : DatasetLoaders
            The dataset loaders created.
        """
        train_loader = DataLoader(dataset.train, **kwargs)
        val_loader = DataLoader(dataset.val, **kwargs)
        test_loader = DataLoader(dataset.test, **kwargs)
        dataset_loaders = cls(train_loader, val_loader, test_loader)
        return dataset_loaders
