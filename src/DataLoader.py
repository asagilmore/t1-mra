class DataLoader():
    def __init__(self, dataset, batchsize):
        for i in range(0, dataset.len(), batchsize):
            self.images, self.masks = dataset.getslices()
            yield
            
    def create_batches(dataset, batch_size, shuffle=True):
        data_size = len(dataset)
        indices = list(range(data_size))

        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = [dataset[i] for i in batch_indices]
            images, labels = zip(*batch)
            
            images = torch.stack(images, dim=0)
            labels = torch.tensor(labels)
            
            yield images, labels
