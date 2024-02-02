import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Toy dataset of length <N> where each sample are <L> points
    from a circle. There are <C> different circle radius.
    """
    def __init__(self,
                 N:int=10000,
                 L:int=3,
                 C:int=2) -> None:
        super().__init__()
        self.N = N
        self.points, self.radius = self.sample_circle(N, L, C)
    
    def sample_circle(self, n_sample, n_per_sample, n_radius):
        radius = (torch.randint(n_radius, (n_sample, 1)) + 1.) / n_radius
        radius_exp = radius.expand(-1, n_per_sample)
        theta = 2 * torch.pi * torch.rand(n_sample, n_per_sample)

        x = radius_exp * torch.cos(theta)
        y = radius_exp * torch.sin(theta)
        points = torch.stack([x, y], dim=-1)

        return points, radius
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        item = {
            "data" : self.points[index],
            "condition" : self.radius[index]
        }
        return item
    
def get_dataloaders(batch_size:int, points_per_circle:int=3, n_radius:int=2):
    """
    Function called to load dataloader in main.
    """
    train_dataset = MyDataset(L=points_per_circle, C=n_radius)
    test_dataset = MyDataset(N=300, L=points_per_circle, C=n_radius)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # None if no test dataset
    
    batch = next(iter(train_dataloader))
    print("Train batch shape:")
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_dataloader, test_dataloader