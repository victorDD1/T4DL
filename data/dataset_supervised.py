import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Toy dataset of length <N> to infer radius of a circle
    given <L> points of the circle
    """
    def __init__(self,
                 N:int=10000,
                 L:int=3) -> None:
        super().__init__()
        self.N = N
        self.points, self.radius = self.sample_sphere(N, L)
    
    def sample_sphere(self, n_sample, n_per_sample):
        radius = torch.rand(n_sample, 1) / 2. + 0.5
        radius_exp = radius.expand(-1, n_per_sample)
        theta = 2 * torch.pi * torch.rand(n_sample, n_per_sample)

        # Convert spherical coordinates to Cartesian coordinates
        x = radius_exp * torch.cos(theta)
        y = radius_exp * torch.sin(theta)
        points = torch.stack([x, y], dim=1).reshape(n_sample, -1)

        return points, radius
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return self.points[index, :], self.radius[index, :]
    
def get_dataloaders(batch_size:int, points_per_circle:int=3):
    """
    Function called to load dataloader in main.
    """
    train_dataset = MyDataset(L=points_per_circle)
    test_dataset = MyDataset(N=300, L=points_per_circle)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # None if no test dataset
    
    batch_input, batch_target = next(iter(train_dataloader))
    print(f"Input batch shape {list(batch_input.shape)}. Target batch shape {list(batch_target.shape)}")

    return train_dataloader, test_dataloader