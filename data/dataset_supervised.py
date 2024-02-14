import os
import tyro
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
NOISE_LEVEL = 0.01

INPUT_NAME = "input"
TARGET_NAME = "target"
INDEX_NAME = "index"

N_BOXES = 81
STATE_SIZE = 4*3 + 4*3 + 6 # 4 goal positions (x, y, z) + 4 leg positions (x, y, z) + velocitites (v, w)

def find_closest_points(A, B):
    """
    Find the closest points in array A for each point in array B.

    Parameters:
    - A: Tensor of shape (NA, 3) representing points.
    - B: Tensor of shape (NB, 3) representing points.

    Returns:
    - index: Tensor of shape (NB) with indexes in tensor A for the closest points of B.
    """
    # Calculate pairwise distances
    if not(torch.is_tensor(A)):
        A = torch.from_numpy(A)
    if not(torch.is_tensor(B)):
        B = torch.from_numpy(B)
    distances = torch.cdist(B, A)

    # Find indices of closest points
    closest_indices = torch.argmin(distances, dim=1)

    # Create one-hot encoding
    one_hot = torch.zeros(B.size(0), A.size(0))
    one_hot.scatter_(1, closest_indices.view(-1, 1), 1)
    return closest_indices

class BoxLocationDataset(Dataset):
    def __init__(self, dataset_dir, return_index:bool=False, augmentation:bool=False, noise_level=NOISE_LEVEL):
        self.dataset_dir = dataset_dir
        self.input_file = os.path.join(self.dataset_dir, INPUT_NAME + ".npy")
        self.target_file = os.path.join(self.dataset_dir, TARGET_NAME + ".npy")
        self.augmentation = augmentation
        self.noise_level = noise_level
        self.return_index = return_index

        inputs = np.load(self.input_file)
        targets = np.load(self.target_file)

        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()
        self.index = self.compute_index(self.inputs, self.targets) if self.return_index else None
 
        print(f"Data loaded. Inputs shape {list(self.inputs.shape)}. Targets shape {list(self.targets.shape)}.")
    
    def compute_index(self, inputs, targets):
        """
        Compute indexes of the target boxes from the input
        """
        assert inputs.shape[0] == targets.shape[0], "Inputs and targets have different number of samples."
        index = torch.zeros((len(inputs), 8), dtype=torch.long)

        for c, (i, t) in enumerate(zip(inputs, targets)):
            i = i[STATE_SIZE:].reshape(-1, 3) # Take only boxes
            t = t.reshape(-1, 3)
            index[c, :] = find_closest_points(i, t) + STATE_SIZE // 3 # Return index within the whole input sequence

        return index

    def __getitem__(self, id):
    
        x = self.inputs[id]

        if self.augmentation: x += torch.randn_like(x) * self.noise_level
        if self.return_index: x = x.reshape(-1, 3)

        y = self.targets[id] if not(self.return_index) else self.index[id]

        return x, y
    
    def __len__(self):
        return len(self.targets)
    
def shuffle_collate(batch):

    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    B, T = targets.shape

    shuffle_indices = torch.hstack((torch.arange(10), torch.randperm(N_BOXES) + 10)).unsqueeze(-1).unsqueeze(0) # Shuffle boxes

    # Shuffle inputs along dimension 1
    shuffled_inputs = torch.take_along_dim(inputs, shuffle_indices, dim=1)

    shuffled_targets = torch.empty_like(targets).long()
    for b, target in enumerate(targets):
        shuffled = torch.tensor([torch.nonzero(shuffle_indices.squeeze() == t, as_tuple=False) for t in target])
        shuffled_targets[b, :] = shuffled.long()

    return shuffled_inputs, shuffled_targets

class DataRecorder():
    def __init__(self, data_directory) -> None:
        self.data_directory = data_directory
        if not(os.path.exists(self.data_directory)):
            os.mkdir(self.data_directory)

        self.data = {}
        self.data_paths = {}

    def load_data_if_exists(self, name):
        data_path = os.path.join(self.data_directory, name + ".npy")
        self.data_paths[name] = data_path
        
        if os.path.exists(data_path):
            self.data[name] = np.load(data_path).tolist()
        else:
            self.data[name] = []

    def check_shape(self):
        passed = True
        for name, data in self.data.items():
            try:
                np.array(data)
            except ValueError:
                print(f"Wrong input data size for {name}")
                passed = False

        return passed
    
    def record(self, name, new_data:np.ndarray) -> None:
        if not(name in self.data):
            self.load_data_if_exists(name)
        self.data[name].append(new_data)
        
    def remove_last(self, n_to_remove) -> None:
        for name, data in self.data.items():
            if len(data) > n_to_remove:
                data = self.target_data[:-n_to_remove]
                data = self.input_data[:-n_to_remove]
            else:
                data = []
                data = []
            self.data[name] = data

    def remove_start(self, n_to_remove) -> None:
        for name, data in self.data.items():
            if len(data) > n_to_remove:
                data = self.target_data[n_to_remove:]
                data = self.input_data[n_to_remove:]
            else:
                data = []
                data = []
            self.data[name] = data

    def save_data(self) -> None:
        if self.check_shape():
            for name, path in self.data_paths.items():
                np.save(path, np.array(self.data[name]))
    
    
def get_dataloaders(data_dir, dataset, batch_size, return_index:bool=False, augmentation:bool=False, shuffle:bool=False, noise_level:float=NOISE_LEVEL):
    train_data_path = os.path.join(data_dir, dataset, "train")
    test_data_path = os.path.join(data_dir, dataset, "test")


    train_dataset = BoxLocationDataset(train_data_path, return_index, augmentation, noise_level)
    if os.path.exists(test_data_path):
        test_dataset = BoxLocationDataset(test_data_path, return_index, augmentation=False)
    else:
        test_dataset = None

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=shuffle_collate if shuffle else None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset != None else None
    
    batch_input, batch_target = next(iter(train_dataloader))

    print(f"Input batch shape {list(batch_input.shape)}. Target batch shape {list(batch_target.shape)}")

    return train_dataloader, test_dataloader

def write_index(
        dataset_dir: str
    ) -> None:
    """
    Write index.npy array that contains index of the boxes in target among the boxes in input. 
    """
    input_file = os.path.join(dataset_dir, INPUT_NAME + ".npy")
    target_file = os.path.join(dataset_dir, TARGET_NAME + ".npy")
    index_file = os.path.join(dataset_dir, INDEX_NAME + ".npy")

    inputs = np.load(input_file)
    targets = np.load(target_file).reshape(inputs.shape[0], 8, 3)

    box_loc = inputs[:, -N_BOXES * 3:].reshape(inputs.shape[0], N_BOXES, 3)

    index_targets = find_closest_points(box_loc, targets)

    np.save(index_file, index_targets)
    return index_targets
    

def split_train_test_data(
        dataset_dir: str,
        train_test_ratio: float = 0.8,
        random_split: bool = False,
        write_id: bool = True
    ) -> None:
    """Split data into 2 train and test directories.

    Args:
        dataset_dir: Initial data directory
        train_test_ratio: ratio of numbre of samples between train and test datasets
    """
    input_file = os.path.join(dataset_dir, INPUT_NAME + ".npy")
    target_file = os.path.join(dataset_dir, TARGET_NAME + ".npy")

    inputs = np.load(input_file)
    targets = np.load(target_file)

    N_data = len(targets)
    N_train = int(train_test_ratio*N_data)
    print("Train dataset size:", N_train, "Test dataset size:", N_data - N_train)

    if (random_split):
        training_idx = np.random.choice(np.arange(N_data), N_train, replace=False)
        test_idx = np.setdiff1d(np.arange(N_data), training_idx)
    else:
        training_idx = np.arange(N_train)
        test_idx = np.arange(N_train, N_data)

    input_train, input_test = inputs[training_idx], inputs[test_idx]
    target_train, target_test = targets[training_idx], targets[test_idx]

    data_type_name = [INPUT_NAME, TARGET_NAME]

    if write_id:
        index = write_index(dataset_dir)
        index_train, index_test = index[training_idx], index[test_idx]
        data_type_name += [INDEX_NAME]

    def create_if_not_exists(path):
        if not(os.path.exists(path)):
            os.mkdir(path)
    
    for mode in ["train", "test"]:
        for data in data_type_name:
            path = os.path.join(dataset_dir, mode)
            create_if_not_exists(path)
            exec(f"np.save(os.path.join(path, data), {data}_{mode})")

if __name__ == "__main__":
    args = tyro.cli(split_train_test_data)
