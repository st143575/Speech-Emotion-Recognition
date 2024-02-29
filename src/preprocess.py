import os, argparse, torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import ChainDataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for SER')
    parser.add_argument('-i', '--input_dir', type=str, default='../dataset', help='Path to the raw dataset')
    parser.add_argument('-o', '--output_dir', type=str, default='./data', help='Path to save the preprocessed data')
    return parser.parse_args()

class SERDataset(Dataset):
    """
    Create a Dataset object from ser_train, ser_dev and ser_test_1.

    Args:
        data (pandas.DataFrame): ['valece', 'activation', 'features'].
    """

    def __init__(self, data):
        self.valence = data['valence']
        self.activation = data['activation']
        self.features = data['features']

    def __len__(self):
        return len(self.valence)

    def __getitem__(self, idx):
        sample = {
            'valence': self.valence[idx],
            'activation': self.activation[idx],
            'features': torch.FloatTensor(self.features[idx])
        }
        return sample

def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory: ", input_dir, "\n")
    print("Output directory: ", output_dir)
    
    # Load training data
    ser_train = pd.read_json(f"{input_dir}/train.json")
    ser_train = ser_train.transpose()

    # Load dev data
    ser_dev = pd.read_json(f"{input_dir}/dev.json")
    ser_dev = ser_dev.transpose()
    ser_dev = ser_dev[['valence', 'activation', 'features']]

    # Load test data for submission test (ser_test_1.json)
    ser_test_1 = pd.read_json(f"{input_dir}/ser_test_1.json")
    ser_test_1 = ser_test_1.transpose()
    # Add a placehoder -1 for valence and activation
    ser_test_1.insert(0, 'valence', -1)
    ser_test_1.insert(1, 'activation', -1)

    # Instantiate the Dataset object for train and dev splits.
    train_set = SERDataset(ser_train)
    dev_set = SERDataset(ser_dev)
    test_set_1 = SERDataset(ser_test_1)

    # print('train:\n', train_set.__getitem__(0), '\n')
    # print('dev:\n', dev_set.__getitem__(0), '\n')
    # print('test:\n', test_set_1.__getitem__(0), '\n')

    merged_set = ChainDataset([train_set, dev_set, test_set_1])
    dataset = {
        'train': train_set, 
        'dev': dev_set, 
        'test': test_set_1,
        }

    # Save dataset to disk
    torch.save(dataset, f"{output_dir}/dataset.pt")
    print("Data preprocessing done!")


if __name__ == "__main__":
    main()