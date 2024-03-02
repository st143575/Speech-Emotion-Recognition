import argparse, random, json, torch
import numpy as np
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from model import *
from torchsummary import summary
from data_preprocessing.preprocess import SERDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the pretrained model')
    parser.add_argument('-i', '--input_dir', type=str, default='./data_preprocessing/output', help='Path to the pretrained model')
    parser.add_argument('-o', '--output_fn', type=str, help='Path + file name of the predictions')
    parser.add_argument('-m', '--model_fn', type=str, help='Path + file name of the pretrained model')
    arser.add_argument('--mode', type=str, choices=['test', 'final'], help="Whether the evaluation is done on ser_test_1.json ('test') or ser_test_2.json ('final')")
    return parser.parse_args()


def custom_collate_fn(batch):
    features = [item['features'] for item in batch]
    indexes = [item['idx'] for item in batch]
    valence = [item['valence'] for item in batch]
    activation = [item['activation'] for item in batch]
    emotion = [item['emotion'] for item in batch]
    
    # Apply zero-padding to features
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    
    # Convert the emotion list to a LongTensor (64-bit integer)
    #valence = torch.tensor(valence, dtype=torch.long)
    #activation = torch.tensor(activation, dtype=torch.long)
    emotion = torch.tensor(emotion, dtype=torch.long)
    
    return {
        'idx': indexes, 
        'valence': valence, 
        'activation': activation, 
        'emotion': emotion,
        'features': padded_features
        }


def predict(dataloader, model):
    size = len(dataloader.dataset)
    print("Test set size:", size)
    
    indexes = []
    y_preds = []
    
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            indexes.append(str(batch['idx'][0]))

            X = batch['features'].to(device)
            y = batch['emotion'].to(device)
            
            # Compute prediction and loss
            y_pred = model(X)
            y_preds.append(y_pred)
            
            # Compute the number of correct predictions
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            
    correct /= size
    # Print test error
    print(f"Test Error:\nAccuracy: {(100*correct):>0.1f}")
    
    y_preds = [y_pred.to('cpu') for y_pred in y_preds]
    y_pred_emotion_labels = [y_pred.argmax(1).item() for y_pred in y_preds]
    
    return indexes, y_pred_emotion_labels


def postprocess(indexes: list, y_pred_emotion_labels: list):
    """
    Post-processing of the model prediction.
    Map the 1-dimensional emotion label bach to the two-dimensional (valence, activation) label.

    Args:
        indexes (list): List of indexes of the test samples.
        y_pred_emotion_labels (list): List of predicted emotion labels.

    Returns:
        output (dict): A dictionary with the test sample indexes as keys and the corresponding (valence, activation) labels as values.
    """
    
    # Map the emotion label to valence and activation labels.
    mapping = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }
    y_pred_val_act_labels = [mapping[emotion_label] for emotion_label in y_pred_emotion_labels]
    
    output = {k: {'valence': v[0], 'activation': v[1]} for k, v in zip(indexes, y_pred_val_act_labels)}
    
    return output


def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_fn = Path(args.output_fn)
    model_fn = Path(args.model_fn)
    mode = args.mode
    print("Input directory:", input_dir, "\n")
    print("Output fiele name:", output_fn, "\n")
    print("Model:", model_fn)
    print("Mode:", mode)
    
    # Load preprocessed dataset
    dataset = torch.load(f"{input_dir}/dataset.pt")
    
    # Load test data
    if mode == 'test':
        test_loader = DataLoader(dataset['test_1'], batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    elif mode == 'final':
        test_loader = DataLoader(dataset['test_2'], batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load pre-trained model
    model = torch.load(f"{model_fn}")
    
    # Do inference and get model outputs
    print("Evaluation starts...")
    indexes, y_preds_emotion = predict(test_loader, model)
    
    # Post-process model outputs
    output = postprocess(indexes, y_preds_emotion)
    
    with open(output_fn, 'w') as file:
        json.dump(output, file)
    
    print("Evaluation done!")

    
if __name__ == "__main__":
    main()












