import random, argparse, torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
from model import *
from datetime import datetime
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from data_preprocessing.preprocess import SERDataset
# from torch.nn.utils import clip_grad_norm_
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import *
# from torch.optim.swa_utils import AveragedModel
# import custom_loss_fn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Set random seed for reproducibility
# random.seed(42)
# torch.manual_seed(42)
# np.random.seed(42)
random.seed(2048)
torch.manual_seed(2048)
np.random.seed(2048)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-i', '--input_dir', type=str, default='./data_preprocessing/output', help='Path to the preprocessed dataset')
    parser.add_argument('--dataset_fn', type=str, default='dataset.pt', help='Filename of the preprocessed dataset')
    parser.add_argument('-o', '--output_dir', type=str, default='./model_ckpts', help='Path to save the trained model')
    parser.add_argument('--input_size', type=int, default=26, help='Dimension of the input features')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--pos_enc', type=bool, default=True, choices=[True, False], help='Whether adding positional encoding to the input features')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01, help='Weight decay')
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
    emotion = torch.tensor(emotion, dtype=torch.long)
    
    return {
        'idx': indexes, 
        'valence': valence, 
        'activation': activation, 
        'emotion': emotion,
        'features': padded_features
        }


def train_loop(dataloader, 
               model, 
               loss_fn, 
               optimizer, 
               scheduler
              ):
    size = len(dataloader.dataset)
    # print("Train set size:", size)
    
    model.train()
    losses = []
    
    for idx, batch in enumerate(dataloader):
        X = batch['features'].to(device)
        y = batch['emotion'].to(device)
        # print("X: ", X, '\n')
        # print("y: ", y, '\n')
        
        # Compute prediction and loss
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses.append(loss)

        # Backpropagation
        # (1) Backpropagate the prediction loss
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=0.5)
        # (2) Update the model's weights
        optimizer.step()
        # averaged_model.update_parameters(model)
        # (3) Reset the gradient
        optimizer.zero_grad()
        # Apply learning rate scheduler
        scheduler.step()
        
        # Log the training loss to TensorBoard
        # writer.add_scalar('Training Loss', loss, idx)

        # Print the loss every 10 batches
        if idx % 10 == 0:
            loss, current = loss.item(), idx * idx + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return losses


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # print("Dev set size:", size)
    
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    losses = []
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            X = batch['features'].to(device)
            y = batch['emotion'].to(device)
            
            # Compute prediction and loss
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            losses.append(loss)
            test_loss += loss.item()
            
            # Compute the number of correct predictions
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    # Print test error
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg validation loss: {test_loss:>8f} \n")
    
    return losses


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    dataset_fn = args.dataset_fn
    output_dir = args.output_dir
    pos_enc = args.pos_enc
    if pos_enc:
        pos_enc_mode = 'pos_enc'
    else:
        pos_enc_mode = 'simple'

    print("input_dir:", input_dir)
    print("dataset_fn:", dataset_fn)
    print("output_dir:", output_dir)
    print("input_size:", args.input_size)
    print("num_heads:", args.num_heads)
    print("num_encoder_layers:", args.num_encoder_layers)
    print("num_classes:", args.num_classes)
    print("pos_enc:", args.pos_enc)
    print("num_epochs:", args.num_epochs)
    print("batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", args.weight_decay)
    
    # Load the dataset
    dataset = torch.load(f"{input_dir}/{dataset_fn}")

    # Create DataLoaders for the training and dev sets
    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Instantiate the model
    model = TransformerModel(
        input_dim=args.input_size, 
        num_heads=args.num_heads, 
        num_encoder_layers=args.num_encoder_layers, 
        output_dim=args.num_classes,
        mode=pos_enc_mode,
        device=device,
    ).to(device)
    print(summary(model), '\n')

    ## Use weighted loss to deal with unbalanced label distribution
    # Count the 4 class labels in training set
    train_counter = dict(Counter(dataset['train'].emotion.tolist()))
    print("Label distribution in training set:\n", train_counter)

    # Compute the weight for each class
    weights = [len(dataset['train']) / (args.num_classes * x) for x in train_counter.values()]
    class_weights = torch.FloatTensor(weights).to(device)
    print("Class weights:", class_weights, '\n')

    # Loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = custom_loss_fn.FocalLoss(alpha=class_weights, gamma=2)
    # loss_fn = nn.MultiMarginLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr=learning_rate, 
    #                             momentum=0.5, 
    #                             dampening=0, 
    #                             weight_decay=weight_decay)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ## Use learning rate scheduler
    # scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    # scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    scheduler3 = LinearLR(optimizer, start_factor=0.5)
    # scheduler4 = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler5 = ConstantLR(optimizer, factor=0.5)

    # averaged_model = AveragedModel(model)

    # writer = SummaryWriter('runs/ser_train_1_18')

    for t in range(args.num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses = train_loop(dataloader=train_loader, 
                                #   model=averaged_model, 
                                model=model, 
                                loss_fn=loss_fn, 
                                optimizer=optimizer, 
                                scheduler=scheduler3
                                )
        
        val_losses = test_loop(dataloader=dev_loader, 
                            #    model=averaged_model, 
                               model=model, 
                               loss_fn=loss_fn
                               )
                            
    print("Done!")
    # writer.close()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_checkpoint_name = f"model_{timestamp}.pth"
    torch.save(model, f"{output_dir}/{model_checkpoint_name}")
    # torch.save(averaged_model, f"{output_dir}/{model_checkpoint_name}")
    print("Model saved!")


if __name__ == "__main__":
    main()