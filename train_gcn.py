import argparse
import math
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_utils import Tokenizer4Bert, DatasetReader
from models import BERTGCN, BERTGAT
import warnings
import logging

warnings.filterwarnings('ignore')

# Set up logging to save logs to a file and print to stdout
logging.basicConfig(filename="bertgcn.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        """Initialize the instructor with model, dataset, and data loaders."""
        self.opt = opt

        # Initialize BERT tokenizer and model from pretrained BERT model
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)

        # Load the training and test datasets
        self.trainset = DatasetReader(opt.dataset_file['train'], tokenizer)
        self.testset = DatasetReader(opt.dataset_file['test'], tokenizer)

        # If validation ratio is specified, split the trainset into train and validation sets
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        # Create data loaders for train, validation, and test sets
        self.val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        self.train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        # Initialize the appropriate model based on user selection (BERTGCN or BERTGAT)
        if opt.model_name == 'bertgcn':
            self.model = opt.model_class(bert, opt).to(opt.device)
        elif opt.model_name == 'bertgat':
            self.model = opt.model_class(bert, opt.embed_dim).to(opt.device)

        # If running on CUDA, log the GPU memory allocated
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

        # Print model parameters and training configuration
        self._print_args()

    def _print_args(self):
        """Print out the model's trainable and non-trainable parameters, along with the training configuration."""
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        """Reset model parameters (excluding BERT parameters) using the specified initializer."""
        for child in self.model.children():
            if type(child) != BertModel:  # Skip BERT parameters
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)  # Initialize using the selected initializer
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)  # Uniform initialization

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        """Train the model, evaluate on validation set, and save the best model."""
        max_val_f1 = 0
        global_step = 0
        path = None

        epochTrainAcc = []
        epochValAcc = []

        # Loop over epochs for training
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('Epoch : {}'.format(i_epoch + 1))
            n_correct, n_total, loss_total, counter, tot_train_acc = 0, 0, 0, 0, 0

            # Switch model to training mode
            self.model.train()
            for _, batch in enumerate(train_data_loader):
                global_step += 1

                # Zero out gradients before backward pass
                optimizer.zero_grad()

                # Prepare inputs and targets for the model
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['label'].to(self.opt.device)

                # Compute loss and backpropagate
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Calculate accuracy and total loss for logging
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                # Log training accuracy and loss periodically
                if global_step % self.opt.log_step == 0:
                    counter += 1
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    tot_train_acc += train_acc
                    logger.info('Loss : {:.4f}, Accuracy : {:.4f}'.format(train_loss, train_acc))

            # Evaluate on the validation set
            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)

            epochValAcc.append(val_acc)
            epochTrainAcc.append(tot_train_acc / counter)

            logger.info('> Validation Accuracy : {:.4f}, Validation F1 Score : {:.4f}'.format(val_acc, val_f1))

            # Save the best model if validation F1 score improves
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                max_val_epoch = i_epoch
                if not os.path.exists('saved_models'):
                    os.mkdir('saved_models')

                path = 'saved_models/{0}_{1}{2}'.format(self.opt.model_name, self.opt.dataset, ".pkl")
                torch.save(self.model.state_dict(), path)
                logger.info('>> Best model saved {}'.format(path))

            # Early stopping if validation F1 hasn't improved in 'patience' epochs
            if i_epoch - max_val_epoch >= self.opt.patience:
                logger.info('>> Early stopping!')
                break

        # Plot training and validation accuracy over epochs
        epoch_count = range(1, len(epochTrainAcc) + 1)
        plt.plot(epoch_count, epochTrainAcc, 'r-')
        plt.plot(epoch_count, epochValAcc, 'b-')
        plt.legend(['Train Accuracy', 'Validation Accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

        return path

    def _evaluate_acc_f1(self, data_loader):
        """Evaluate the model's accuracy and F1 score on the given data loader."""
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        # Switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for _, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                # Calculate accuracy
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                # Collect outputs and targets for calculating F1 score
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        # Compute accuracy and macro F1 score
        acc = n_correct / n_total
        f1 = metrics.f1_score(
            t_targets_all.cpu(),  # Move target tensor to CPU
            torch.argmax(t_outputs_all, -1).cpu(),  # Move output tensor to CPU
            labels=np.unique(torch.argmax(t_outputs_all.cpu(), -1)),  # Ensure unique labels on CPU
            average='macro'  # Use macro F1 score for multi-class classification
        )
        return acc, f1

    def run(self):
        """Run the entire training and evaluation pipeline."""
        # Set loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        # Reset model parameters
        self._reset_params()

        # Train the model and get the path of the best model
        best_model_path = self._train(criterion, optimizer, self.train_data_loader, self.val_data_loader)

        # Load the best model and evaluate on the test set
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(self.test_data_loader)
        logger.info('>> Test Accuracy : {:.4f}, Test F1 Score : {:.4f}'.format(test_acc, test_f1))


def main():
    """Main function to parse arguments and start the training process."""
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bertgcn', type=str)
    parser.add_argument('--dataset', default='riloff', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=20, type=int)
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--valset_ratio', default=0, type=float)
    opt = parser.parse_args()

    # Set random seeds for reproducibility
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    # Model and optimizer selection
    model_classes = {
        'bertgcn': BERTGCN,
    }

    # Dataset paths
    dataset_files = {
        'headlines': {
            'train': './datasets/headlines/train.raw',
            'test': './datasets/headlines/test.raw'
        },
        'riloff': {
            'train': './datasets/riloff/train.raw',
            'test': './datasets/riloff/test.raw'
        },
    }

    # Input columns based on the model type
    input_colses = {
        'bertgcn': ['text_bert_indices', 'bert_segments_indices', 'dependency_graph', 'affective_graph'],
    }

    # Initializers and optimizers
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    # Set options based on user input
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # Initialize and run the instructor
    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
