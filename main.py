import argparse
import os
import pathlib
import re
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from dataset import build_clean_dataset
from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
# if load_local is given, will eval the trained model
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
# the data path have been modified on our server
parser.add_argument('--data_path', default='/remote-home/iot_lishizhong/Datasets/', help='Place to load dataset (default: ./dataset/)')
# we use cuda:1 for training
parser.add_argument('--device', default='cuda:1', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

# retrain settings
parser.add_argument('--retrain', type=bool, default=True, help='Load trained model, if we load the local model')
parser.add_argument('--retrain_ratio', type=float, default=0.1, help='The Ratio of clean data used for retraining')
parser.add_argument('--retrain_epochs', type=int, default=20, help='The retraining epochs')
args = parser.parse_args()

def main():
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    
    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()  # used the cross entropy loss not the config in args
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    basic_model_path = "./checkpoints/badnet-%s.pth" % args.dataset
    start_time = time.time()
    if args.load_local:
        print("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
        if args.retrain:
            print(f'The model would be retrained for {args.retrain_epochs} to clean the back-door')
            retrain_model_path = "./checkpoints/badnetRrtrain-%s.pth" % args.dataset
            trainset_CL, testset_CL = build_clean_dataset(args)
            train_loaderCL = DataLoader(trainset_CL, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            stats = []
            for epoch in range(args.retrain_epochs):
                train_stats = train_one_epoch(train_loaderCL, model, criterion, optimizer, args.loss, device)
                test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
                print(
                    f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")

                # save model
                torch.save(model.state_dict(), retrain_model_path)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             }

                # save retraining stats
                stats.append(log_stats)
                df = pd.DataFrame(stats)
                df.to_csv("./logs/%s_trigger%d_Retrain.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            # save model 
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()
