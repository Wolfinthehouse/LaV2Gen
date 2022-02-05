import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from time import time
from os.path import join as osj
from model import GPT2
from dataloader import VisualCOMETDataset
from dataloader import VisCOMETEvaluationDataset
from dataloader import ATOMICDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from utils import str2bool, print_log, csv2list
from utils import setup_logger, compute_eval_loss, compute_eval_metrics
import math


def main():
    parser = argparse.ArgumentParser(description='Visual COMET: Train + Eval')

    # Experiment params
    parser.add_argument('--mode',           type=str,       help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir',       type=str,       help='root directory to save model & summaries')
    parser.add_argument('--expt_name',      type=str,       help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name',       type=str,       help='expt_dir/expt_name/run_name: organize training runs')
    parser.add_argument('--text_only',      type=str2bool,  help='Excludes visual modality (T/F)', default='F')

    # Model params
    parser.add_argument('--model',          type=str,       help='Transformer backbone', default='gpt2')
    parser.add_argument('--max_text_len',   type=int,       help='Max input text sequence length', default=64)
    parser.add_argument('--im_size',        type=int,       help='input image size', default=224)
    parser.add_argument('--patch_size',     type=int,       help='image patch size', default=16)

    # Data params
    parser.add_argument('--data_dir',       type=str,       help='raw dataset directory', required=True)

    # Training params
    parser.add_argument('--lr',             type=float,     help='learning rate', default=1e-5)
    parser.add_argument('--epochs',         type=int,       help='number of epochs', default=50)
    parser.add_argument('--batch_size',     type=int,       help='batch size', default=8)
    parser.add_argument('--ckpt',           type=str,       help='path to model checkpoint .pth file')
    parser.add_argument('--save',           type=str2bool,  help='whether to save models', default='T')
    parser.add_argument('--val_size',       type=int,       help='validation set size for evaluating metrics', default=2048)
    parser.add_argument('--log_interval',   type=int,       help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval',  type=int,       help='save model after `n` weight update steps', default=30000)

    # GPU params
    parser.add_argument('--gpu_ids',        type=str,       help='GPU Device ID', default='0')
    parser.add_argument('--use_amp',        type=str2bool,  help='Automatic-Mixed Precision (T/F)', default='T')

    # Misc params
    parser.add_argument('--num_workers',    type=int,       help='number of worker threads for Dataloader', default=1)
    parser.add_argument('--csv_out',        type=str,       help='Path: save `Test` results to csv', default=1)

    # Parse Args
    args = parser.parse_args()

    # GPU device
    device_ids = csv2list(args.gpu_ids, cast=int)
    device = torch.device('cuda:{}'.format(device_ids[0]))

    print('GPUs: {}'.format(device_ids))

    # Configs
    lr = args.lr
    n_epochs = args.epochs
    batch_size = args.batch_size

    # Dataset Params
    data_params = dict(max_text_len=args.max_text_len,
                       tok_name=args.model)
                    #    im_size=args.im_size,
                    #    patch_size=args.patch_size,
                    #    text_only=args.text_only)

    # Train
    if args.mode == 'train':
        # Setup train log directory
        log_dir = osj(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logger(parser, log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # Dataset
        # train_dataset = VisualCOMETDataset(args.data_dir, split='train', **data_params)
        train_dataset = ATOMICDataset(args.data_dir, split='train', **data_params)

        # val_loss_dataset = VisualCOMETDataset(args.data_dir, split='val', **data_params)
        val_loss_dataset = ATOMICDataset(args.data_dir, split='val', **data_params)
        #val_metrics_dataset = VisCOMETEvaluationDataset(args.data_dir, split='val', **data_params)

        # Dataloader
        loader_params = dict(batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=args.num_workers)

        train_loader = DataLoader(train_dataset, **loader_params)
        val_loss_loader = DataLoader(val_loss_dataset, **loader_params)
        #val_metrics_loader = DataLoader(val_metrics_dataset, **loader_params)

        # Print split sizes
        train_size = train_dataset.__len__()
        val_size = val_loss_dataset.__len__()

        log_msg = '\nTrain: {} \nValidation: {}\n\n'.format(train_size, val_size)

        # Validation set size
        val_used_size = min(val_size, args.val_size)
        log_msg += '** Validation Metrics are computed using {} samples. See --val_size\n'.format(val_used_size)

        # Model
        vocab_size = train_dataset.get_tokenizer_len()

        model = GPT2(args.model, vocab_size)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr)
        optimizer.zero_grad()

        scaler = GradScaler(enabled=args.use_amp)

        # Step & Epoch
        start_epoch = 1
        curr_step = 1
        best_val_acc = -math.inf

        # Load model checkpoint file (if specified)
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')

            model.load_weights(checkpoint)

            # Load training info
            curr_step = checkpoint['curr_step']
            start_epoch = checkpoint['epoch']
            prev_loss = checkpoint['loss']
            best_val_acc = checkpoint['val_acc']

            log_msg += 'Resuming Training...\n'
            log_msg += 'Model successfully loaded from {}\n'.format(args.ckpt)
            log_msg += 'Training loss: {:2f} (from ckpt)\n'.format(prev_loss)

        # DataParallel (GPU)
        # model = nn.DataParallel(model, device_ids)
        model.to(device)

        # Mode
        model.train()

        # Log
        print_log(log_msg, log_file)

        # Train
        steps_per_epoch = len(train_loader)
        start_time = time()

        for epoch in range(start_epoch, n_epochs+1):
            for batch in train_loader:
                # Load batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast(args.use_amp):
                    # Forward Pass
                    loss = model(batch)

                # Backward Pass
                scaler.scale(loss).backward()

                # Update Weights
                scaler.step(optimizer)
                scaler.update()

                # Clear
                optimizer.zero_grad()

                # Print Results - Loss
                if curr_step % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if val_loss_dataset:
                        #metrics = compute_eval_metrics(model, val_metrics_loader, device, val_used_size)
                        metrics = {}
                        metrics['loss'] = compute_eval_loss(model, val_loss_loader, device, val_used_size)
                        metrics['accuracy'] = -metrics['loss']

                        # Reset the mode to training
                        model.train()
                        log_msg = 'Validation Loss: {:.4f} || Accuracy: {:.4f}'.format(
                                    metrics['loss'], metrics['accuracy'])

                        print_log(log_msg, log_file)

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Loss', metrics['loss'], curr_step)
                        writer.add_scalar('Val/Accuracy', metrics['accuracy'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                            epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    print_log(log_msg, log_file)

                #Save the model
                if curr_step % args.save_interval == 0:
                    path = osj(log_dir, 'model_' + str(curr_step) + '.pth')

                    state_dict = {'model_state_dict': model.state_dict(), 'val_acc': best_val_acc,
                                  'curr_step': curr_step, 'loss': loss.item(), 'epoch': epoch}

                    if args.save:
                        torch.save(state_dict, path)

                    log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                    print_log(log_msg, log_file)

                curr_step += 1

            # Validation accuracy on the entire set
            if val_loss_dataset:
                log_msg = '-------------------------------------------------------------------------\n'
                #metrics = compute_eval_metrics(model, val_metrics_loader, device, val_size)
                metrics = {}
                metrics['loss'] = compute_eval_loss(model, val_loss_loader, device, val_size)
                metrics['accuracy'] = -metrics['loss']

                log_msg += '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Loss: {:.4f} || Accuracy: {:.4f}\n'.format(metrics['loss'], metrics['accuracy'])

                # Save model after every epoch, if improved
                if metrics['accuracy'] > best_val_acc:
                    best_val_acc = metrics['accuracy']

                    step = '{:.1f}k'.format(curr_step/1000) if curr_step > 1000 else f'{curr_step}'
                    filename = 'ep_{}_stp_{}_acc_{:.2f}_model.pth'.format(epoch, step, best_val_acc*100)

                    path = osj(log_dir, filename)

                    state_dict = {'model_state_dict': model.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_acc': best_val_acc}

                    if args.save:
                        torch.save(state_dict, path)

                    log_msg += "\n** Best Performing Model: {:.4f} ** \nSaving weights at {}\n".format(best_val_acc, path)

                log_msg += '-------------------------------------------------------------------------\n\n'

                print_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

        writer.close()
        log_file.close()

    elif args.mode == 'test':
        pass
        '''
        # Dataloader
        dataset = VisCOMETEvaluationDataset(args.data_dir, 'val', **data_params)

        # Dataloader
        loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

        vocab_size = dataset.get_tokenizer_len()

        # Model
        model = GPT2(vocab_size, args.model) if not args.text_only else Bert(args.model)
        model.eval()

        # Load model weights
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_weights(checkpoint)
        model.to(device)

        data_len = len(dataset)
        print('Total Samples: {}'.format(data_len))

        # Inference
        metrics = compute_eval_metrics(model, loader, device, data_len)

        # Results filename given, save to disk
        if args.csv_out:
            df = pd.DataFrame(metrics['meta'])
            df.to_csv(args.csv_out)

        print('Test Sentence Accuracy: {:.4f}'.format(metrics['accuracy']))
        '''


if __name__ == '__main__':
    main()
