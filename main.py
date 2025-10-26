import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

import warnings

from models.model import NoiScaleFormer
from utils.data_utils import get_dataset, load_loader
from utils.loss import delay_loss, freq_aug_loss
from utils.utils import set_seed, get_accuracy, create_dir, create_file, refurb_label, count_refurb_matrix, draw_curve

from utils.constants import Multivariate2018_arff_DATASET as UEA_DATASET
from utils.constants import Four_dataset as OTHER_DATASET

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# Base setup
parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')
parser.add_argument('--data_dir', type=str, default='../data/Multivariate2018_arff/Multivariate_arff', help='dataset directory')
parser.add_argument('--model_save_dir', type=str, default='./outputs/model_save/', help='model save directory')
parser.add_argument('--result_save_dir', type=str, default='./outputs/result_save/', help='result save directory')
parser.add_argument('--model_names', type=list, default=['encoder1.pth', 'encoder2.pth', 'encoder3.pth'], help='model names')

# Dataset setup
parser.add_argument('--archive', type=str, default='UEA', help='UEA, other')
parser.add_argument('--dataset', type=str, default='JapaneseVowels', help='dataset name')  # [all dataset in Multivariate_arff]

# Label noise
parser.add_argument('--noise_type', type=str, default='symmetric', help='symmetric, instance, pairflip')
parser.add_argument('--label_noise_rate', type=float, default=0.2, help='label noise ratio, sym: 0.2, 0.5, asym: 0.4, ins: 0.4')
parser.add_argument('--scale_list', type=list, default=[1, 2, 4], help='')
parser.add_argument('--periods', type=list, default=[1], help='')
parser.add_argument('--period', type=int, default=0, help='')

# training setup
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--epoch', type=int, default=50, help='training epoch')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# model setup
parser.add_argument('--embed_size', type=int, default=73, help='model hyperparameters')
parser.add_argument('--feature_size', type=int, default=64, help='model output dimension')
parser.add_argument('--num_layers', type=int, default=1, help='number of  layers')
parser.add_argument('--num_heads', type=int, default=4, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='MCR hyperparameters')

parser.add_argument('--input_channel', type=int, default=1, help='')
parser.add_argument('--seq_len', type=int, default=1, help='')
parser.add_argument('--num_classes', type=int, default=1, help='')

# feature augmentation and MCR setup
parser.add_argument('--alpha', type=float, default=-0.01, help='MCR hyperparameters')
parser.add_argument('--beta', type=float, default=0.01, help='feature augmentation hyperparameters')
parser.add_argument('--gamma', type=float, default=0.5, help='feature augmentation hyperparameters')

# delay loss and refurb setup
parser.add_argument('--start_mask_epoch', type=int, default=10, help='sample mask epoch')
parser.add_argument('--start_delay_loss', type=int, default=15, help='start delayed loss epoch')
parser.add_argument('--delay_loss_k', type=int, default=3, help='the length of delay loss')
parser.add_argument('--start_refurb', type=int, default=30, help='start refurb epoch')
parser.add_argument('--refurb_len', type=int, default=5, help='the length of refurb')

# GPU setup
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

def pretrain(args, train_loader, models, model_path, pretrain_optimizer):
    # train encoder
    model1, model2, model3 = models[0], models[1], models[2]

    mse_criterion = nn.MSELoss('mean').cuda()
    best_loss = float('inf')
    for epoch in range(args.epoch):
        model1.train()
        model2.train()
        model3.train()

        epoch_train_loss = 0
        for i, (x, y, indexes) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            down_x1, reconst_x1, recons_aug_list1, freq_aug_list1 = model1(x, task='pretrain', epoch=epoch)
            down_x2, reconst_x2, recons_aug_list2, freq_aug_list2 = model2(x, task='pretrain', epoch=epoch)
            down_x3, reconst_x3, recons_aug_list3, freq_aug_list3 = model3(x, task='pretrain', epoch=epoch)

            loss1 = mse_criterion(down_x1, reconst_x1) + freq_aug_loss(mse_criterion, x, freq_aug_list1) + freq_aug_loss(mse_criterion, down_x1, recons_aug_list1)
            loss2 = mse_criterion(down_x2, reconst_x2) + freq_aug_loss(mse_criterion, x, freq_aug_list2) + freq_aug_loss(mse_criterion, down_x2, recons_aug_list2)
            loss3 = mse_criterion(down_x3, reconst_x3) + freq_aug_loss(mse_criterion, x, freq_aug_list3) + freq_aug_loss(mse_criterion, down_x3, recons_aug_list3)
            loss = (loss1 + loss2 + loss3) / 3

            # if epoch == 45:
            #     draw_curve(x.detach().cpu().numpy(), freq_aug_list1[0].detach().cpu().numpy())

            pretrain_optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward()
            pretrain_optimizer.step()

            epoch_train_loss += loss.item() * x.shape[0]

        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)

        print('Epoch:', epoch, 'train Loss:', epoch_train_loss)

        # save best model
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(model1.state_dict(), model_path + args.model_names[0])
            torch.save(model2.state_dict(), model_path + args.model_names[1])
            torch.save(model3.state_dict(), model_path + args.model_names[2])

def evaluate(args, epoch, test_loader, models, last_five_accs, last_five_losses):
    model1, model2, model3 = models[0], models[1], models[2]

    model1.eval()
    model2.eval()
    model3.eval()

    test_correct = 0
    epoch_test_loss = 0
    for i, (x, y, indexes) in enumerate(test_loader):
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()

            # foreward
            c_out1 = model1(x, task='test')
            c_out2 = model2(x, task='test')
            c_out3 = model3(x, task='test')

            loss1 = F.cross_entropy(c_out1, y, reduction='mean')
            loss2 = F.cross_entropy(c_out2, y, reduction='mean')
            loss3 = F.cross_entropy(c_out3, y, reduction='mean')

            loss = (loss1 + loss2 + loss3)
            epoch_test_loss += loss.item() * int(x.shape[0])
            test_correct += get_accuracy(c_out1, c_out2, c_out3, y)

    epoch_test_loss = epoch_test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    # compute last five accs and losses
    if (epoch + 5) >= args.epoch:
        last_five_accs.append(test_acc)
        last_five_losses.append(epoch_test_loss)

    return epoch_test_loss, test_acc, last_five_accs, last_five_losses

def train(epoch, train_loader, models, optimizer, refurb_matrixs,
          k_train_losses, unselected_inds, update_inds):
    model1, model2, model3 = models[0], models[1], models[2]

    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]
    k_train_loss1, k_train_loss2, k_train_loss3 = k_train_losses[0], k_train_losses[1], k_train_losses[2]

    model1.model_train()
    model2.model_train()
    model3.model_train()

    remember_rate = 1 if epoch < args.start_mask_epoch else 1 - args.label_noise_rate
    epoch_train_loss, epoch_train_correct = 0, 0
    loss_all = np.zeros(len(train_loader.dataset))

    for i, (x, y, indexes) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        # foreward
        c_out1 = model1(x, task='train')
        c_out2 = model2(x, task='train')
        c_out3 = model3(x, task='train')

        # compute refurb matrix
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = (
            count_refurb_matrix(c_out1, c_out2, c_out3, [refurb_matrix1, refurb_matrix2, refurb_matrix3], args.refurb_len, indexes, epoch))

        # compute delay loss
        loss1, loss2, loss3, k_train_loss1, k_train_loss2, k_train_loss3, loss_all = (
            delay_loss(args, [c_out1, c_out2, c_out3], y, [k_train_loss1, k_train_loss2, k_train_loss3], epoch, loss_all, indexes, update_inds))

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()

        loss = (loss1 + loss2 + loss3) / 3
        epoch_train_loss += loss.item() * int(x.shape[0] * remember_rate)
        epoch_train_correct += get_accuracy(c_out1, c_out2, c_out3, y)

    epoch_train_loss = epoch_train_loss / int(len(train_loader.dataset) * remember_rate)
    epoch_train_acc = epoch_train_correct / len(train_loader.dataset)

    # obtain unselected inds
    if epoch == args.start_mask_epoch:
        ind_1_sorted = torch.argsort(torch.from_numpy(loss_all).cuda(), descending=True)
        for i in range(int(len(ind_1_sorted) * args.label_noise_rate)):
            unselected_inds.append(ind_1_sorted[i].cpu().numpy().item())

    # refurb label
    if epoch >= args.start_refurb:
        train_target, unselected_inds, update_inds = refurb_label(train_loader, [refurb_matrix1, refurb_matrix2, refurb_matrix3], unselected_inds, update_inds)
        # reload train loader
        train_loader = load_loader(args, train_loader.dataset.dataset.detach().cpu().numpy(), train_target)

    return epoch_train_loss, epoch_train_acc, train_loader, [model1, model2, model3], [refurb_matrix1, refurb_matrix2, refurb_matrix3], [k_train_loss1, k_train_loss2, k_train_loss3], unselected_inds, update_inds


def main(archive='UEA', gpu_id=0, noise_type='symmetric', noise_rates=[0.5], low_freq_ratio_list=[0.3], seed=42):
    if archive == 'UEA':
        args.archive = archive
        datasets = UEA_DATASET
        args.data_dir = '../data/Multivariate2018_arff/Multivariate_arff'
    elif archive == 'other':
        args.archive = archive
        datasets = OTHER_DATASET
        args.data_dir = '../data/ts_noise_data'
    args.random_seed = seed

    torch.cuda.set_device(gpu_id)
    set_seed(args)

    low_freq_ratio_list = low_freq_ratio_list

    for dataset in datasets:
        args.dataset = dataset
        for noise_rate in noise_rates:
            args.noise_type = noise_type
            args.label_noise_rate = noise_rate

            for low_freq_ratio in low_freq_ratio_list:
                args.low_freq_ratio = low_freq_ratio

                # get dataset
                train_loader, test_loader, num_classes = get_dataset(args)
                args.input_channel, args.seq_len, args.num_classes = train_loader.dataset.dataset.shape[1], train_loader.dataset.dataset.shape[2], num_classes

                # create model
                model1 = NoiScaleFormer(args, args.scale_list[0]).cuda()
                model2 = NoiScaleFormer(args, args.scale_list[1]).cuda()
                model3 = NoiScaleFormer(args, args.scale_list[2]).cuda()

                # define delay loss
                k_train_loss1 = np.zeros((len(train_loader.dataset), args.delay_loss_k))
                k_train_loss2 = np.zeros((len(train_loader.dataset), args.delay_loss_k))
                k_train_loss3 = np.zeros((len(train_loader.dataset), args.delay_loss_k))

                # define refurb matrix
                refurb_matrix1, refurb_matrix2, refurb_matrix3 = (np.zeros((len(train_loader.dataset), args.refurb_len)),
                                                                  np.zeros((len(train_loader.dataset), args.refurb_len)),
                                                                  np.zeros((len(train_loader.dataset), args.refurb_len)))

                # define optimizer
                optimizer = torch.optim.Adam([{'params': model1.parameters()}, {'params': model2.parameters()}, {'params': model3.parameters()}], lr=args.lr)

                # define start epoch and best loss
                unselected_inds, update_inds = [], []
                last_five_accs, last_five_losses = [], []

                args.model_names = ['%s_%s%.2f_seed%d_lfr%.1f_scalefomer_1' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed, args.low_freq_ratio),
                                    '%s_%s%.2f_seed%d_lfr%.1f_scalefomer_2' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed, args.low_freq_ratio),
                                    '%s_%s%.2f_seed%d_lfr%.1f_scalefomer_3' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed, args.low_freq_ratio)]

                # pretrain model if model parameters do not exist
                model_path = args.model_save_dir + args.archive + '/' + str(args.dataset) + '/'
                create_dir(model_path)
                if not os.path.exists(args.model_save_dir + args.archive + '/' + str(args.dataset) + '/' + args.model_names[0]):
                    # if not exist, pretrain and save model
                    pretrain(args, train_loader, [model1, model2, model3], model_path, optimizer)

                # load pretrain model
                model1.load_state_dict(torch.load(model_path + args.model_names[0]))
                model2.load_state_dict(torch.load(model_path + args.model_names[1]))
                model3.load_state_dict(torch.load(model_path + args.model_names[2]))

                out_dir = args.result_save_dir + args.archive + '/' + str(args.dataset) + '/'
                out_file = '%s_%s%.2f_seed%d.txt' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed)
                out_file = create_file(out_dir, out_file, 'epoch,train loss,train acc,test loss,test acc')
                total_file = create_file(out_dir, 'total_result.txt', 'statement,test loss,test acc', exist_create_flag=False)
                for epoch in range(args.epoch):
                    # train
                    train_loss, train_acc, train_loader, models, refurb_matrixs, k_train_losses, unselected_inds, update_inds = (
                        train(epoch, train_loader, [model1, model2, model3], optimizer,
                              [refurb_matrix1, refurb_matrix2, refurb_matrix3], [k_train_loss1, k_train_loss2, k_train_loss3], unselected_inds, update_inds))

                    model1, model2, model3 = models[0], models[1], models[2]
                    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]
                    k_train_loss1, k_train_loss2, k_train_loss3 = k_train_losses[0], k_train_losses[1], k_train_losses[2]

                    # test
                    test_loss, test_acc, last_five_accs, last_five_losses = evaluate(args, epoch, test_loader, models, last_five_accs, last_five_losses)
                    print('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (
                           epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc))

                    with open(out_file, "a") as myfile:
                        myfile.write(str('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f'
                                         % (epoch + 1, args.epoch, train_loss, train_acc, test_loss, test_acc) + "\n"))

                test_accuracy = round(np.mean(last_five_accs), 4)
                test_loss = round(np.mean(last_five_losses), 4)

                print('Test Accuracy:', test_accuracy, 'Test Loss:', test_loss)
                with open(total_file, "a") as myfile:
                    myfile.write('%s_%s%.2f_seed%d_lfr%.1f, %f, %f\n' % (args.dataset, args.noise_type, args.label_noise_rate, args.random_seed, args.low_freq_ratio, test_loss, test_accuracy))

if __name__ == '__main__':
    # main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.2])
    # main(archive='UEA', gpu_id=3, noise_type='symmetric', noise_rates=[0.2])
    # main(archive='other', gpu_id=1, noise_type='symmetric', noise_rates=[0.5])
    # main(archive='UEA', gpu_id=1, noise_type='symmetric', noise_rates=[0.5])
    # main(archive='other', gpu_id=2, noise_type='instance', noise_rates=[0.4])
    # main(archive='UEA', gpu_id=2, noise_type='instance', noise_rates=[0.4])
    # main(archive='other', gpu_id=3, noise_type='pairflip', noise_rates=[0.4])
    # main(archive='UEA', gpu_id=3, noise_type='pairflip', noise_rates=[0.4])

    main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], low_freq_ratio_list=[0.5], seed=40)
    # main(archive='other', gpu_id=3, noise_type='symmetric', noise_rates=[0.5], low_freq_ratio_list=[0.1, 0.4, 0.8, 0.9], seed=256)
    # main(archive='other', gpu_id=2, noise_type='instance', noise_rates=[0.4], low_freq_ratio_list=[0.9], seed=32)
    # main(archive='other', gpu_id=1, noise_type='pairflip', noise_rates=[0.4], low_freq_ratio_list=[0.1, 0.2, 0.3], seed=256)
