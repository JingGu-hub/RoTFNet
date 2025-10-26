import os
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as data
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.io.arff import loadarff
from scipy import stats
import torch.nn.functional as F
from math import inf
import datetime

from statsmodels.tsa.stattools import acf as sm_acf

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

def build_dataset_uea(args):
    data_path = args.data_dir
    train_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TRAIN.arff'))[0]
    test_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TEST.arff'))[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    train_dataset = train_X.transpose(0, 2, 1)
    train_target = train_y
    test_dataset = test_X.transpose(0, 2, 1)
    test_target = test_y

    num_classes = len(np.unique(train_target))
    train_target = transfer_labels(train_target)
    test_target = transfer_labels(test_target)

    ind = np.where(np.isnan(train_dataset))
    col_mean = np.nanmean(train_dataset, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_dataset[ind] = np.take(col_mean, ind[1])

    ind_test = np.where(np.isnan(test_dataset))
    test_dataset[ind_test] = np.take(col_mean, ind_test[1])

    train_dataset, train_target = shuffler_dataset(train_dataset, train_target)
    test_dataset, test_target = shuffler_dataset(test_dataset, test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def build_dataset_pt(args):
    data_path = os.path.join(args.data_dir, args.dataset)
    train_dataset_dict = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = train_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    train_target = train_dataset_dict["labels"].numpy()
    num_classes = len(np.unique(train_dataset_dict["labels"].numpy(), return_counts=True)[0])
    train_target = transfer_labels(train_target)

    test_dataset_dict = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = test_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    test_target = test_dataset_dict["labels"].numpy()
    test_target = transfer_labels(test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def shuffler_dataset(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


def flip_label(dataset, target, ratio, args=None):
    """
    Induce label noise by randomly corrupting labels
    :param target: list or array of labels
    :param ratio: float: noise ratio
    :param pattern: flag to choose which type of noise.
            0 or mod(pattern, #classes) == 0 = symmetric
            int = asymmetric
            -1 = flip
    :return:
    """
    assert 0 <= ratio < 1

    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    if args.noise_type == 'instance':
        # Instance
        num_classes = len(np.unique(target, return_counts=True)[0])
        data = torch.from_numpy(dataset).type(torch.FloatTensor)
        targets = torch.from_numpy(target).type(torch.FloatTensor).to(torch.int64)
        dataset_ = zip(data, targets)
        feature_size = dataset.shape[1] * dataset.shape[2]
        label = get_instance_noisy_label(n=ratio, dataset=dataset_, labels=targets, num_classes=num_classes,
                                         feature_size=feature_size, seed=args.random_seed if args is not None else 42)
    else:
        for i in range(label.shape[0]):
            # symmetric noise
            if args.noise_type == 'symmetric':
                p1 = ratio / (n_class - 1) * np.ones(n_class)
                p1[label[i]] = 1 - ratio
                label[i] = np.random.choice(n_class, p=p1)
            elif args.noise_type == 'pairflip':
                # pairflip
                label[i] = np.random.choice([label[i], (target[i] + 1) % n_class], p=[1 - ratio, ratio])

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])

    return label, mask

def new_length(seq_length, sample_rate):
    last_one = 0
    if seq_length % sample_rate > 0:
        last_one = 1
    new_length = int(np.floor(seq_length / sample_rate)) + last_one
    return new_length

def downsample_torch(x_data, sample_rate):
    """
     Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
     rate k. hence, every k-th element of the original time series is kept.
    """
    last_one = 0
    if x_data.shape[2] % sample_rate > 0:
        last_one = 1

    new_length = int(np.floor(x_data.shape[2] / sample_rate)) + last_one
    output = torch.zeros((x_data.shape[0], x_data.shape[1], new_length)).cuda()
    output[:, :, range(new_length)] = x_data[:, :, [i * sample_rate for i in range(new_length)]]

    return output

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=42):
    # n -> noise_rate
    # dataset
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size
    # norm_std -> default 0.1
    # seed -> random_seed

    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        t = W[y]
        A = x.reshape(1, -1).mm(W[y]).squeeze(0)

        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        #
    print('****************************************')
    print('following is flip percentage:')

    for i in range(label_num):
        sum_i = sum(record[i])
        for j in range(label_num):
            if i != j:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
            else:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
        print()

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            for j in range(label_num):
                print(f"{a[j]:.2f}", end="\t")
            print()
            cnt += 1
        if cnt >= 10:
            break
    print(P)
    return np.array(new_label)


def get_clean_loss_tensor_mask(loss_all, remember_rate):
    '''
    :param loss: numpy
    :param remember_rate: float, 1 - noise_rate
    :return: mask_loss, 1 is clean, 0 is noise
    '''

    ind_1_sorted =  torch.argsort(loss_all)
    mask_loss = torch.zeros(len(ind_1_sorted)).cuda()
    for i in range(int(len(ind_1_sorted) * remember_rate)):
        mask_loss[ind_1_sorted[i]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return mask_loss

def get_clean_mask(loss_all, inds, clean_inds, remember_rate):
    '''
    :param loss: numpy
    :param remember_rate: float, 1 - noise_rate
    :return: mask_loss, 1 is clean, 0 is noise
    '''

    ind_1_sorted =  torch.argsort(loss_all)
    mask_loss = torch.zeros(len(ind_1_sorted)).cuda()
    for i in range(len(ind_1_sorted)):
        if i < int(len(ind_1_sorted) * remember_rate):
            mask_loss[ind_1_sorted[i]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)
        else:
            if inds[ind_1_sorted[i]] in clean_inds:
                mask_loss[ind_1_sorted[i]] = 1

    return mask_loss

def get_accuracy(classifier_output1, classifier_output2, classifier_output3, y):
    target_pred1 = torch.argmax(classifier_output1.data, axis=1)
    target_pred2 = torch.argmax(classifier_output2.data, axis=1)
    target_pred3 = torch.argmax(classifier_output3.data, axis=1)

    target_pred1 = target_pred1.unsqueeze(1)
    target_pred2 = target_pred2.unsqueeze(1)
    target_pred3 = target_pred3.unsqueeze(1)

    final_target_pred = torch.cat((target_pred1, target_pred2, target_pred3), 1)
    target_pred_temp = torch.mode(final_target_pred, axis=1).values

    return target_pred_temp.eq(y).sum().item()

def count_class_num(train_target, num_classes):
    class_vector = np.zeros(num_classes)
    for i in range(num_classes):
        class_vector[i] = np.sum(train_target == i)
    class_vector = [int(item) for item in class_vector]

    return class_vector

def make_dir(base_path, paths=[], is_clinodiagonal=False):
    for path in paths:
        if not os.path.exists(os.path.join(base_path, path)):
            os.makedirs(os.path.join(base_path, path))
        base_path = os.path.join(base_path, path)

    return base_path if is_clinodiagonal==False else base_path + '/'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line=None, exist_create_flag=True):
    create_dir(path)
    filename = os.path.join(path, filename)

    if filename != None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            with open(filename, "a") as myfile:
                print("create new file: %s" % filename)
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')
        elif exist_create_flag:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')

    return filename

def count_refurb_matrix(classifier_output1, classifier_output2, classifier_output3, refurb_matrixs, refurb_len, inds, epoch):
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

    train_target_prob1, target_pred1 = classifier_output1.max(1)
    train_target_prob2, target_pred2 = classifier_output2.max(1)
    train_target_prob3, target_pred3 = classifier_output3.max(1)

    refurb_matrix1[inds, epoch % refurb_len] = target_pred1.cpu().numpy()
    refurb_matrix2[inds, epoch % refurb_len] = target_pred2.cpu().numpy()
    refurb_matrix3[inds, epoch % refurb_len] = target_pred3.cpu().numpy()

    return refurb_matrix1, refurb_matrix2, refurb_matrix3

def refurb_label(train_loader, refurb_matrixs, unselected_inds, update_inds):
    refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]

    train_target_pred_mode1 = torch.mode(torch.from_numpy(refurb_matrix1).cuda(), axis=1).values.unsqueeze(1)
    train_target_pred_mode2 = torch.mode(torch.from_numpy(refurb_matrix2).cuda(), axis=1).values.unsqueeze(1)
    train_target_pred_mode3 = torch.mode(torch.from_numpy(refurb_matrix3).cuda(), axis=1).values.unsqueeze(1)

    pred_label = torch.cat((train_target_pred_mode1, train_target_pred_mode2, train_target_pred_mode3), dim=1)
    refurb_y = torch.mode(pred_label, axis=1).values.to(dtype=torch.long)

    train_target = train_loader.dataset.target.cuda()
    unselected_inds = set(unselected_inds)
    for batch_x, batch_y, indexes in train_loader:
        indexes = indexes.cuda()
        pred_labels_batch = pred_label[indexes]  # shape: (batch_size, 3)

        # 统计多数投票（每行中出现次数最多的标签的数量）
        pred_labels_batch = pred_labels_batch.to(dtype=torch.long)
        max_label = int(pred_labels_batch.max().item()) + 1
        mode_counts = torch.stack([
            torch.bincount(row, minlength=max_label).max()
            for row in pred_labels_batch
        ])

        # 构建掩码条件
        all_agree = mode_counts == pred_labels_batch.shape[1]  # 所有模型一致
        labels_differ = train_target[indexes] != refurb_y[indexes]
        in_unselected = torch.tensor([i.item() in unselected_inds for i in indexes]).cuda()

        mask = all_agree & labels_differ & in_unselected
        selected_indexes = indexes[mask]

        # 更新标签和未选中的索引集合
        train_target[selected_indexes] = refurb_y[selected_indexes]
        unselected_inds.difference_update(selected_indexes.tolist())
        update_inds.extend(selected_indexes.tolist())

    return train_target.detach().cpu().numpy(), list(unselected_inds), update_inds

def fft(data):
    N = data.shape[2]  # sequence length
    T = 1.0 / N  # sampling interval

    # average batch_size and num_channels dim
    averaged_data = data.mean(axis=0).mean(axis=0)
    window = np.hanning(data.shape[2])
    averaged_data = averaged_data * window

    # compute FFT
    yf = np.fft.fft(averaged_data)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    power_spectrum = 2.0 / N * np.abs(yf[:N // 2])
    # power_spectrum = np.abs(yf[:N//2])

    return xf, power_spectrum

def fft_main_periods_wo_duplicates(data, k=5, dataset_name='Dataset'):
    xf, power_spectrum = fft(data)
    N = data.shape[1]
    averaged_data = data.mean(axis=0).mean(axis=-1)

    # filter zero frequency and frequency equals to 1
    valid_indices = (xf > 0) & (xf != 1)
    xf = xf[valid_indices]
    power_spectrum = power_spectrum[valid_indices]

    # top amplitudes and frequencies
    indices = np.argsort(power_spectrum)[::-1]  # rank from high to low
    main_frequencies = xf[indices]
    main_amplitudes = power_spectrum[indices]

    unique_periods = []
    unique_amplitudes = []
    unique_frequencies = []
    used_periods = set()

    i = 0
    while len(unique_periods) < k and i < len(main_frequencies):
        period = np.round(1 / main_frequencies[i] * N).astype(int)
        if period not in used_periods:
            unique_periods.append(period)
            unique_amplitudes.append(main_amplitudes[i])
            unique_frequencies.append(main_frequencies[i])
            used_periods.add(period)
        i += 1

    # Print main periods and amplitudes
    print("Main periods and amplitudes: ")
    for i, (period, freq, amp) in enumerate(zip(unique_periods, unique_frequencies, unique_amplitudes)):
        print(f"period {i + 1}: {period}, amplitude: {int(amp)}")

    # # Plot time series and power spectrum
    time = np.arange(N)
    plt.figure(figsize=(14, 20))

    plt.subplot(2, 1, 1)
    plt.plot(time, averaged_data)
    plt.title(dataset_name + ' Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    plt.plot(xf, power_spectrum)
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.scatter(unique_frequencies, unique_amplitudes, color='red', zorder=5)  # mark main frequency points

    for i, (freq, amp) in enumerate(zip(unique_frequencies, unique_amplitudes)):
        plt.annotate(f'{int(freq)} Hz\n{int(amp)}', xy=(freq, amp), xytext=(freq, amp + 0.02),
                     textcoords='data', ha='center', color='red')
    plt.subplots_adjust(hspace=0.4)
    # plt.tight_layout()
    plt.show()
    return unique_periods


def detect_period(time_series, method='fft', axis=2, max_lag=None):
    ts = np.asarray(time_series)
    n = ts.shape[axis]

    if method.lower() == 'fft':
        # 去均值
        mean_along = np.expand_dims(ts.mean(axis=axis), axis=axis)
        ts_centered = ts - mean_along

        # FFT（numpy.fft 支持 axis）
        freq_spectrum = np.fft.fft(ts_centered, axis=axis)
        power = np.abs(freq_spectrum)**2

        # 频率坐标
        freqs = np.fft.fftfreq(n, d=1.0)

        # 只保留正频率
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = np.take(power, indices=np.where(pos_mask)[0], axis=axis)

        if freqs_pos.size == 0 or np.all(power_pos == 0):
            return 0

        # 在除时间轴以外的维度上取平均，得到一维功率谱
        other_axes = tuple(i for i in range(power_pos.ndim) if i != axis)
        mean_power = power_pos
        for ax in sorted(other_axes, reverse=True):
            mean_power = mean_power.mean(axis=ax)

        # 找主频，计算周期，并四舍五入取整
        dominant_index = int(np.argmax(mean_power))
        dominant_freq = freqs_pos[dominant_index]
        period = 1.0 / dominant_freq if dominant_freq > 0 else 0
        estimated_period = int(round(period))

    elif method.lower() == 'acf':
        # 平均到 1D
        other_axes = tuple(i for i in range(ts.ndim) if i != axis)
        ts_1d = ts.mean(axis=other_axes)

        if max_lag is None:
            max_lag = n // 2

        acf_vals = sm_acf(ts_1d, nlags=max_lag, fft=True)
        peaks = np.diff(np.sign(np.diff(acf_vals))) < 0
        peak_idxs = np.where(peaks)[0] + 1

        estimated_period = int(peak_idxs[0]) if peak_idxs.size > 0 else 0

    return estimated_period

def instance_detect_period(datasets):
    time_step_len = datasets.shape[2]

    periods = np.zeros(datasets.shape[0])
    for i, time_series in enumerate(datasets):
        instance_period = detect_period(np.expand_dims(time_series, axis=0), method='fft')
        if instance_period <= 1 or instance_period >= time_step_len:
            instance_period = detect_period(np.expand_dims(time_series, axis=0), method='acf')

        periods[i] = instance_period if instance_period > 1 and instance_period < time_step_len else time_step_len
    return periods

def plot_curve(y, x=None, cp_list=None, title="Array Curve", xlabel="Index", ylabel="Value", color='b'):
    if x is None:
        x = range(len(y))

    plt.plot(x, y, color=color, marker='o')

    # 画竖直线
    if cp_list:
        for cp in cp_list:
            plt.axvline(x=cp, color='r', linestyle='--', linewidth=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

#绘制变化点图像
def draw_curve(data1, data2, cp_list=None):
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            plot_curve(data1[i, j, :], cp_list=cp_list)
            plot_curve(data2[i, j, :], cp_list=cp_list)
