import time
import os
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import utils


def get_dataset_statics(num_data,
                        batchsize,
                        data_loader):
    '''
    for extracted feature.
    '''
    mean_list = []
    std_list = []
    data_max, data_min = 0, 0
    for data, target in tqdm(data_loader):
        data = data.to(torch.float)
        mean = data.to(torch.float).mean(dim=-1).sum(0)
        std = data.to(torch.float).std(dim=-1).sum(0)
        max = data.max()
        min = data.min()
        mean_list.append(mean)
        std_list.append(std)
        data_max = max if max > data_max else data_max
        data_min = min if min < data_min else data_min
    mean = sum(mean_list) / num_data
    std = sum(std_list) / num_data

    print(f'mean: {mean}, std: {std}, max: {data_max}, min: {data_min}')


def train_end_to_end(device_output,
                     num_data,
                     num_epoch,
                     batchsize,
                     train_loader,
                     model,
                     optimizer,
                     scheduler,
                     criterion,
                     device_tested_number,
                     num_pulse,
                     save_dir_name):

    start_time = time.time()

    # train
    acc_list = []
    loss_list = []
    log_list = []
    for epoch in range(num_epoch):

        acc = []
        loss = 0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            this_batch_size = len(data)

            # reservoir output
            oect_output = utils.batch_rc_feat_extract(data,
                                                    device_output,
                                                    device_tested_number,
                                                    num_pulse,
                                                    this_batch_size)

            # readout layer
            logic = F.softmax(model(oect_output), dim=-1)

            batch_loss = criterion(logic, target)
            loss += batch_loss
            batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
            acc.append(batch_acc)
            batch_loss.backward()
            optimizer.step()

            # if i_batch % 300 == 0:
            #     print('%d data trained' % i_batch)
        scheduler.step()
        acc_epoch = (sum(acc) * batchsize / num_data).numpy()
        acc_list.append(acc_epoch)
        loss_list.append(loss)

        epoch_end_time = time.time()
        if epoch == 0:
            epoch_time = epoch_end_time - start_time
        else:
            epoch_time = epoch_end_time - epoch_start_time
        epoch_start_time = epoch_end_time

        # log info
        log = "epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time)
        print(log)
        log_list.append(log + '\n')
    utils.write_log(save_dir_name, log_list)

    # save readout layer
    torch.save(model, os.path.join(save_dir_name, 'downsampled_img_mode.pt'))


def train_with_feature(num_data,
                       num_epoch,
                       batchsize,
                       train_loader,
                       model,
                       optimizer,
                       scheduler,
                       criterion,
                       save_dir_name):

    start_time = time.time()
    acc_list = []
    loss_list = []
    log_list = []
    for epoch in range(num_epoch):

        acc = []
        loss = 0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            this_batch_size = len(data)

            data = data.to(torch.float).squeeze()
            # readout layer
            logic = F.softmax(model(data), dim=-1)
            # logic = model(data)

            batch_loss = criterion(logic, target)
            loss += batch_loss
            batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
            acc.append(batch_acc)
            batch_loss.backward()
            optimizer.step()

            # if i_batch % 300 == 0:
            #     print('%d data trained' % i_batch)
        scheduler.step()
        acc_epoch = (sum(acc) * batchsize / num_data).numpy()
        acc_list.append(acc_epoch)
        loss_list.append(loss)

        epoch_end_time = time.time()
        if epoch == 0:
            epoch_time = epoch_end_time - start_time
        else:
            epoch_time = epoch_end_time - epoch_start_time
        epoch_start_time = epoch_end_time

        # log info
        log = "epoch: %d, loss: %.2f, acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, epoch_time)
        print(log)
        log_list.append(log + '\n')
    utils.write_log(save_dir_name, log_list)

    # save readout layer
    torch.save(model, os.path.join(save_dir_name, 'downsampled_img_mode.pt'))


def test(tran_type,
         device_output,
         device_tested_number,
         num_data,
         num_pulse,
         num_class,
         batchsize,
         test_loader,
         model,
         save_dir_name,
         img_save=False):
    # test
    te_accs = []
    te_outputs = []
    targets = []
    # ims_dir = os.path.join(save_dir_name, 'demo_images')
    # img_for_demo = utils.ImagesForDemo(path=ims_dir)
    with torch.no_grad():
        for i, (data, img, target) in enumerate(test_loader):

            this_batch_size = len(data)
            if tran_type == 'train_ete':
                data =utils.batch_rc_feat_extract(data,
                                                  device_output,
                                                  device_tested_number,
                                                  num_pulse,
                                                  this_batch_size)
            data = data.to(torch.float)
            output = F.softmax(model(data.squeeze()), dim=-1)
            # output = model(data)
            te_outputs.append(output)
            acc = torch.sum(output.argmax(dim=-1) == target) / this_batch_size
            te_accs.append(acc)
            targets.append(target)
            # img_for_demo.update_images(data, img, target, output)
        te_acc = (sum(te_accs) * batchsize / num_data).numpy()

        # log infos
        log = "test acc: %.6f" % te_acc + '\n'
        print(log)
        utils.write_log(save_dir_name, log)
        # img_for_demo.save_images()

        # te_outputs = torch.cat(te_outputs, dim=0)
        te_outputs = torch.stack(te_outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        # targets = torch.stack(targets, dim=0)

        # confusion matrix
        conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=-1))

        conf_mat_dataframe = pd.DataFrame(conf_mat,
                                        index=list(range(num_class)),
                                        columns=list(range(num_class)))

        conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_mat_dataframe, annot=True, fmt='d')
        plt.savefig(os.path.join(save_dir_name, f'conf_mat_{te_acc * 1e4:.0f}'))
        plt.close()
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_mat_normalized, annot=True)
        plt.savefig(os.path.join(save_dir_name, f'conf_mat_normalized_{te_acc * 1e4:.0f}'))
        plt.close()
        print('confusion matrix saved')


def save_rc_feature(train_loader,
                    test_loader,
                    num_pulse,
                    device_output,
                    device_tested_number,
                    filename):
    device_features = []
    tr_targets = []

    for i, (data, target) in enumerate(train_loader):
        data = data.squeeze()
        oect_output = utils.rc_feature_extraction(data, device_output, device_tested_number, num_pulse)
        device_features.append(oect_output)
        tr_targets.append(target)
    tr_features = torch.stack(device_features, dim=0)
    tr_targets = torch.stack(tr_targets).squeeze()

    te_oect_outputs = []
    te_targets = []
    for i, (data, im, target) in enumerate(test_loader):
        data = data.squeeze()
        oect_output = utils.rc_feature_extraction(data, device_output, device_tested_number, num_pulse)
        te_oect_outputs.append(oect_output)
        te_targets.append(target)
    te_features = torch.stack(te_oect_outputs, dim=0)
    te_targets = torch.stack(te_targets).squeeze()

    tr_filename = filename + f'_tr.pt'
    te_filename = filename + f'_te.pt'
    torch.save((tr_features, tr_targets), tr_filename)
    torch.save((te_features, te_targets), te_filename)
