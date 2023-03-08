import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

'''
To train your own target detection model, you must pay attention to the following points:
1、Check out whether your format meets the requirements before training. The library requires that the data set format is VOC format.
   Enter the picture as a .jpg picture. Without fixing the size, Resize will be automatically performed before the training.
   The gray diagram will be automatically converted into RGB pictures for training, without having to modify it by itself.
   Enter the picture If the suffix is not JPG, you need to switch to JPG in batches before starting training.

   The label is .xml format, and there will be target information that requires detection in the file. The label file and enter the picture file will be corresponding.

2、The size of the loss value is used to determine whether to converge. The more important thing is that there is a trend of convergence, 
   that is, the verification set loss is constantly declining. If the verification set loss is basically not changed, the model basically converges.
   The specific size of the loss value does not make much sense. The small and small is only the calculation method of loss, not close to 0. 
   If you want to make the loss look better, you can directly go to the corresponding loss function to remove 10000.
   The loss value during the training process will be stored in the loss_%y_%m_%d_%h_%m_%s file folder under the LOGS folder

3、The training right value file is stored in the LOGS folder. Each training generation (EPOCH) contains several training steps (STEP), and each training step length (STEP) has a gradient decrease.
   If you just train a few STEPs, it will not be saved, the concepts of EPOCH and STEP should be clear.
'''
if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    Whether to use CUDA
    #   No GPU can be set to false
    # ---------------------------------#
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    # ---------------------------------------------------------------------#
    #   classes_path    Point to the txt under the Model_data, and related to the data set trained by yourself
    #                   Before training, you must modify the CLASSSES_PATH to make it corresponding to your data set
    # ---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    # ---------------------------------------------------------------------#
    #   When model_path = '' is not loaded, the entire model is not loaded.
    #
    #   The weight of the entire model is used here, so loading is loaded in Train.py.
    #   If you want to let the model train from 0, set the model_path = '', the following freeze_train = Fasle,
    #   at this time the training starts from 0, and there is no process of freezing the main trunk.
    #
    #   Generally speaking, the training effect of the network from 0 will be poor, because the value of power is too random and the feature extraction effect is not obvious. Therefore,
    #   it is very, very, very not recommended to start training from 0!
    #   There are two schemes starting from 0:
    #   1、Thanks to the Data Augementation data enhancement method, the powerful data enhancement capabilities, the larger (300 and above), large BATCH (16 and above),
    #   more data (more than 10,000) of the Unfreeze_epoch, more data, more data, and more data (more than 10,000 or more).
    #   You can set MOSAIC = TRUE, directly randomly initialize parameters to start training, but the effect is still not as pre -training as pre -training. (Big data sets like Coco can do this)
    #   2、Understand the ImageNET dataset, first train the classification model, obtain the network's main part of the weight, the main part of the classification model is universal, and the model is generally trained based on this.
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'logs/best_palm/best_epoch_weights.pth'   #File folder stored in the model
    # ------------------------------------------------------#
    #   input_shape     The size of the input shape must be the multiple of 32
    # ------------------------------------------------------#
    input_shape     = [640, 640]
    phi             = 's'
    mosaic              = True   #Data Augmentation
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    # ------------------------------------------------------#
    #   Selection of training schemes, different optimizers:
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0.
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0.
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4.
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4.
    #       Among them: Unfreeze_epoch can be adjusted between 100-300.
    #   （二）Training from 0:
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False
    #       Among them: Unfreeze_epoch should be as small as 300 as possible. Optimizer_type = 'sgd', init_lr = 1E-2, mosaic = true.
    #   （三）Batch_size settings：
    #       In the range that the graphics card can accept, it is great. Insufficient video memory has nothing to do with the size of the dataset.
    #       It is prompted to adjust the small Batch_size for insufficient memory memory (OOM or CUDA OUT of Memory).
    #       Affected by the Batchnorm layer, Batch_size is the smallest to 2 and cannot be 1.
    #       Under normal circumstances, freeze_batch_size is recommended to be 1-2 times of UNFREEZE_BATCH_SIZE.
    #       It is not recommended to set the gap too large because it is related to the automatic adjustment of learning rate.
    # ----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch          = 0      #During the training interruption, redefine the initial turn round
    Freeze_Epoch        = 100
    Freeze_batch_size   = 16
    UnFreeze_Epoch      = 618   #Whether to use migration learning to freeze weight training
    Unfreeze_batch_size = 10
    Freeze_Train        = True
    # ------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decreased related
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate
    #   Min_lr          The minimum learning rate of the model is 0.01 that defaults to the maximum learning rate
    # ------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 10
    save_dir            = 'logs'

    eval_flag           = True
    eval_period         = 10

    num_workers         = 2
    # ------------------------------------------------------------------#
    #   Get the picture path and label
    # ------------------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    class_names, num_classes = get_classes(classes_path)
    # ------------------------------------------------------#
    #   Create YOLO model
    # ------------------------------------------------------#
    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        # ------------------------------------------------------#
        #   Load according to the key of the pre -training weight and the model of the model
        # ------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    # ----------------------#
    #   Get the loss function
    # ----------------------#
    yolo_loss    = YOLOLoss(num_classes, fp16)
    # ----------------------#
    #   Record LOSS
    # ----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small and cannot be trained. Please expand the data set.')   #The dataset is too small and cannot be trained. Please expand the data set.
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small and cannot be trained. Please expand the data set.")   #The dataset is too small and cannot be trained. Please expand the data set.
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small and cannot be trained. Please expand the data set")  #The dataset is too small and cannot be trained. Please expand the data set.

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                if ema:
                    ema.updates     = epoch_step * epoch
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
