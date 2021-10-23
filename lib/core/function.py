# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import islice

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from core.loss import FinetuneUnsupLoss, JointsMSELoss
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    loader_len = len(train_loader)
    if config.TRAIN.LIMIT_EPOCH_SIZE > 0:
        loader_len = config.TRAIN.LIMIT_EPOCH_SIZE
        train_loader = islice(train_loader, loader_len)
    for i, inputs in enumerate(train_loader):
        if len(inputs) == 5:
            input, target, target_weight, is_labeled, meta = inputs
        else:
            input, target, target_weight, meta = inputs
            is_labeled = None
        # measure data loading time
        data_time.update(time.time() - end)

        # baseline models
        if not config.MODEL.IS_IMM:
            # compute output
            outputs = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, (list, tuple)):
                loss = criterion(outputs[0], target, target_weight, is_labeled=is_labeled)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight, is_labeled=is_labeled)
            else:
                output = outputs
                loss = criterion(output, target, target_weight, is_labeled=is_labeled)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, loader_len, batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                prefix)
        
        # train for IMM joint training
        else:
            # compute output
            outputs = model(input)

            if len(outputs) == 5:
                pred_images, output_unsup, output, joints_x, joints_y = outputs
            else:
                pred_images, output_unsup, output = outputs
            ref_images = input[:, 0, :, :, :].cuda(non_blocking=True)
            input = input[:, 1, :, :, :].cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
            _, _, _, unsup_pred = accuracy(output_unsup.detach().cpu().numpy(),
                                            output_unsup.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            loss = criterion(output, target, target_weight, input, pred_images,
                            joints_pred=torch.from_numpy(pred).cuda(non_blocking=True),
                            is_labeled=is_labeled)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, loader_len, batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                debug_images = save_debug_images(config, input, meta, target, pred*4, output,
                                prefix, ref_images, pred_images, unsup_pred*4, output_unsup)
                for key in debug_images:
                    writer.add_image("Train/" + key, np.transpose(debug_images[key], (2, 0, 1))[[2,1,0]], global_steps)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            # compute output
            outputs = model(input)

            if not config.MODEL.IS_IMM:
                if isinstance(outputs, (list, tuple)):
                    output = outputs[-1]
                else:
                    output = outputs

                if config.TEST.FLIP_TEST:
                    input_flipped = input.flip(3)
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, (list, tuple)):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])

                idx += num_images

                if i % config.PRINT_FREQ == 0:
                    msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc)
                    logger.info(msg)

                    prefix = '{}_{}'.format(
                        os.path.join(output_dir, 'val'), i
                    )
                    save_debug_images(config, input, meta, target, pred*4, output,
                                    prefix)

            # validate for IMM joint training
            else:
                if len(outputs) == 5:
                    pred_images, output_unsup, output, joints_x, joints_y = outputs
                else:
                    pred_images, output_unsup, output = outputs

                if config.TEST.FLIP_TEST:
                    # raise NotImplementedError
                    input_flipped = input.flip(4)
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, (list, tuple)):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                ref_images = input[:, 0, :, :].cuda(non_blocking=True)
                input = input[:, 1, :, :, :].cuda(non_blocking=True)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())
                _, _, _, unsup_pred = accuracy(output_unsup.detach().cpu().numpy(),
                                                output_unsup.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                loss = criterion(output, target, target_weight, input, pred_images,
                            joints_pred=torch.from_numpy(pred).cuda(non_blocking=True))

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)
                unsup_preds, _ = get_final_preds(
                    config, output_unsup.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])

                idx += num_images

                if i % config.PRINT_FREQ == 0:
                    msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc)
                    logger.info(msg)

                    prefix = '{}_{}'.format(
                        os.path.join(output_dir, 'val'), i
                    )
                    debug_images = save_debug_images(config, input, meta, target, pred*4, output,
                                prefix, ref_images, pred_images, unsup_pred*4, output_unsup)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def test_time_training(config, loader, dataset, model, model_state_dict,
          criterion, optimizer, optimizer_state_dict, output_dir,
          tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    # for ttp dataset, dataset.length == len(dataset) // ttp_batchsize
    num_samples = dataset.length // config.TEST.DOWNSAMPLE
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    idx = 0

    print("Number of samples: {}".format(num_samples))
    sup_loss_fn = JointsMSELoss(use_target_weight=False).cuda()
    unsup_loss_fn = FinetuneUnsupLoss(config).cuda()
    sup_loss = AverageMeter()
    unsup_loss = AverageMeter()

    end = time.time()
    reset_count = 0
    is_first_frame = True
    input_queue = []
    sup_sample = None  # Add a sup sample in every batch
    for i, (input, target, target_weight, meta) in enumerate(loader):
        if i % config.TEST.DOWNSAMPLE == config.TEST.DOWNSAMPLE - 1:
            # measure data loading time
            data_time.update(time.time() - end)

            # Save the input for future testing
            input_queue.append((i, input, target, target_weight, meta))

            # Train
            for _ in range(config.TEST.REPEAT_STEP):
                # Skip update if we only train on the first frame
                if config.TEST.FIRST_FRAME_ONLY and not is_first_frame:
                    break

                # compute output
                outputs = model(input)
                
                pred_images, output_unsup, output = outputs
                
                ref_images = input[:, 0, :, :].cuda(non_blocking=True)
                images = input[:, 1, :, :, :].cuda(non_blocking=True)
                
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())
                loss = criterion(output, target, target_weight, images, pred_images,
                                joints_pred=torch.from_numpy(pred).cuda(non_blocking=True))

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                iter_sup_loss = sup_loss_fn(output, target, target_weight, images, pred_images,
                                joints_pred=torch.from_numpy(pred).cuda(non_blocking=True))
                sup_loss.update(iter_sup_loss.detach().cpu().item())
                iter_unsup_loss = unsup_loss_fn(output, target, target_weight, images, pred_images,
                                joints_pred=torch.from_numpy(pred).cuda(non_blocking=True))
                unsup_loss.update(iter_unsup_loss.detach().cpu().item())

            losses.update(loss.item(), input.size(0))

        # Continue to train without testing if we want to train on the whole video first
        is_last_frame = meta["is_last_frame"][0]

        for i, input, target, target_weight, meta in input_queue:
            # Nomally input_queue contains only one or no samples
            with torch.no_grad():
                pred_images, output_unsup, output = model(input)

            # all samples in the batch are from the same frame
            # only use the first one (which is not augmented)
            _, avg_acc, cnt, pred = accuracy(output[0:1].detach().cpu().numpy(),
                                            target[0:1].detach().cpu().numpy())
            _, _, _, unsup_pred = accuracy(output_unsup[0:1].detach().cpu().numpy(),
                                            output_unsup[0:1].detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            c = meta['center'][0:1].numpy()
            s = meta['scale'][0:1].numpy()
            score = meta['score'][0:1].numpy()

            preds, maxvals = get_final_preds(
                config, output[0:1].clone().detach().cpu().numpy(), c, s)

            all_preds[idx:idx + 1, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + 1, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + 1, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + 1, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + 1, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + 1, 5] = score

            idx += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # if i % config.PRINT_FREQ == 0:
            if True:
                msg = 'Iter: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, num_samples, batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                logger.info(msg)
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

            reset_count += 1
            if config.TEST.RESET_EVERY > 0 and reset_count >= config.TEST.RESET_EVERY:
                logger.info("Reiniting weight")
                reset_count = 0
                model.load_state_dict(model_state_dict)
                optimizer.load_state_dict(optimizer_state_dict)

            meta_first_sample = {key: meta[key][0:1] for key in meta}
            debug_images = save_debug_images(config, images[0:1], meta_first_sample, target[0:1],
                            pred*4, output[0:1], prefix, ref_images[0:1], pred_images[0:1],
                            unsup_pred*4, output_unsup[0:1])
            for key in debug_images:
                writer.add_image("TTP/" + key, np.transpose(debug_images[key], (2, 0, 1))[[2,1,0]], global_steps)

        input_queue.clear()
        is_first_frame = False
        # reset weight at the end of each video
        if is_last_frame:
            reset_count = 0
            is_first_frame = True
            logger.info("reaching end of a video, reiniting weight")
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            sup_sample = None

    name_values, perf_indicator = dataset.evaluate(
            config, all_preds, output_dir, downsample=config.TEST.DOWNSAMPLE
        )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['train_global_steps'] = global_steps + 1
    
    return perf_indicator


def test_time_training_offline(config, loader, dataset, model, model_state_dict,
          criterion, optimizer, optimizer_state_dict, output_dir,
          tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    # for ttp dataset, dataset.length == len(dataset) // ttp_batchsize
    num_samples = dataset.length
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    idx = 0

    end = time.time()
    for video_idx in range(len(dataset.idx_range)):
        dataset.is_train = True
        dataset.cur_vid_idx = video_idx
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(config.GPUS),
            shuffle=True,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        print("Training on {}th video, length: {}".format(video_idx, len(loader)))
        # Train
        for i, (input, target, target_weight, meta) in enumerate(loader):
            data_time.update(time.time() - end)
            for _ in range(config.TEST.REPEAT_STEP):
                if 'is_auxiliary_frame' in meta.keys() and meta['is_auxiliary_frame'][0]:
                    # skip training if is auxiliary
                    break
                # compute output
                outputs = model(input)
                pred_images, output_unsup, output = outputs
                ref_images = input[:, 0, :, :].cuda(non_blocking=True)
                images = input[:, 1, :, :, :].cuda(non_blocking=True)

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())

                loss = criterion(output, target, target_weight, images, pred_images,
                                joints_pred=torch.from_numpy(pred).cuda(non_blocking=True))
                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

            # only use valid or auxiliary
            if 'is_valid_frame' in meta.keys() and meta['is_valid_frame'][0] == False and \
                meta['is_auxiliary_frame'][0] == False:
                continue

        # Test
        dataset.is_train = False
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(config.GPUS),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        assert config.TEST.DOWNSAMPLE == 1, "Downsample should be 1 in offline scenario"
        for i, (input, target, target_weight, meta) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # We test it with the updated weight.
            with torch.no_grad():
                outputs = model(input)

            # all samples in the batch are from the same frame
            # only use the first one (which is not augmented)

            if len(outputs) == 5:
                pred_images, output_unsup, output, joints_x, joints_y = outputs
            elif len(outputs) == 4:
                pred_images, output_unsup, output, delta = outputs
            else:
                pred_images, output_unsup, output = outputs

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
            _, _, _, unsup_pred = accuracy(output_unsup.detach().cpu().numpy(),
                                            output_unsup.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().detach().cpu().numpy(), c, s)
            
            bs = preds.shape[0]

            all_preds[idx:idx + bs, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + bs, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + bs, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + bs, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + bs, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + bs, 5] = score

            idx += bs

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # if i % config.PRINT_FREQ == 0:
            if True:
                msg = 'Iter: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, num_samples, batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                logger.info(msg)
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

        logger.info("reaching end of a video, reiniting weight")
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        
    assert idx == all_preds.shape[0]
    name_values, perf_indicator = dataset.evaluate(
            config, all_preds, output_dir, downsample=1
        )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['train_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0