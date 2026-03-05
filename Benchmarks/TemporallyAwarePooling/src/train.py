import gc
import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import AverageMeter, getMetaDataTask
import glob
from utils import evaluate as evaluate_spotting
from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate as evaluate_dvc
from nlgeval import NLGEval
from torch.nn.utils.rnn import pack_padded_sequence

import wandb

# Single shared NLGEval instance (avoids recreating and any per-call state buildup)
caption_scorer = NLGEval(no_glove=True, no_skipthoughts=True, metrics_to_omit=['SPICE'])


def _get_process_memory_mb():
    """Current process memory (RSS) in MB, or None if unavailable. Works on macOS and Linux."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = getattr(usage, "ru_maxrss", 0) or 0
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return None

def trainer(phase, train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20,
            device=torch.device('cpu'),
            start_epoch=0,
            initial_best_loss=9e99):

    logging.info("start training" + (f" (resuming from epoch {start_epoch})" if start_epoch > 0 else ""))

    best_loss = initial_best_loss
    epoch_durations = []  # last N epoch times (seconds) for ETA
    max_durations = 5
    training_start = time.time()

    os.makedirs(os.path.join("models", model_name, phase), exist_ok=True)
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        best_model_path = os.path.join("models", model_name, phase, "model.pth.tar")

        # Memory logging (macOS/Linux): if "after train" grows each epoch, leak is in training; if "after validation metrics" jumps and never drops, leak is in validation.
        mem_mb = _get_process_memory_mb()
        if mem_mb is not None:
            logging.info("Process memory (start of epoch %s): %.1f MB", epoch + 1, mem_mb)

        # train for one epoch
        loss_training = train(phase, train_loader, model, criterion,
                              optimizer, epoch + 1, train=True, device=device, run_start_time=training_start)

        mem_mb = _get_process_memory_mb()
        if mem_mb is not None:
            logging.info("Process memory (after train): %.1f MB", mem_mb)

        # evaluate on validation set
        loss_validation = train(phase, val_loader, model, criterion, optimizer, epoch + 1, train=False, device=device, run_start_time=training_start)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            mem_mb = _get_process_memory_mb()
            if mem_mb is not None:
                logging.info("Process memory (before validation metrics): %.1f MB", mem_mb)
            test = validate_captioning if phase == "caption" else validate_spotting
            performance_validation = test(
                val_metric_loader,
                model,
                model_name,
                device=device)
            mem_mb = _get_process_memory_mb()
            if mem_mb is not None:
                logging.info("Process memory (after validation metrics): %.1f MB", mem_mb)

            logging.info("Validation performance at epoch " +
                         str(epoch+1) + " -> " + str(performance_validation))
            
            wandb.log({**{
                f"loss_train_{phase}": loss_training,
                f"loss_val_{phase}": loss_validation,
                "epoch" : epoch,
                }, **{f"{k}_val" : v for k, v in performance_validation.items()}} )
        else:
            wandb.log({
                f"loss_train_{phase}": loss_training,
                f"loss_val_{phase}": loss_validation,
                "epoch" : epoch,
                })

        # Reduce LR on Plateau after patience reached
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")

        if (prevLR < 2 * scheduler.eps and
                scheduler.num_bad_epochs >= scheduler.patience):
            logging.info(
                "Plateau Reached and no more reduction -> Exiting Loop")
            break

        # Estimated time remaining
        epoch_elapsed = time.time() - epoch_start
        epoch_durations.append(epoch_elapsed)
        if len(epoch_durations) > max_durations:
            epoch_durations.pop(0)
        epochs_left = max_epochs - (epoch + 1)
        if epochs_left > 0 and len(epoch_durations) > 0:
            avg_epoch_sec = sum(epoch_durations) / len(epoch_durations)
            remaining_sec = epochs_left * avg_epoch_sec
            hours = int(remaining_sec // 3600)
            minutes = int((remaining_sec % 3600) // 60)
            logging.info(
                f"Estimated time remaining: ~{hours}h {minutes}m "
                f"({epochs_left} epochs at ~{avg_epoch_sec/60:.1f} min/epoch)"
            )

        # Force garbage collection and return freed GPU memory to limit fragmentation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return

def train(phase, dataloader, model, criterion, optimizer, epoch, train=False, device=torch.device('cpu'), run_start_time=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    epoch_start = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, batch in t:
            # measure data loading time
            data_time.update(time.time() - end)
            if phase == "spotting":
                feats, labels = batch
                feats = feats.to(device)
                labels = labels.to(device)
                # compute output
                output = model(feats)

                # hand written NLL criterion
                loss = criterion(labels, output)
            elif phase == "caption":
                (feats, caption), lengths, mask, caption_or, cap_id = batch
                caption = caption.to(device)
                target = caption[:, 1:] #remove SOS token
                lengths = lengths - 1
                #pack_padded_sequence to do less computation
                target = pack_padded_sequence(target, lengths, batch_first=True, enforce_sorted=False)[0]
                mask = pack_padded_sequence(mask[:, 1:], lengths, batch_first=True, enforce_sorted=False)[0]
                feats = feats.to(device)
                # compute output
                # output = model(feats, caption, lengths)
                
                # compute output (audio_embeddings=None until dataset provides them)
                audio = batch[0][2] if len(batch[0]) > 2 else None
                output = model(feats, caption, lengths, audio_embeddings=audio)

                loss = criterion(output[mask], target[mask])
            else:
                NotImplementedError()
            
            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Free batch tensors to reduce memory retention and GPU fragmentation.
            # Must delete batch last: it holds references to feats/caption/etc., so without
            # deleting it those tensors cannot be freed even after we del their names.
            if phase == "caption":
                del feats, caption, output, loss
                if train:
                    del target, mask
                del lengths
            else:
                del feats, labels, output, loss
            del batch

            n = i + 1
            total_batches = len(dataloader)
            pct = 100 * n / total_batches if total_batches else 0
            elapsed = time.time() - epoch_start
            if train:
                desc = f'Train {epoch}: [{n}/{total_batches} {pct:.0f}%] '
            else:
                desc = f'Evaluate {epoch}: [{n}/{total_batches} {pct:.0f}%] '
            if run_start_time is not None:
                total_sec = time.time() - run_start_time
                th, tm = int(total_sec // 3600), int((total_sec % 3600) // 60)
                desc += f'Total {th}h {tm}m '
            else:
                desc += f'Elapsed {elapsed:.0f}s '
            desc += f'Avg batch {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    return losses.avg

def validate_spotting(dataloader, model, model_name, device=torch.device('cpu')):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.to(device)

            # compute output
            output = model(feats)

            all_labels.append(labels.detach().numpy())
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (cls): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    AP = []
    for i in range(1, dataloader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    mAP = np.mean(AP)

    return {"mAP-sklearn" : mAP}

def test_spotting(dataloader, model, model_name, save_predictions=True, NMS_window=30, NMS_threshold=0.5, device=torch.device('cpu')):
    
    split = '_'.join(dataloader.dataset.split)
    output_folder = f"outputs/{split}"
    output_results = os.path.join("models", model_name, output_folder)
    

    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    _, _, _, inv_dict = getMetaDataTask("caption", "SoccerNet", dataloader.dataset.version)

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (game_ID, feat_half1, feat_half2, label_half1, label_half2) in t:
            data_time.update(time.time() - end)

            # Batch size of 1
            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.squeeze(0)
            label_half2 = label_half2.float().squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = []
            for b in range(int(np.ceil(len(feat_half1)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half1) else len(feat_half1)
                feat = feat_half1[start_frame:end_frame].to(device)
                output = model(feat).cpu().detach().numpy()
                timestamp_long_half_1.append(output)
            timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

            timestamp_long_half_2 = []
            for b in range(int(np.ceil(len(feat_half2)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half2) else len(feat_half2)
                feat = feat_half2[start_frame:end_frame].to(device)
                output = model(feat).cpu().detach().numpy()
                timestamp_long_half_2.append(output)
            timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)


            timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
            timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)



            def get_spot_from_NMS(Input, window=60, thresh=0.0):

                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index,0))
                    nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1

                return np.transpose([indexes, MaxValues])

            framerate = dataloader.dataset.framerate
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()

            for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(dataloader.dataset.num_classes):
                    spots = get_spot(
                        timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold)
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        # confidence = predictions_half_1[frame_index, l]

                        seconds = int((frame_index//framerate)%60)
                        minutes = int((frame_index//framerate)//60)

                        prediction_data = dict()
                        prediction_data["gameTime"] = f'{half+1} - {int(minutes):02d}:{int(seconds):02d}'
                        prediction_data["label"] = inv_dict[l]

                        prediction_data["position"] = str(int((frame_index/framerate)*1000))
                        prediction_data["half"] = str(half+1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)
            
            json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: (int(x["half"]), int(x["position"])))
            if save_predictions:
                os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    
    tight = evaluate_spotting(SoccerNet_path=dataloader.dataset.path, 
                Predictions_path=output_results,
                split=dataloader.dataset.split,
                prediction_file="results_spotting.json", 
                version=dataloader.dataset.version, 
                framerate=dataloader.dataset.framerate, metric="tight")
    
    loose = evaluate_spotting(SoccerNet_path=dataloader.dataset.path, 
                Predictions_path=output_results,
                split=dataloader.dataset.split,
                prediction_file="results_spotting.json", 
                version=dataloader.dataset.version, 
                framerate=dataloader.dataset.framerate, metric="loose")
    
    medium = evaluate_spotting(SoccerNet_path=dataloader.dataset.path, 
                Predictions_path=output_results,
                split=dataloader.dataset.split,
                prediction_file="results_spotting.json", 
                version=dataloader.dataset.version, 
                framerate=dataloader.dataset.framerate, metric="medium")

    tight = {f"{k}_tight" : v for k, v in tight.items() if v!= None}
    loose = {f"{k}_loose" : v for k, v in loose.items() if v!= None}
    medium = {f"{k}_medium" : v for k, v in medium.items() if v!= None}

    results = {**tight, **loose, **medium}

    return results

def validate_captioning(dataloader, model, model_name, device=torch.device('cpu')):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    
    with tqdm(dataloader) as t:
        for (feats, caption), lengths, mask, caption_or, cap_id in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.to(device)
            # compute output string
            output = [dataloader.dataset.detokenize(list(model.sample(feats[idx]).detach().cpu())) for idx in range(feats.shape[0])]
            all_outputs.extend(output)
            all_labels.extend(caption_or)

            batch_time.update(time.time() - end)
            end = time.time()
            del feats, output

            desc = f'Test (cap): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    # Pass copies so we can clear our lists immediately and avoid keeping huge refs
    ref_list = [list(all_labels)]
    hyp_list = list(all_outputs)
    scores = caption_scorer.compute_metrics(ref_list=ref_list, hyp_list=hyp_list)
    # Release our large lists so they can be GC'd (NLGEval has its own copies if it kept any)
    all_labels.clear()
    all_outputs.clear()
    del ref_list, hyp_list
    return scores

def test_captioning(dataloader, model, model_name, output_filename = "results_dense_captioning.json", input_filename="results_spotting.json", device=torch.device('cpu')):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_outputs = []
    all_index = []

    split = '_'.join(dataloader.dataset.split)
    output_folder = f"outputs/{split}"
    output_results = os.path.join("models", model_name, f"results_dense_captioning_{split}.zip")

    with tqdm(dataloader) as t:
        for feats, game_id, cap_id in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.to(device)
            output = [dataloader.dataset.detokenize(list(model.sample(feats[idx]).detach().cpu())) for idx in range(feats.shape[0])]
            
            all_outputs.extend(output)
            all_index.extend([(i.item(), j.item()) for i, j in zip(game_id, cap_id)])

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (dense_caption): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)
    
    #store output
    captions = dict(zip(all_index, all_outputs))
    for game_id, game in enumerate(dataloader.dataset.listGames):
        path = os.path.join("models", model_name, output_folder, game, input_filename)
        with open(path, 'r') as pred_file:
            preds = json.load(pred_file)
        for caption_id, annotation in enumerate(preds["predictions"]):
            annotation["comment"] = captions[game_id, caption_id]
        with open(os.path.join("models", model_name, output_folder, game, output_filename), 'w') as output_file:
                    json.dump(preds, output_file, indent=4)
    
    def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])
    
    zipResults(zip_path = output_results,
            target_dir = os.path.join("models", model_name, output_folder),
            filename=output_filename)

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    
    tight = evaluate_dvc(SoccerNet_path=dataloader.dataset.path, Predictions_path=output_results, split=dataloader.dataset.split, version=dataloader.dataset.version, prediction_file=output_filename, window_size=5, include_SODA=False)
    loose = evaluate_dvc(SoccerNet_path=dataloader.dataset.path, Predictions_path=output_results, split=dataloader.dataset.split, version=dataloader.dataset.version, prediction_file=output_filename, window_size=30, include_SODA=False)

    results = {**{f"{k}_tight" : v for k, v in tight.items()}, **{f"{k}_loose" : v for k, v in loose.items()}}

    return results