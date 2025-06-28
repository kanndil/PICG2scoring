import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from utils import AverageMeter, calculate_accuracy, calculate_precision_and_recall, calculate_mse_and_mae
from scipy import stats


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, logger, inf_json, device):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()


    end_time = time.time()
    accuracies = AverageMeter()
    results = {"result": defaultdict(list)}

    mae, mse, num_examples = 0., 0., 0
    s_predict = []
    s_target = []
    with torch.no_grad():
        for i, (inputs, image_name, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            # video_ids, segments = zip(*targets)
            
            # target_label = target_label.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            targets = targets.to(device, non_blocking=True)
            
            num_examples += targets.size(0)
            acc = calculate_accuracy(outputs, targets)
            # precision, recall = calculate_precision_and_recall(outputs, targets)
            mse_o, mae_o = calculate_mse_and_mae(outputs, targets)
            mse += mse_o
            mae += mae_o
            # mae = mae / num_examples
            # mse = mse / num_examples
  

            accuracies.update(acc, inputs.size(0))


            outputs = F.softmax(outputs, dim=1)
            outputs_cpu = outputs.cpu()
            outputs_value = np.argmax(outputs_cpu.numpy(), axis=1)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      acc=accuracies))

            targets_r = targets.cpu().numpy()
            for j in range(targets_r.shape[0]):
                image_id = image_name[j]
                output_probs = outputs_cpu[j].numpy().tolist()
                prediction = int(outputs_value[j])
                target = int(targets_r[j])
                results["result"][image_id].append({
                    "target": target,
                    "output_value": prediction,
                    "output": [float(p) for p in output_probs],
                    "acc": float(acc),
                    "mse": float(mse),
                    "mae": float(mae)
                })

            s_predict.append(outputs_value)
            s_target.append(targets_r)
            logger.log({
                'image_name': image_name,
                'target': targets_r,
                'output_value': outputs_value,
                'output': outputs_cpu.numpy(),
                'acc': acc,
                'mse': mse,
                'mae': mae
            })

    # Flatten predictions and targets
    flat_preds = np.concatenate(s_predict)
    flat_targets = np.concatenate(s_target)

    # Spearman Correlation
    res = stats.spearmanr(flat_preds, flat_targets)
    print("\nSpearman Correlation Coefficient:", res.statistic)

    # Per-class evaluation
    print("\nPer-class Evaluation:\n")
    print(classification_report(flat_targets, flat_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(flat_targets, flat_preds))

    # Save detailed results
    with inf_json.open('w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
