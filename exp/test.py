import os
import time
import pickle
import logging

import cv2
import numpy as np

from metrics.metric_group import MetricGroup
from rankseg_rma import rankseg_rma
from rankseg_ba import rankseg_ba

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


logger = logging.getLogger(__name__)


def test(model, data_loader, device, epoch, args):
    model.eval()

    dataset = data_loader.dataset
    metric_group = MetricGroup(accuracyD=args.test_config["METRIC"]["accuracyD"],
                               accuracyI=args.test_config["METRIC"]["accuracyI"],
                               accuracyC=args.test_config["METRIC"]["accuracyC"],
                               ECED=args.test_config["METRIC"]["ECED"],
                               ECEI=args.test_config["METRIC"]["ECEI"],
                               SCED=args.test_config["METRIC"]["SCED"],
                               SCEI=args.test_config["METRIC"]["SCEI"],
                               q=args.test_config["METRIC"]["q"],
                               binary=args.data_config["num_classes"] == 2,
                               num_bins=args.test_config["METRIC"]["num_bins"],
                               num_classes=args.data_config["num_classes"],
                               ignore_index=args.data_config["num_classes"])

    predict_method = args.predict_config["METHOD"]
    predict_params = args.predict_config.get("PARAMS", {})

    logger.info(f"Predict method: {predict_method}, temperature: {predict_params.get('temperature', 1.0)}")

    all_predict_time, all_forward_time = 0, 0
    end = time.time()
    for iter, (image, label, image_file) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{iter}|{len(data_loader)}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        prob = model.multi_scale_predict(image,
                                         device,
                                         args.data_config["num_classes"],
                                         args.data_config["crop_size"],
                                         args.test_config["INF"]["flip"],
                                         args.test_config["INF"]["ratios"],
                                         args.test_config["INF"]["stride_rate"],
                                         temperature=predict_params.get('temperature', 1.0)).to(device)

        mid1 = time.time()
        all_forward_time += (mid1 - start)
        toprint += f"Model forward Time: {(mid1 - start):.6f}, "

        if predict_method == 'argmax':
            pred = prob.argmax(1)
        elif predict_method == 'rankseg_rma':
            pred = rankseg_rma(prob, **predict_params).long()
        else:
            raise ValueError(f'Invalid seg_predict: {predict_method}')
        mid2 = time.time()
        all_predict_time += (mid2 - mid1)
        toprint += f"Seg predict Time: {(mid2 - mid1):.6f}, "

        prob, pred = prob.cpu(), pred.cpu()

        metric_group.add(prob, pred, label, image_file)

        if args.test_config["ITER"]["save_pred"]:
            save_pred(pred, label, image_file, dataset.color_map, args, prefix=predict_method+'.')

        end = time.time()
        toprint += f"Metric Time: {(end - mid2):.6f}"

        if iter % args.test_config["ITER"]["log_iters"] == 0:
            logger.info(toprint)

    logger.info(f"All predict time: {(all_predict_time):.6f}, all forward time: {(all_forward_time):.6f}")

    toprint = "\n"
    results = metric_group.value()
    for key, value in results.items():
        if key in ["instanceIoU", "classIoU", "instanceDice", "classDice"]:
            continue
        toprint += f"{key}: {value:.2f}\n"
    toprint = toprint[:-2]
    logger.info(toprint)

    np.savez(
        os.path.join(args.output_dir, f"{predict_method}.instance_and_class_metric.npz"), 
        instance_iou=results["instanceIoU"].cpu().numpy(), class_iou=results["classIoU"].cpu().numpy(), 
        instance_dice=results["instanceDice"].cpu().numpy(), class_dice=results["classDice"].cpu().numpy(),
        image_files=metric_group.accuracy_metric.image_file,
    )

    with open(os.path.join(args.output_dir, "metric_group.pkl"), "wb") as pkl:
        pickle.dump(metric_group, pkl)


def test_medical(model, data_loader, device, epoch, args):
    model.eval()

    val_cases = [False] * (args.data_config["num_cases"] + 1)

    metric_groups = {}
    dataset = data_loader.dataset
    for case in range(args.data_config["num_cases"] + 1):
        metric_groups[case] = MetricGroup(accuracyD=args.test_config["METRIC"]["accuracyD"],
                                          accuracyI=args.test_config["METRIC"]["accuracyI"],
                                          accuracyC=args.test_config["METRIC"]["accuracyC"],
                                          ECED=args.test_config["METRIC"]["ECED"],
                                          ECEI=args.test_config["METRIC"]["ECEI"],
                                          SCED=args.test_config["METRIC"]["SCED"],
                                          SCEI=args.test_config["METRIC"]["SCEI"],
                                          q=args.test_config["METRIC"]["q"],
                                          binary=args.data_config["num_classes"] == 2,
                                          num_bins=args.test_config["METRIC"]["num_bins"],
                                          num_classes=args.data_config["num_classes"],
                                          ignore_index=args.data_config["num_classes"])

    predict_method = args.predict_config["METHOD"]
    predict_params = args.predict_config.get("PARAMS", {})

    instanceIoU, instanceDice, slice_files = [], [], []
    all_predict_time, all_forward_time = 0, 0
    end = time.time()
    for iter, (image, label, image_file) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{iter}|{len(data_loader)}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        prob = model.multi_scale_predict(image,
                                         device,
                                         args.data_config["num_classes"],
                                         args.data_config["crop_size"],
                                         args.test_config["INF"]["flip"],
                                         args.test_config["INF"]["ratios"],
                                         args.test_config["INF"]["stride_rate"]).to(device)

        mid1 = time.time()
        all_forward_time += (mid1 - start)
        toprint += f"Model forward Time: {(mid1 - start):.6f}, "

        if predict_method == 'argmax':
            pred = prob.argmax(1)
        elif predict_method == 'rankseg_rma':
            pred = rankseg_rma(prob, **predict_params).long()
        elif predict_method == 'rankseg_ba':
            pred = rankseg_ba(prob, **predict_params).long()
        else:
            raise ValueError(f'Invalid seg_predict: {predict_method}')
        mid2 = time.time()
        all_predict_time += (mid2 - mid1)
        toprint += f"Seg predict Time: {(mid2 - mid1):.6f}, "

        prob, pred = prob.cpu(), pred.cpu()

        if "qubiq" in args.data_config["dataset"]:
            ignore = label == args.data_config["num_classes"]
            label = (label >= (args.data_config["num_raters"] // 2 + 1)).long()
            label[ignore] = args.data_config["num_classes"]
            label = label.long()

        for i in range(image.shape[0]):
            if args.data_config["dataset"] in ["lits", "kits"]:
                case = int(image_file[i].split("/")[-1].split("_")[0])
            elif "qubiq" in args.data_config["dataset"]:
                case = int(image_file[i].split("/")[-2][-2:])
            else:
                raise NotImplementedError

            val_cases[case] = True

            prob_i = prob[i, :, :, :].unsqueeze(0)
            label_i = label[i, :, :].unsqueeze(0)
            pred_i = pred[i, :, :].unsqueeze(0)
            metric_groups[case].add(prob_i, pred_i, label_i, [image_file[i]])

        if args.test_config["ITER"]["save_pred"]:
            save_pred(pred, label, image_file, dataset.color_map, args, prefix=predict_method+'.')

        end = time.time()
        toprint += f"Metric Time: {(end - mid2):.6f}"

        if iter % args.test_config["ITER"]["log_iters"] == 0:
            logger.info(toprint)

    logger.info(f"All predict time: {(all_predict_time):.6f}, all forward time: {(all_forward_time):.6f}")

    results = {}
    for i, case in enumerate(val_cases):
        if case:
            results_case = metric_groups[i].value()
            slice_files.extend(metric_groups[i].accuracy_metric.image_file)
            for key, value in results_case.items():
                if key in ["instanceIoU", "classIoU", "instanceDice", "classDice"]:
                    if key == "instanceIoU":
                        instanceIoU.extend(value.cpu().numpy())
                    elif key == "instanceDice":
                        instanceDice.extend(value.cpu().numpy())
                    continue
                if key not in results:
                    results[key] = value
                else:
                    results[key] += value

    for key in results:
        results[key] /= sum(val_cases)

    toprint = ""
    for key, value in results.items():
        toprint += f"{key}: {value:.2f}\n"
    toprint = toprint[:-2]
    logger.info(toprint)

    np.savez(
        os.path.join(args.output_dir, f"{predict_method}.instance_and_class_metric.npz"), 
        instance_iou=np.array(instanceIoU), 
        instance_dice=np.array(instanceDice),
        image_files=slice_files,
    )

    with open(os.path.join(args.output_dir, f"metric_groups.pkl"), "wb") as pkl:
        pickle.dump(metric_groups, pkl)


def save_pred(pred, label, image_file, color_map, args, prefix=""):
    pred_dir = os.path.join(args.output_dir, f"{prefix}pred")
    os.makedirs(pred_dir, exist_ok=True)

    pred[label == args.data_config["num_classes"]] = args.data_config["num_classes"]

    for i in range(label.shape[0]):
        pred_i = pred[i, :, :].cpu().numpy()
        label_i = label[i, :, :].cpu().numpy()

        pred_rgb = np.zeros((label.shape[1], label.shape[2], 3))
        label_rgb = np.zeros((label.shape[1], label.shape[2], 3))

        for j in range(len(color_map)):
            pred_rgb[:, :, 2][pred_i == j] = color_map[j][0]
            pred_rgb[:, :, 1][pred_i == j] = color_map[j][1]
            pred_rgb[:, :, 0][pred_i == j] = color_map[j][2]
            label_rgb[:, :, 2][label_i == j] = color_map[j][0]
            label_rgb[:, :, 1][label_i == j] = color_map[j][1]
            label_rgb[:, :, 0][label_i == j] = color_map[j][2]

        label_file = os.path.join(pred_dir, image_file[i].split("/")[-1])
        if label_file.endswith(".jpg"):
            label_file = label_file.replace(".jpg", ".png")
            pred_file = label_file.replace(".png", "_pred.png")
        elif label_file.endswith(".npz"):
            label_file = label_file.replace(".npz", ".png")
            pred_file = label_file.replace(".png", "_pred.png")
        else:
            raise NotImplementedError

        cv2.imwrite(pred_file, pred_rgb)
        cv2.imwrite(label_file, label_rgb)
