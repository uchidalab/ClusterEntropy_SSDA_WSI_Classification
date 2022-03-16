import os
import sys
import logging
import yaml
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from S.eval import eval_net_test, eval_metrics, plot_confusion_matrix
from S.dataset import WSI, get_files
from S.util import fix_seed
from S.model import build_model


def log_out(out_path: str, cm, test_set: str = 'trg_unl'):
    with open(out_path, mode='a') as f:
        f.write(
            f"\n cm ({test_set}):\n{np.array2string(cm, separator=',')}\n"
        )
        val_metrics = eval_metrics(cm)
        f.write("===== eval metrics =====")
        f.write(
            f"\n Accuracy ({test_set}):  {val_metrics['accuracy']}"
        )
        f.write(
            f"\n Precision ({test_set}): {val_metrics['precision']}"
        )
        f.write(f"\n Recall ({test_set}):    {val_metrics['recall']}")
        f.write(f"\n F1 ({test_set}):        {val_metrics['f1']}")
        f.write(f"\n Dice ({test_set}):      {val_metrics['dice']}")
        f.write(f"\n mIoU ({test_set}):      {val_metrics['mIoU']}")


def test_net(
    net,
    files: list,
    classes: list,
    test_set: str,
    output_dir: str,
    project: str = "test_net",
    device=torch.device('cuda'),
    shape: tuple = (256, 256),
    batch_size: int = 32,
    rotation: int = 0,
    logout_path: str = None,
):
    criterion = nn.CrossEntropyLoss()

    dataset = WSI(
        files,
        classes,
        shape,
        transform={"Resize": True, "HFlip": False, "VFlip": False},
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    _, cm = eval_net_test(
        net,
        loader,
        criterion,
        device,
        save_dir=output_dir,
    )

    logging.info(
        f"\n cm ({test_set}):\n{np.array2string(cm, separator=',')}\n"
    )
    val_metrics = eval_metrics(cm)
    logging.info("===== eval metrics =====")
    logging.info(
        f"\n Accuracy ({test_set}):  {val_metrics['accuracy']}"
    )
    logging.info(
        f"\n Precision ({test_set}): {val_metrics['precision']}"
    )
    logging.info(f"\n Recall ({test_set}):    {val_metrics['recall']}")
    logging.info(f"\n F1 ({test_set}):        {val_metrics['f1']}")
    logging.info(f"\n Dice ({test_set}):      {val_metrics['dice']}")
    logging.info(f"\n mIoU ({test_set}):      {val_metrics['mIoU']}")

    if logout_path is not None:
        log_out(out_path=logout_path, cm=cm, test_set=test_set)

    # 軸入れ替え
    cm_rep = np.copy(cm)
    cm_rep = cm_rep[:, [0, 2, 1]]
    cm_rep = cm_rep[[0, 2, 1], :]
    cl_labels = ["Non-\nNeop.", "LSIL", "HSIL"]

    # Not-Normalized
    cm_plt = plot_confusion_matrix(cm_rep, cl_labels, normalize=False, font_size=25, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(cm_rep, cl_labels, normalize=True, font_size=35, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_confmatrix.png"
    )
    plt.clf()
    plt.close()
    return cm, val_metrics


def main_trg(trg_l_wsi: str, config_path: str, test_set: str = "trg_unl", l_trg_set: str = "top"):
    fix_seed(0)
    rotation = 0

    # ==== load config ===== #
    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    output_dir = (
        f"{config['test']['output_dir']}"
        + f"{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_{config['main']['classes']}/")
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    logout_path = output_dir + f"{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_{test_set}-{config['main']['trg_facility']}.txt"

    weight_list = [
        f"{config['test']['weight_dir']}{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_{config['main']['classes']}/" + name
        for name
        in config['test']['weight_names'][l_trg_set][trg_l_wsi]
    ]

    # logging.basicConfig(
    #     level=logging.INFO,
    #     filename=f"{output_dir}{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_{test_set}-{config['main']['trg_facility']}.txt",
    #     format="%(levelname)s: %(message)s",
    # )
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_metrics_list = []
    for cv_num in range(config['main']['cv']):

        logging.info(f"\n\n\n== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_cv{cv_num}_"

        project = (
            project_prefix
            + config['main']['model']
            + "_"
            + config['main']['optim']
            + "_batch"
            + str(config['main']['batch_size'])
            + "_shape"
            + str(config['main']['shape'])
            + "_"
            + test_set
            + "-"
            + config['main']['trg_facility']
        )
        logging.info(f"{project}\n")

        # --- targetデータ --- #
        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"{config['main']['trg_facility']}/"
            + f"{test_set}_wsi.jb"
        )
        files = get_files(
            wsis, config['main']['classes'], config['dataset']['trg_imgs_dir']
        )
        # ------------- #

        net = build_model(
            config['main']['model'], num_classes=len(config['main']['classes'])
        )
        logging.info("Loading model {}".format(weight_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device=device)
        net.load_state_dict(torch.load(weight_path, map_location=device))

        with open(logout_path, mode='a') as f:
            f.write(f"\n\n\n== CV{cv_num} ==")
            f.write(f"{project}\n")
            f.write("Loading model {}".format(weight_path))

        cm, val_metrics = test_net(
            net=net,
            files=files,
            classes=config['main']['classes'],
            test_set=test_set,
            output_dir=output_dir,
            project=project,
            device=device,
            shape=tuple(config['main']['shape']),
            batch_size=config['main']['batch_size'],
            rotation=rotation,
            logout_path=logout_path,
        )

        val_metrics_list.append(list(str(value) for value in val_metrics.values()))
        metrics_keys = list(val_metrics.keys())

        if cv_num == 0:
            cm_all = cm
        else:
            cm_all += cm

    # ===== cv_all ===== #
    logging.info("\n\n== ALL ==")
    project_prefix = f"{config['main']['prefix']}_{config['main']['src_facility']}_{l_trg_set}_{trg_l_wsi}_all_"

    project = (
        project_prefix
        + config['main']['model']
        + "_"
        + config['main']['optim']
        + "_batch"
        + str(config['main']['batch_size'])
        + "_shape"
        + str(config['main']['shape'])
        + "_"
        + test_set
        + "-"
        + config['main']['trg_facility']
    )
    logging.info(f"{project}\n")
    logging.info(
        f"\n cm ({test_set}):\n{np.array2string(cm_all, separator=',')}\n"
    )
    val_metrics_all = eval_metrics(cm_all)
    logging.info("===== eval metrics =====")
    logging.info(
        f"\n Accuracy ({test_set}):  {val_metrics_all['accuracy']}"
    )
    logging.info(
        f"\n Precision ({test_set}): {val_metrics_all['precision']}"
    )
    logging.info(f"\n Recall ({test_set}):    {val_metrics_all['recall']}")
    logging.info(f"\n F1 ({test_set}):        {val_metrics_all['f1']}")
    logging.info(f"\n Dice ({test_set}):      {val_metrics_all['dice']}")
    logging.info(f"\n mIoU ({test_set}):      {val_metrics_all['mIoU']}")

    with open(logout_path, mode='a') as f:
        f.write("\n\n== ALL ==")
        f.write(f"{project}\n")
    if logout_path is not None:
        log_out(out_path=logout_path, cm=cm_all, test_set=test_set)

    # 軸入れ替え
    cm_all = cm_all[:, [0, 2, 1]]
    cm_all = cm_all[[0, 2, 1], :]
    cl_labels = ["Non-\nNeop.", "LSIL", "HSIL"]

    # Not-Normalized
    cm_plt = plot_confusion_matrix(cm_all, cl_labels, normalize=False, font_size=25, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_nn-confmatrix.png"
    )
    plt.clf()
    plt.close()

    # Normalized
    cm_plt = plot_confusion_matrix(cm_all, cl_labels, normalize=True, font_size=35, rotation=rotation)
    cm_plt.savefig(
        output_dir
        + project
        + "_confmatrix.png"
    )
    plt.clf()
    plt.close()

    df = pd.DataFrame(
        val_metrics_list,
        columns=metrics_keys
    )
    df.to_csv(f"{output_dir}{project}_val_metrics.csv", encoding="shift_jis")


# TODO: top, med, btmのそれぞれで平均値を出力
# TODO: l_trg_set (top, med, max) のそれぞれでまとめてtestを実行
# TODO: cvの平均値は各cvの評価値から平均値を算出するように変更

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # config_path = "../ST_MICCAI/config_st_cl[0, 1, 2]_valt20_pretrained.yaml"

    # === top === #
    # main_trg(trg_l_wsi='03_G144', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='03_G34', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='03_G139-1', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='03_G170', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='03_G180', config_path=config_path, test_set="trg_unl", l_trg_set='top')

    # === med === #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # main_trg(trg_l_wsi='03_G212', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='03_G293', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='03_G177', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='03_G95', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='03_G148', config_path=config_path, test_set="trg_unl", l_trg_set='med')

    # # === btm === #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # main_trg(trg_l_wsi='03_G204', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    # main_trg(trg_l_wsi='03_G176', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    # main_trg(trg_l_wsi='03_G58', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    # main_trg(trg_l_wsi='03_G51', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    # main_trg(trg_l_wsi='03_G109-1', config_path=config_path, test_set="trg_unl", l_trg_set='btm')

    config_path = "../ST_MICCAI/config_st_cl[0, 1, 2]_valt20_srcMF0003_pretrained.yaml"

    # # === top === #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # main_trg(trg_l_wsi='0067_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='0056_a-4', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='0289_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='0055_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='top')
    # main_trg(trg_l_wsi='0299_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='top')

    # # === med === #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # main_trg(trg_l_wsi='0421_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='0469_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='0401_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='0037_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='med')
    # main_trg(trg_l_wsi='0030_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='med')

    # === btm === #
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main_trg(trg_l_wsi='0364_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    main_trg(trg_l_wsi='0094_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    main_trg(trg_l_wsi='0418_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    main_trg(trg_l_wsi='0065_a-2', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
    main_trg(trg_l_wsi='0089_a-1', config_path=config_path, test_set="trg_unl", l_trg_set='btm')
