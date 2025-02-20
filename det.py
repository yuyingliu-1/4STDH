import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

# 获取YOLOv5的根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将根目录添加到路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# 运行推理的函数
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # 模型权重路径
        source=ROOT / 'data/images',  # 文件/目录/URL/通配符，0表示摄像头
        imgsz=640,  # 推理尺寸（像素）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 每张图像的最大检测数量
        device='',  # cuda设备，如0、0,1,2,3或cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 保存结果到*.txt文件
        save_conf=False,  # 在保存的labels中保存置信度
        save_crop=False,  # 保存裁剪后的预测框
        nosave=False,  # 不保存图像/视频
        classes=None,  # 按类别过滤：--classes 0，或--classes 0 2 3
        agnostic_nms=False,  # 类别不敏感的NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 结果保存路径
        name='exp',  # 结果保存文件夹名称
        exist_ok=False,  # 如果结果文件夹存在，是否覆盖
        line_thickness=3,  # 边界框线条厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 是否是文件路径
    device = select_device(device)  # 选择设备（cuda或cpu）

    # 运行推理前的一些设置
    half &= device.type != 'cpu'  # 半精度推理仅支持CUDA设备
    agnostic_nms |= half  # 半精度推理时使用类别不敏感的NMS

    # 检查文件/目录/URL是否存在，并打印一些信息
    if not is_file and not os.path.isfile(source):
        LOGGER.error(f'文件/目录/URL "{source}" 不存在！')
        return
    LOGGER.info(colorstr('参数: ') + ', '.join(f'{k}={v}' for k, v in locals().items()))

    # 创建结果保存路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 新建文件夹
    save_txt = save_txt or save_crop or save_img  # 是否保存txt文件

    # 检查并下载YOLOv5模型
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # 导入YOLOv5模型定义
    try:
        from models import build_model
        from utils.datasets import letterbox
        model = build_model(weights, imgsz, device=device)  # 加载YOLOv5模型
    except Exception as e:
        LOGGER.error(f'模型加载失败: {e}')
        return

    # DNN模式：将模型转换为ONNX并进行推理
    if dnn:
        save_img = False
        LOGGER.info('将模型转换为ONNX并进行推理...')
        model.model[-1].export = not nosave
        torch.onnx.export(model.model[-1], torch.zeros(1, 3, imgsz, imgsz, device=device),
                          str(save_dir / 'model.onnx'), verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'])
        LOGGER.info(f'ONNX模型保存路径: {save_dir / "model.onnx"}')
        return

    # 半精度推理：模型转换为半精度格式
    if half:
        model.half()

    # 更新所有模型
    if update:
        LOGGER.info('更新所有模型...')
        strip_optimizer(weights)  # 去除模型优化器
        for _weights in [weights] + [str(w) for w in Path(weights).glob('*.pt')]:
            _weights = Path(_weights).resolve()
            if _weights.is_file() and _weights.exists():
                # 更新模型
                LOGGER.info(f'更新 {_weights} ...')
                try:
                    strip_optimizer(_weights)  # 去除优化器
                    # 加载模型并保存
                    checkpoint = torch.load(_weights, map_location=device)
                    model.model.load_state_dict(checkpoint['model'])
                    with torch.no_grad():
                        torch.save({'model': model.model.state_dict()}, _weights)
                    LOGGER.info(f'模型 {_weights} 更新成功!')
                except Exception as e:
                    LOGGER.error(f'模型 {_weights} 更新失败: {e}')
            else:
                LOGGER.warning(f'模型 {_weights} 不存在!')

    # 推理数据装载器
    dataset = LoadImages(source, img_size=imgsz, stride=32)

    # 图像和视频推理
    if not is_file:
        LOGGER.info(f'推理: {source} -> {save_dir}')
        if os.path.isdir(source):
            dataset.imgs = [str(Path(source) / name) for name in os.listdir(source) if
                            name.lower().endswith(IMG_FORMATS)]  # 目录中的所有图像
        else:
            LOGGER.error(f'文件/目录/URL "{source}" 不存在！')
            return

        # 多图像推理
        if len(dataset) > 0:
            LOGGER.info(f"总共有 {len(dataset)} 张图像")
            for path, img, im0s, vid_cap in dataset:
                LOGGER.info(f'图像路径: {path}')
                save_path = str(Path(save_dir) / Path(path).name)  # 保存图像的路径
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # 半精度转换
                img /= 255.0  # 归一化
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # 推理
                pred = model(img, augment=augment)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=agnostic_nms)

                # 处理预测结果
                for i, det in enumerate(pred):  # 对于每个图像
                    im0 = im0s.copy()
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # 保存预测结果
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt or save_img or view_img:
                                label = f'{model.names[int(cls)]} {conf:.2f}'
                                save_one_box(xyxy, im0, save_path, label=label, color=colors(int(cls)), line_thickness=line_thickness,
                                             hide_labels=hide_labels, hide_conf=hide_conf)

                    # 显示预测结果
                    if view_img:
                        cv2.imshow(str(path), im0)
                        if cv2.waitKey(1) == ord('q'):  # 按下 'q' 键退出
                            raise StopIteration

                    # 保存裁剪后的预测框
                    if save_crop:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        s = f'{save_dir}/{Path(path).stem}'
                        for *xyxy, conf, cls in det:
                            w, h = int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                            save_path = str(Path(s) + f'_{w}x{h}' + Path(path).suffix)
                            save_one_box(xyxy, im0s, save_path, BGR=True, crop=True)

                    # 保存文本结果
                    if save_txt:
                        save_path = str(Path(save_dir) / 'labels' / Path(path).name)
                        s = ''
                        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        for *xyxy, conf, cls in det:
                            if save_conf:
                                s += f'{model.names[int(cls)]} {conf:.2f} '
                            else:
                                s += f'{model.names[int(cls)]} '

                            # 将坐标转换为YOLO格式
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            s += ' '.join(map(str, xywh))  # 添加坐标信息
                            s += '\n'
                        with open(save_path, 'a') as f:
                            f.write(s)

                    # 保存图像
                    if save_img:
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:  # 视频模式
                            if vid_cap:  # 视频文件
                                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, dataset.frame)  # 设置当前帧
                                _, im0 = vid_cap.read()
                                cv2.imwrite(save_path, im0)

                    LOGGER.info(f'{i + 1}/{len(pred)}: {save_path}')  # 显示保存的图像路径
        else:
            LOGGER.error(f'没有找到图像文件！支持的文件类型: {IMG_FORMATS}')

    # 视频推理
    else:
        LOGGER.info(f'推理: {source} -> {save_dir}')
        dataset = LoadStreams(source, img_size=imgsz, stride=32)

        # 多视频推理
        if len(dataset) > 0:
            LOGGER.info(f"总共有 {len(dataset)} 个视频")
            for path, vid_cap in dataset.cap:
                save_path = str(Path(save_dir) / Path(path).name)  # 保存视频的路径
                LOGGER.info(f'视频路径: {path}')
                frames = 0

                # 视频推理
                while vid_cap.isOpened():
                    _, im0 = vid_cap.read()
                    if im0 is None:
                        break
                    LOGGER.debug(f'图像 {frames + 1} ({frames + 1}/{len(dataset)}) {path}: ')

                    # 图像预处理
                    img = letterbox(im0, new_shape=imgsz)[0]
                    img = img.transpose(2, 0, 1)[::-1]  # BGR to RGB
                    img = np.ascontiguousarray(img)

                    # 推理
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # 半精度转换
                    img /= 255.0  # 归一化
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # 推理
                    pred = model(img, augment=augment)[0]
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=agnostic_nms)

                    # 处理预测结果
                    for i, det in enumerate(pred):  # 对于每个图像
                        p, s, im0 = path, f'{save_path}_{frames}', im0s.copy()
                        save_path = str(Path(s) + Path(path).suffix)  # 保存图像的路径
                        s += ''  # 图像序号
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # 保存预测结果
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt or save_img or view_img:
                                    label = f'{model.names[int(cls)]} {conf:.2f}'
                                    save_one_box(xyxy, im0, save_path, label=label, color=colors(int(cls)),
                                                 line_thickness=line_thickness, hide_labels=hide_labels,
                                                 hide_conf=hide_conf)

                        # 保存文本结果
                        if save_txt:
                            save_path = str(Path(save_dir) / 'labels' / Path(path).stem) + ('_' + str(frames)) + Path(
                                path).suffix
                            s = ''
                            for *xyxy, conf, cls in det:
                                if save_conf:
                                    s += f'{model.names[int(cls)]} {conf:.2f} '
                                else:
                                    s += f'{model.names[int(cls)]} '

                                # 将坐标转换为YOLO格式
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                s += ' '.join(map(str, xywh))  # 添加坐标信息
                                s += '\n'
                            with open(save_path, 'a') as f:
                                f.write(s)

                        # 保存图像
                        if save_img:
                            cv2.imwrite(save_path, im0)
                            save_path = str(Path(save_dir) / 'labels' / Path(path).stem) + ('_' + str(frames)) + '.txt'
                            with open(save_path, 'a') as file:
                                for *xyxy, conf, cls in reversed(det):
                                    file.write(f'{model.names[int(cls)]} {conf:.2f} '
                                               f'{int(xyxy[0])} {int(xyxy[1])} {int(xyxy[2])} {int(xyxy[3])}\n')

                        # 显示预测结果
                        if view_img:
                            cv2.imshow(p, im0)
                            if cv2.waitKey(1) == ord('q'):  # 按下 'q' 键退出
                                raise StopIteration

                        LOGGER.info(f'{i + 1}/{len(pred)}: {save_path}')  # 显示保存的图像路径
                    frames += 1

        else:
            LOGGER.error(f'没有找到视频文件！支持的文件类型: {VID_FORMATS}')

    LOGGER.info(f'推理完成。结果保存在: {save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='detect.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default='data/images', help='文件/目录/URL/通配符，0表示摄像头')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='推理尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='每张图像的最大检测数量')
    parser.add_argument('--device', default='', help='cuda设备，如0、0,1,2,3或cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存结果为txt文件')
    parser.add_argument('--save-conf', action='store_true', help='在txt结果中保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪后的预测框')
    parser.add_argument('--nosave', action='store_true', help='不保存输出图像和预测结果')
    parser.add_argument('--classes', nargs='+', type=int, help='仅检测指定的类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='使用类别不敏感的NMS')
    parser.add_argument('--augment', action='store_true', help='在推理时进行数据增强')
    parser.add_argument('--update', action='store_true', help='更新模型')
    parser.add_argument('--project', default='runs/detect', help='保存结果的目录')
    parser.add_argument('--name', default='exp', help='保存结果的文件夹名称')
    parser.add_argument('--exist-ok', action='store_true', help='结果目录存在时不报错')
    parser.add_argument('--line-thickness', type=int, default=3, help='边框线宽')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    parser.add_argument('--save-img', action='store_true', help='保存预测结果图像')
    parser.add_argument('--device-check', action='store_true', help='检查设备的相关信息')
    parser.add_argument('--dnn', action='store_true', help='将模型转换为ONNX并进行推理')
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    with torch.no_grad():
        detect(**vars(opt))
