import cv2
import numpy as np

def count_objects(det):
    # Count the number of unique classes in detections
    classes = det[:, -1].unique()
    count = len(classes)
    return count

def draw_bbox_center(det, im0):
    for *xyxy, conf, cls in reversed(det):
        # Draw bounding box
        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        
        # Calculate and draw center point
        center_x = int((xyxy[0] + xyxy[2]) / 2)
        center_y = int((xyxy[1] + xyxy[3]) / 2)
        cv2.circle(im0, (center_x, center_y), 5, (0, 255, 0), -1)
    
    return im0

# Run inference
# ...

# Process predictions
for i, det in enumerate(pred):  # per image
    # ...

    # Count objects
    count = count_objects(det)
    LOGGER.info(f'Number of objects detected: {count}')
    
    # Draw bounding box centers
    im0 = draw_bbox_center(det, im0)
    
    # ...


要将检测到的物体的中心点连起来并保存到图像中，您可以在以下位置添加代码：

在Process predictions的部分，找到以下代码块：
python
Copy code
for *xyxy, conf, cls in reversed(det):
    if save_txt:  # Write to file
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(txt_path + '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    if save_img or save_crop or view_img:  # Add bbox to image
        c = int(cls)  # integer class
        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        annotator.box_label(xyxy, label, color=colors(c, True))
        if save_crop:
            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
在上述代码块的末尾添加以下代码，用于绘制物体中心点的连线：
python
Copy code
# Draw center point and connect
for *xyxy, conf, cls in reversed(det):
    if save_img or view_img:
        c = int(cls)  # integer class
        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        annotator.center_point(xyxy, color=colors(c, True))
        annotator.connect_points(xyxy, color=colors(c, True))

# Save results (image with detections)
if save_img:
    if dataset.mode == 'image':
        cv2.imwrite(save_path, im0)
    else:  # 'video' or 'stream'
        if vid_path[i] != save_path:  # new video
            vid_path[i] = save_path
            if isinstance(vid_writer[i], cv2.VideoWriter):
                vid_writer[i].release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path += '.mp4'
            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer[i].write(im0)
请注意，您需要确保在添加这些代码之前已经导入了annotator模块，该模块用于绘制中心点和连线。此外，如果您希望将连线的绘制保存到图像中，确保设置save_img参数为True。

这样，当运行脚本时，检测到的物体中心点将与边界框一起绘制并保存到图像中。


# Process detections
for i, det in enumerate(pred):  # detections per image
    # ...

    if len(det):
        # ...

        # Draw line between centers of two objects
        if len(det) >= 2:  # 判断是否存在至少两个物体
            center_1 = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]  # 计算第一个物体的中心点坐标
            for j in range(i+1, len(det)):
                xyxy_2 = det[j, :4]
                center_2 = [(xyxy_2[0] + xyxy_2[2]) / 2, (xyxy_2[1] + xyxy_2[3]) / 2]  # 计算第二个物体的中心点坐标
                cv2.line(im0, (int(center_1[0]), int(center_1[1])), (int(center_2[0]), int(center_2[1])), (0, 255, 0), line_thickness)  # 绘制两个中心点之间的连线

    # ...

    # Save results (image with detections)
    if save_img:
        # ...

        if len(det):
            if dataset.mode == 'image':
                # ...

                # Add bbox to image
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=line_thickness)

    # ...

# ...


for i, det in enumerate(pred):  # 遍历每张图像的预测结果
    seen += 1
    if webcam:  # 如果是使用摄像头，批处理大小 >= 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
    else:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    p = Path(p)  # 转换为Path类型
    save_path = str(save_dir / p.name)  # 保存图像的路径
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 保存标签的路径
    s += '%gx%g ' % im.shape[2:]  # 打印图像尺寸的字符串表示
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
    imc = im0.copy() if save_crop else im0  # 用于保存裁剪区域
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 创建Annotator对象用于绘制标注框
    if len(det):
        # 将预测框从img_size缩放到im0尺寸
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # 打印预测结果
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # 每个类别的检测数量
            s += f"{n} {names[int(c)]}{'个' if n > 1 else ''}, "  # 添加到字符串中

        # 写入结果
        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # 写入到文件
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签的格式
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img or save_crop or view_img:  # 添加边界框到图像
                c = int(cls)  # 类别的整数表示
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    # 打印时间（仅推理时间）
    LOGGER.info(f'{s}完成。({t3 - t2:.3f}秒)')

    # 显示结果
    im0 = annotator.result()
    if view_img:
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1毫秒

    # 保存结果（带有检测结果的图像）
    if save_img:
        if dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if vid_path[i] != save_path:  # 新视频
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # 释放之前的视频写入器
                if vid_cap:  # 视频
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # 流
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

# 打印结果
t = tuple(x / seen * 1E3 for x in dt)  # 每张图像的速度
LOGGER.info(f'速度：%.1fms 预处理，%.1fms 推理，%.1fms NMS，每张图像的尺寸为{(1, 3, *imgsz)}' % t)
if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 个标签已保存至 {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"结果已保存至 {colorstr('bold', save_dir)}{s}")
if update:
    strip_optimizer(weights)  # 更新模型（修复SourceChangeWarning警告）



for i, det in enumerate(pred):  # 遍历每张图像的预测结果
    seen += 1
    if webcam:  # 如果是使用摄像头，批处理大小 >= 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
    else:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    p = Path(p)  # 转换为Path类型
    save_path = str(save_dir / p.name)  # 保存图像的路径
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 保存标签的路径
    s += '%gx%g ' % im.shape[2:]  # 打印图像尺寸的字符串表示
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
    imc = im0.copy() if save_crop else im0  # 用于保存裁剪区域
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 创建Annotator对象用于绘制标注框
    if len(det):
        # 将预测框从img_size缩放到im0尺寸
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # 打印预测结果
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # 每个类别的检测数量
            s += f"{n} {names[int(c)]}{'个' if n > 1 else ''}, "  # 添加到字符串中

        # 写入结果
        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # 写入到文件
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签的格式
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img or save_crop or view_img:  # 添加边界框到图像
                c = int(cls)  # 类别的整数表示
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # 连接两个bbox的中心点并添加到图像
                if len(det) == 2:
                    center1 = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
                    for *xyxy2, _, _ in reversed(det):
                        center2 = [(xyxy2[0] + xyxy2[2]) / 2, (xyxy2[1] + xyxy2[3]) / 2]
                        annotator.line(center1, center2, color=colors(c, True))

    # 打印时间（仅推理）
    LOGGER.info(f'{s}完成。 ({t3 - t2:.3f}s)')

    # 显示结果
    im0 = annotator.result()
    if view_img:
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1毫秒

    # 保存结果（带有检测结果的图像）
    if save_img:
        if dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if vid_path[i] != save_path:  # 新视频
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # 释放之前的视频写入器
                if vid_cap:  # 视频
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # 流
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

# 打印结果
t = tuple(x / seen * 1E3 for x in dt)  # 每张图像的速度
LOGGER.info(f'速度：%.1fms 预处理，%.1fms 推理，%.1fms NMS，每张图像的尺寸为{(1, 3, *imgsz)}' % t)
if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 个标签已保存至 {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"结果已保存至 {colorstr('bold', save_dir)}{s}")
if update:
    strip_optimizer(weights)  # 更新模型（修复SourceChangeWarning警告）
