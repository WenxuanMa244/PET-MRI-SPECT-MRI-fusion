import cv2
import os
from PIL import Image
import numpy as np

def extract_video_frames(video_path, output_folder, frame_interval=1, show_preview=True):
    """
    提取视频帧并保存为BMP文件
    
    参数:
        video_path: 视频文件路径
        output_folder: 输出文件夹路径
        frame_interval: 帧间隔（每N帧提取一帧）
        show_preview: 是否显示预览窗口
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    print(f"帧将保存到文件夹: {output_folder}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"总帧数: {total_frames}")
    print(f"帧率: {fps:.2f} FPS")
    print(f"分辨率: {width}x{height}")
    print(f"帧间隔: {frame_interval}")
    print(f"预计提取帧数: {total_frames // frame_interval}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 只保存指定间隔的帧
        if frame_count % frame_interval == 0:
            # 保存为BMP文件
            frame_filename = os.path.join(output_folder, f'frame_{saved_count:06d}.bmp')
            cv2.imwrite(frame_filename, frame)
            
            # 同时保存为JPEG格式以便查看（可选）
            # jpeg_filename = os.path.join(output_folder, f'frame_{saved_count:06d}.jpg')
            # cv2.imwrite(jpeg_filename, frame)
            
            saved_count += 1
            
            # 显示进度
            if saved_count % 5 == 0:
                print(f"已保存 {saved_count} 帧...")
        
        frame_count += 1
        
        # 显示预览
        if show_preview:
            # 调整显示大小以适合屏幕
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Frame Extraction Preview (按Q键退出)', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print(f"帧提取完成！共保存了 {saved_count} 帧到 {output_folder} 文件夹")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    return saved_count

def create_frame_summary(output_folder, frames_per_row=5):
    """创建帧摘要图像"""
    
    # 获取所有BMP文件
    bmp_files = [f for f in os.listdir(output_folder) if f.endswith('.bmp')]
    bmp_files.sort()
    
    if len(bmp_files) == 0:
        print("没有找到BMP文件")
        return
    
    # 只取前20帧做摘要
    summary_frames = bmp_files[:20]
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(os.path.join(output_folder, summary_frames[0]))
    height, width = first_frame.shape[:2]
    
    # 调整单帧大小
    thumb_width = 200
    thumb_height = int(height * thumb_width / width)
    
    # 计算网格大小
    rows = (len(summary_frames) + frames_per_row - 1) // frames_per_row
    cols = min(frames_per_row, len(summary_frames))
    
    # 创建摘要图像
    summary_img = np.zeros((rows * thumb_height, cols * thumb_width, 3), dtype=np.uint8)
    
    for i, frame_file in enumerate(summary_frames):
        frame = cv2.imread(os.path.join(output_folder, frame_file))
        frame = cv2.resize(frame, (thumb_width, thumb_height))
        
        row = i // frames_per_row
        col = i % frames_per_row
        
        y_start = row * thumb_height
        y_end = (row + 1) * thumb_height
        x_start = col * thumb_width
        x_end = (col + 1) * thumb_width
        
        summary_img[y_start:y_end, x_start:x_end] = frame
    
    # 保存摘要图像
    summary_path = os.path.join(output_folder, 'frame_summary.jpg')
    cv2.imwrite(summary_path, summary_img)
    print(f"帧摘要图像已保存到: {summary_path}")
    
    return summary_path

if __name__ == '__main__':
    # 视频文件路径
    video_path = r'C:\Users\Administrator\Downloads\spatial1.mpg'
    
    # 输出文件夹
    output_folder = 'extracted_frames1'
    
    # 提取所有帧（frame_interval=1表示每帧都提取）
    saved_frames = extract_video_frames(
        video_path=video_path,
        output_folder=output_folder,
        frame_interval=1,  # 每1帧提取一帧
        show_preview=True  # 显示预览
    )
    
    # 创建帧摘要（可选）
    if saved_frames > 0:
        create_frame_summary(output_folder)
        
    print("全部完成！")