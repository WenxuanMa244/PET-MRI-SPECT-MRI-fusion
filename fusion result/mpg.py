import cv2
import os

# 创建保存帧的文件夹
output_folder = 'video_frames'
os.makedirs(output_folder, exist_ok=True)
print(f"帧将保存到文件夹: {output_folder}")

# 打开MPG视频文件
cap = cv2.VideoCapture(r'C:\Users\Administrator\Downloads\spatial.mpg')

# 检查是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频信息
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频信息:")
print(f"总帧数: {frame_count}")
print(f"帧率: {fps} FPS")
print(f"分辨率: {width}x{height}")

# 帧计数器
frame_num = 0
saved_frames = 0

# 读取并保存每一帧
while True:
    ret, frame = cap.read()
    
    # 如果到达视频末尾，退出循环
    if not ret:
        break
    
    # 保存当前帧为BMP文件
    frame_filename = os.path.join(output_folder, f'frame_{saved_frames:06d}.bmp')
    cv2.imwrite(frame_filename, frame)
    saved_frames += 1
    
    # 显示进度
    if saved_frames % 10 == 0:
        print(f"已保存 {saved_frames} 帧...")
    
    # 显示帧（可选，可以注释掉以加快处理速度）
    cv2.imshow('Frame Extraction', frame)
    
    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"帧提取完成！共保存了 {saved_frames} 帧到 {output_folder} 文件夹")

# 释放资源
cap.release()
cv2.destroyAllWindows()