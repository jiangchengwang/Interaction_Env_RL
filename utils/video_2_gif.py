from moviepy.editor import VideoFileClip


def save_video_to_gif(video_path, output_path):

    if not video_path.endswith('.mp4'):
        video_path += '.mp4'

    if not output_path.endswith('.gif'):
        output_path += '.gif'

    # 读取视频
    clip = VideoFileClip(video_path)
    # end = clip.end if clip.duration < 10 else clip.start + 10
    # # 可以选择截取视频的一部分，比如从10秒到20秒
    # clip = clip.subclip(clip.start, end)  # 可选的，调整时间范围
    # 将视频转换为GIF
    clip.write_gif(output_path, fps=10)  # fps控制GIF的帧率，调整这个值可以改变GIF的流畅度
    clip.close()