import os
import sys
from utils.video_2_gif import save_video_to_gif


if __name__ == '__main__':

    video_path = "./videos"

    for file in os.listdir(video_path):
        if file.endswith(".mp4"):
            save_video_to_gif(os.path.join(video_path, file),
                              os.path.join(video_path, file.replace(".mp4", ".gif")))