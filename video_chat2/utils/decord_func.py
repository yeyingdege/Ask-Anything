from decord import VideoReader
from typing import List
from decord import cpu
import numpy as np
import random
from PIL import Image


def get_frame_indices_with_duration(start_sec, end_sec, fps, total_frame):
    start_frame = int(np.round(start_sec * fps))
    end_frame = int(np.round(end_sec * fps)) - 1
    if end_frame >= total_frame:
        end_frame = total_frame -1
    return range(start_frame, end_frame + 1)


def get_frame_indices(total_frame):
    start_frame = 0
    end_frame = total_frame -1
    return range(start_frame, end_frame + 1)


def random_sample_frame(frame_indices, max_frames):
    cur_len = len(frame_indices)
    if cur_len > max_frames:
        new_frame_indices = []
        rnd_indices    = random.sample(range(0, cur_len), max_frames)
        sorted_indices = sorted(rnd_indices)
        for jj in sorted_indices:
            new_frame_indices.append(frame_indices[jj])
        frame_indices = new_frame_indices
    return frame_indices


def uniform_sample_frame(frame_indices, max_frames):
    cur_len = len(frame_indices)
    if cur_len > max_frames:
        indices = np.linspace(0, cur_len-1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in indices]
    return frame_indices


def random_start_frame(frame_indices, max_frames):
    cur_len = len(frame_indices)
    if cur_len > max_frames:
        start_ind_rand = random.randint(0, cur_len-max_frames)
        frame_indices  = frame_indices[start_ind_rand:start_ind_rand+max_frames]
        
    return frame_indices


def decord_video_given_start_end_seconds(video_pth:str, start_secs:float=-1, end_secs:float=-1,
        skip_interval:int=0, num_video_frames=64):
    # 1. Read video and fps
    #vr = VideoReader(video_pth, ctx=cpu(0))
    vr = VideoReader(video_pth)
    frame_rate = vr.get_avg_fps()
    total_frame  = vr._num_frame

    # 2. Calculate start, stop index
    if start_secs > 0 and end_secs > 0:
        frame_indices = get_frame_indices_with_duration(start_secs, end_secs, frame_rate, total_frame)
    else:
        frame_indices = get_frame_indices(total_frame)

    frame_indices = [i for i in frame_indices]

    ## 3. Skip Sampling to de-redudent
    #frame_indices = frame_indices[::skip_interval]

    # 4. If frames is more thant max_frames
    # Random sample max_frames, keep temporal order
    # Option-1
    #frame_indices = random_sample_frame(frame_indices, num_video_frames)
    # Option-2
    #frame_indices = random_start_frame(frame_indices, max_frames)
    # Option-3
    frame_indices = uniform_sample_frame(frame_indices, num_video_frames)
    
    return frame_indices


def main():
    video_pth = "YSKX3.mp4"
    frames, frame_indices = decord_video_given_start_end_seconds(video_pth, 12.1, 18.0, skip_interval=1)
    # save frames to tmp
    for i, frame in zip(frame_indices, frames):
        #frame_rgb = frame[:, :, ::-1]
        # Convert the RGB frame to a PIL image
        img = Image.fromarray(frame)

        out_name = "./tmp/{}.jpg".format(i)
        img.save(out_name)
    print("frames {}, frame indicies {}".format(frames.shape, frame_indices))


if __name__ == "__main__":
    main()
