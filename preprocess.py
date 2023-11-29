import subprocess
import os
from typing import List, Tuple, Dict, Union
from eval import load_mlcvqa_configuration


def process_video_pairs(video_pairs: List[Tuple[str, str]], temp_folder: str):
    """
    Process a list of video pairs, where each pair is a tuple of (ref, dis)
    Args:
        video_pairs (list): list of tuples
        temp_folder (str): path to temp folder to store the processed videos
    Returns:
        None
    """
    new_video_pairs = []
    for sample_id, video_pair in enumerate(video_pairs):
        curr_pair = []
        for video_path in video_pair:
            video_name = os.path.basename(video_path)
            w, h, r = name2data(video_name)
            d,_,_ = get_video_info(video_path, f"{w}x{h}", r)

            # preprocess if needed (it reurns the current video path if doesn't need to process)
            video_path_proc, error = process_video(w, h, r, d, video_path, os.path.join(temp_folder, update_name(video_name)))
            if error:
                print(f"An error occurred: {error}")
                return None

            # add sample_id and video_path_proc to the new_video_pairs
            curr_pair.append(video_path_proc)
        new_video_pairs.append((curr_pair[0], curr_pair[1])) 
    return new_video_pairs

def process_video(w: int, h: int, r: float, d: float, input_video: str, output_video: str):
    """
    inputs:
        w: width
        h: height
        r: frame rate
        d: duration
        input_video: input video path
        output_video: output video path

    determines if the video needs to be processed
    if yes, then process the video and return the path of new video
    if no, then return the path of the input video
    """

    # if output_video already exists, return the path of the output_video and don't process
    if os.path.exists(output_video):
        print(f'Video already exists: {output_video}')
        return output_video, None

    dur_change = duration_change(d)
    rotate, scale = resolution_change(w, h)

    # check if any of the dur_change, rotate or scale is non-empty
    need_process = dur_change[0]!="" or dur_change[1]!="" or rotate!="" or scale!=""

    if not need_process:
        return input_video, None

    vf_elements = [rotate, scale, dur_change[0]]
    vf = ",".join([i for i in vf_elements if i!=""])
    cmd_vf = f'-vf "{vf}"' if vf!="" else ""
    
    cmd_input = ' '.join(['ffmpeg',
                          '-f', 'rawvideo',
                          '-pixel_format', 'yuv420p',
                          '-video_size', f'{w}x{h}',
                          '-framerate', str(r),
                          '-i', input_video])
    cmd_output = ' '.join(['-pixel_format', 'yuv420p',
                           '-framerate', '30',
                           dur_change[1], output_video])
    command = cmd_input + ' ' + cmd_vf + ' ' + cmd_output
    
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.output.decode()}")
        return None, e.output.decode()
    
    return output_video, None


def resolution_change(w: int, h: int) -> Tuple[str, str]:
    """
    deteremine if resolution change and/or rotation is needed
    returns two phrases for ffmpeg command: rotation and scale
    """
    input_res = f" -video_size {w}x{h}"
    vf_rotate = "transpose=1" if w<h else ""
    if w<h:
        w, h = h, w
    
    res_change = w!=1920 or h!=1080
    vf_scale="scale=1920x1080" if res_change else ""
    
    return vf_rotate, vf_scale


def duration_change(d: float) -> Tuple[str, str]:
    """
    determine if duration change is needed
    returns two phrases for ffmpeg command for -vf and -t
    """
    if d>10:
        return "", " -t 10"
    elif d<10:
        return "tpad=stop_duration=10", ""
    else:
        return "", ""   

def update_name(v_name: str) -> str:
    """
    change the ending of the names to _reco_1920x1080_30fps.yuv
    """
    v_name_reco = v_name.split('.')
    assert v_name_reco[-1] == 'yuv'    
    v_name_reco = ".".join(v_name_reco[:-1]) + '_reco_10sec_1920x1080_30fps.yuv'
    return v_name_reco


def name2data(in_name: str) -> Tuple[int, int, float]:        
    in_name_lst = in_name.split('_')
    ln = len(in_name_lst)
    fps = float(in_name_lst[ln-1].replace('fps.yuv', ''))
    [w, h] = in_name_lst[-2].split('x')
    return int(w), int(h), int(fps)


def get_video_info(in_path: str, in_resolution: str, in_fps: int) -> Tuple[float, int, float]:
    """
    Get the length of a video in seconds and number of frames.
    example:
    ffprobe -f rawvideo -video_size 1080x1920 -framerate 25 -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,duration -print_format csv -hide_banner -i test.yuv
    TODO: check for input format and through error if not yuv420p
    """
    command = ['ffprobe',
               '-v', 'error',
               '-f', 'rawvideo',
               '-video_size', in_resolution,
               '-framerate', str(in_fps),
               '-select_streams v:0 -count_frames -show_entries stream=nb_read_frames,duration -print_format csv -hide_banner',
               '-i', in_path]

        
    command = ' '.join(command)
    res = os.popen(command).read()
    duration_sec = float(res.split(',')[1])
    num_frames = int(res.split(',')[2])
    fps = num_frames / duration_sec
        
    return duration_sec, num_frames, fps


def run_preprocess(args: dict, data_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Preprocesses input videos to ensure they are 1080p, 30fps, and 10 seconds long.

    Args:
        args (dict): A dictionary containing the configuration file path, dataset path, and output file path.
        data_pairs (List[Tuple[str, str]]): A list of tuples containing the paths to the input videos and their corresponding ground truth files.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the paths to the preprocessed videos and their corresponding ground truth files.
    """
    config = load_mlcvqa_configuration(args)
    output_folder = os.path.join(os.path.dirname(args.dataset) if args.dataset else os.path.dirname(args.dis), "preprocessed")
    os.makedirs(output_folder, exist_ok=True)
    print(f'Warning: Input videos are being preprocessed and saved to {output_folder} if they are not 1080p, 30fps, 10sec. The model is trained on 1080p, 30fps, 10sec videos and may not be as accurate with different inputs.')
    data_pairs = process_video_pairs(data_pairs, output_folder)
    return data_pairs, output_folder
