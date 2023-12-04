import os
from parse import parse
import argparse
from easydict import EasyDict as edict
import yaml
import torch
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tridivb_slowfast_feature_extractor.datasets.videoset import VideoSet
# from fb_slowfast.slowfast.datasets.utils import pack_pathway_output #TODO: remove
from slowfast.datasets.utils import pack_pathway_output
import av
from PIL import Image
import matplotlib.pyplot as plt

class ImageCreator:
    """
    Takes in a dataset and creates images from it
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def create_images(self, frames, output_folder):
        """
        Create images from the given dataset
        Args:
            frames (list): a list of numpy arrays with shape (height, width, channels)
            output_folder (Path): the output folder to write the images to

        Returns:
            None
        """
        frames = np.array(frames)

        # create output folder
        os.makedirs(output_folder, exist_ok=True)
        print("Created directory:", output_folder)

        for frame_no, frame in enumerate(frames):
            frame_name = f'{str(frame_no + 1).zfill(5)}.png'
            plt.imsave(os.path.join(output_folder, frame_name), frame)


class VideoDataSet(VideoSet):

    def __init__(self, cfg, vid_path, vid_id, read_vid_file=False):
        """
        Construct the video loader for a given video.
        
        Args:
            cfg (CfgNode): configs. Details can be found in yaml file.
            vid_path (str): path to the video.
            vid_id (int): video id.
            read_vid_file (bool): flag to turn on/off reading video files.
        """
        self.cfg = cfg

        self.vid_path = vid_path
        self.vid_id = vid_id
        self.read_vid_file = read_vid_file

        self.in_fps = cfg.DATA.IN_FPS
        self.out_fps = cfg.DATA.OUT_FPS
        self.step_size = cfg.DATA.SAMPLING_RATE

        self.out_size = cfg.DATA.NUM_FRAMES

        if isinstance(cfg.DATA.SAMPLE_SIZE, list):
            self.sample_width, self.sample_height = cfg.DATA.SAMPLE_SIZE
        elif isinstance(cfg.DATA.SAMPLE_SIZE, int):
            self.sample_width = self.sample_height = cfg.DATA.SAMPLE_SIZE
        else:
            raise Exception(
                "Error: Frame sampling size type must be a list [Height, Width] or int"
            )

        self.frames = self._get_frames()

    def _get_frames(self):
        """
        Extract frames from the video container
        Returns:
            frames(tensor or list): A tensor of extracted frames from a video or a list of images to be processed
        """
        if self.read_vid_file:
            path_to_vid = (
                os.path.join(self.vid_path, self.vid_id) + self.cfg.DATA.VID_FILE_EXT
            )
            assert os.path.exists(path_to_vid), "{} file not found".format(path_to_vid)

            frames = None
            frames = tuple()
            if self.cfg.DATA.VID_FILE_EXT in (".YUV", ".yuv"):
                try:
                    # Open YUV file
                    yuv_file = open(path_to_vid, "rb")

                    # Get frame width and height
                    frame_width = self.cfg.DATA.YUV_WIDTH
                    frame_height = self.cfg.DATA.YUV_HEIGHT
                    
                    # Initialize empty list to store frames
                    # frames = []
                    
                    # Read and process each frame
                    frame_counter = 0
                    while True:
                        # Read Y, U, and V planes
                        y_plane = yuv_file.read(frame_width * frame_height)
                        u_plane = yuv_file.read(frame_width * frame_height // 4)
                        v_plane = yuv_file.read(frame_width * frame_height // 4)

                        # Check if we have reached the end of the file
                        if not y_plane or not u_plane or not v_plane:
                            break
                        
                        # Convert YUV planes to RGB image
                        y_plane = np.frombuffer(y_plane, dtype=np.uint8).reshape((frame_height, frame_width))
                        u_plane = np.frombuffer(u_plane, dtype=np.uint8).reshape((frame_height // 2, frame_width // 2))
                        v_plane = np.frombuffer(v_plane, dtype=np.uint8).reshape((frame_height // 2, frame_width // 2))

                        # Resize u and v color channels to be the same size as y
                        u_plane = cv2.resize(u_plane, (y_plane.shape[1], y_plane.shape[0]))
                        v_plane = cv2.resize(v_plane, (y_plane.shape[1], y_plane.shape[0]))
                        yvu = cv2.merge((y_plane, v_plane, u_plane)) # Stack planes to 3D matrix (use Y,V,U ordering)

                        rgb_image = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2RGB)
                        
                        # Save the frame (for debugging)
                        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                        
                        # Resize and store the frame
                        in_frame = np.resize(rgb_image, (self.sample_height, self.sample_width, 3))
                        frames += (in_frame[np.newaxis, :, :, :],)
                        
                        frame_counter += 1
                    frames = np.concatenate(frames, axis=0)
                    # Close YUV file
                    yuv_file.close()
               
                except Exception as e:
                    print(
                        "Failed to load video from {} with error {}".format(path_to_vid, e)
                    )
            else:
                try:
                    # Load video
                    video_clip = VideoFileClip(path_to_vid, audio=False, fps_source="fps")

                except Exception as e:
                    print(
                        "Failed to load video from {} with error {}".format(path_to_vid, e)
                    )

                for in_frame in video_clip.iter_frames(fps=self.cfg.DATA.IN_FPS):
                    in_frame = cv2.resize(
                        in_frame,
                        (self.sample_width, self.sample_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if frames is None:
                        frames = in_frame[None, ...]
                    else:
                        frames = np.concatenate((frames, in_frame[None, ...]), axis=0)

            frames = self._pre_process_frame(frames)

            return frames
        
        else:
            path_to_frames = os.path.join(self.vid_path, self.vid_id)
            frames = sorted(
                filter(
                    lambda x: x.endswith(self.cfg.DATA.IMG_FILE_EXT),
                    os.listdir(path_to_frames),
                ),
                key=lambda x: parse(self.cfg.DATA.IMG_FILE_FORMAT, x)[0],
            )
            return frames


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        frame_seg = torch.zeros(
            (
                3,
                self.out_size,
                self.cfg.DATA.SAMPLE_SIZE[1],
                self.cfg.DATA.SAMPLE_SIZE[0],
            )
        ).float()

        start = int(index * self.cfg.DATA.STRIDE)
        end = int(index * self.cfg.DATA.STRIDE + self.step_size * self.out_size)
        max_ind = self.frames.shape[1] - 1

        for out_ind, ind in enumerate(range(start, end, self.step_size)):
            if ind < 0 or ind > max_ind:
                continue
            else:
                if self.read_vid_file:
                    frame_seg[:, out_ind, :, :] = self.frames[:, ind, :, :]
                else:
                    frame_seg[:, out_ind, :, :] = self._read_img_file(
                        os.path.join(self.vid_path, self.vid_id), self.frames[ind]
                    )

        # create the pathways
        frame_list = pack_pathway_output(self.cfg, frame_seg)

        return frame_list


    def __len__(self):
        """
        Returns:
            (int): the number of sampled frames in the video.
        """
        # self.frames is a list of frames if read_vid_file is False
        # and a tensor of shape (3, num_frames, height, width) if read_vid_file is True

        num_frames = self.frames.shape[1] if self.read_vid_file else len(self.frames)
        return num_frames // self.cfg.DATA.STRIDE


    def get_frames(self):
        """
        Dataset returns a list of two elements, one for each pathway (Slow and Fast)
        The slow path has all the frames, and the fast path has only a subset of the frames.
        We only need the slow path.

        Returns:
            (list): a list of frames with shape (H, W, C).
        """
        # The slow path is the second element in the list i.e.[1]
        # shape should be: channels x num frames x height x width
        slow_path = self.__getitem__(0)[1]
        frames = []
        for i in range(slow_path.shape[1]):
            # get the frame
            frame = slow_path[:, i, :, :]
            # convert to numpy array, data type uint8
            frame = frame.numpy().astype('uint8')
            # convert to H x W x C
            frame = np.transpose(frame, (1, 2, 0))
            frames.append(frame)
        return frames
