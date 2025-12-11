"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas

import numpy as np
import torchvision
import echonet
import random
import torch
import cv2
from glob import glob


class EchoNetOne(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
    """

    def __init__(self, root=None, split="train", target_type="EF", length=32, period=2,
                 max_length=250):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.length = length
        self.max_length = max_length
        self.period = period

        self.fnames, self.outcome = [], []
        self.return_key = False

        # Load video-level labels
        with open(os.path.join(self.root, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        if data["Split"].dtype == np.int64:
            offset = int(os.environ.get('S_OFFSET', '0'))
            print(f"Warning: Split is kfold (0-9) instead of train/val/test; converting to train/val/test with offset {offset}.")
            sref = ["TRAIN"] * 8 + ["VAL"] + ["TEST"]
            smap = {(i+offset)%10:sref[i] for i in range(10)}
            data["Split"] = data["Split"].map(smap)
                # {0: "TRAIN", 1: "TRAIN", 2: "TRAIN", 3: "TRAIN", 4: "TRAIN", 5: "TRAIN", 6: "TRAIN", 7: "TRAIN", 8: "VAL", 9: "TEST"})
        data["Split"].map(lambda x: x.upper())
        print(data["Split"].unique())

        if self.split != "ALL":
            data = data[data["Split"] == self.split]

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()
        print(len(self.fnames))
        self.fnames = [fn + ".avi" if os.path.splitext(fn)[1] == "" else fn for fn in self.fnames]  # Assume avi if no suffix
        self.outcome = data.values.tolist()

        # Check that files are present
        missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing) != 0:
            print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

        # Load traces
        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        if os.path.exists(os.path.join(self.root, "VolumeTracings.csv")):
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                # assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                if header != ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
                    header = "BAD_HEADER"
                if not header == "BAD_HEADER":
                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        filename = filename + ".avi" if os.path.splitext(filename)[1] == "" else filename
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))

            if not header == "BAD_HEADER":
                for filename in self.frames:
                    for frame in self.frames[filename]:
                        self.trace[filename][frame] = np.array(self.trace[filename][frame])

                # A small number of videos are missing traces; remove these videos
                keep = [len(self.frames[f]) >= 2 for f in self.fnames]
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                # for f, k in zip(self.fnames, keep):
                #     if k:
                #         continue
                #     else:
                #         print('fnames: ', f)
                #         temp = self.frames[f]
                #         print('frames len < 2: ', temp)

                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        else:
            print("Warning: VolumeTracings.csv not found; no traces will be loaded.")

        assert len(self.fnames) > 0, "No videos found."

    def get_length(self):
        return len(self.fnames)

    def set_return_key(self, flag):
        self.return_key = flag

    def get_frames(self, index, return_name=False):
        # Find filename of video
        video_path = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video_path)  #  C T H W
        video = video.astype(np.float32)

        # Gather targets
        key = self.fnames[index]
        two_idx = self.frames[key]
        small_idx = int(min(two_idx))  ## SmallIndex
        large_inx = int(max(two_idx))  ## LargeIndex
        small_frames = video[:, small_idx, :, :]
        large_frames = video[:, large_inx, :, :]

        middle_inx = int((small_idx + large_inx)/2)
        middle_gt = video[:, middle_inx, :, :]

        if return_name:
            return small_frames, middle_gt, large_frames, video_path
        else:
            return small_frames, middle_gt, large_frames

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.get_frames(index, self.return_key)
        img0 = np.transpose(img0, (1, 2, 0))
        gt = np.transpose(gt, (1, 2, 0))
        img1 = np.transpose(img1, (1, 2, 0))

        if 'TRAIN' in self.split:
            # img0, gt, img1 = self.aug(img0, gt, img1, 112, 112) do not need crop
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).unsqueeze(0)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).unsqueeze(0)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0)

        img_set = torch.cat((img0, img1, gt), 0)

        return img_set

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class EchoNetReID(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, reid_path=None, split="train", target_type="SmallFrame", length=32, period=2,
                 max_length=250):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.length = length
        self.max_length = max_length
        self.period = period

        self.reid_path = os.path.join(reid_path, self.split)
        # self.all_reid = glob(os.path.join(self.reid_path, f"*.pt"))

        self.fnames, self.outcome = [], []
        self.return_key = False

        # Load video-level labels
        with open(os.path.join(self.root, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        if data["Split"].dtype == np.int64:
            offset = int(os.environ.get('S_OFFSET', '0'))
            print(f"Warning: Split is kfold (0-9) instead of train/val/test; converting to train/val/test with offset {offset}.")
            sref = ["TRAIN"] * 8 + ["VAL"] + ["TEST"]
            smap = {(i+offset)%10:sref[i] for i in range(10)}
            data["Split"] = data["Split"].map(smap)
                # {0: "TRAIN", 1: "TRAIN", 2: "TRAIN", 3: "TRAIN", 4: "TRAIN", 5: "TRAIN", 6: "TRAIN", 7: "TRAIN", 8: "VAL", 9: "TEST"})
        data["Split"].map(lambda x: x.upper())
        print(data["Split"].unique())

        if self.split != "ALL":
            data = data[data["Split"] == self.split]

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()
        print(len(self.fnames))
        self.fnames = [fn + ".avi" if os.path.splitext(fn)[1] == "" else fn for fn in self.fnames]  # Assume avi if no suffix
        self.outcome = data.values.tolist()

        # Check that files are present
        missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing) != 0:
            print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

        # Load traces
        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        if os.path.exists(os.path.join(self.root, "VolumeTracings.csv")):
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                # assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                if header != ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
                    header = "BAD_HEADER"
                if not header == "BAD_HEADER":
                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        filename = filename + ".avi" if os.path.splitext(filename)[1] == "" else filename
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))

            if not header == "BAD_HEADER":
                for filename in self.frames:
                    for frame in self.frames[filename]:
                        self.trace[filename][frame] = np.array(self.trace[filename][frame])

                # A small number of videos are missing traces; remove these videos
                keep = [len(self.frames[f]) >= 2 for f in self.fnames]
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                # for f, k in zip(self.fnames, keep):
                #     if k:
                #         continue
                #     else:
                #         print('fnames: ', f)
                #         temp = self.frames[f]
                #         print('frames len < 2: ', temp)

                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        else:
            print("Warning: VolumeTracings.csv not found; no traces will be loaded.")

        assert len(self.fnames) > 0, "No videos found."

    def get_length(self):
        return len(self.fnames)

    def set_return_key(self, flag):
        self.return_key = flag

    def get_frames(self, index, return_name=False):
        # Find filename of video
        vid_name = self.fnames[index]
        video_path = os.path.join(self.root, "Videos", vid_name)

        reid_name = os.path.join(self.reid_path, vid_name.split('.')[0] + '.pt')
        vid_reid = torch.load(reid_name, map_location="cpu", weights_only=True)

        # Load video into np.array
        video = echonet.utils.loadvideo(video_path)  #  C T H W
        video = video.astype(np.float32)

        # Gather targets
        key = self.fnames[index]
        two_idx = self.frames[key]
        small_idx = int(min(two_idx))  ## SmallIndex
        large_inx = int(max(two_idx))  ## LargeIndex
        small_frames = video[:, small_idx, :, :]
        large_frames = video[:, large_inx, :, :]

        small_reid = vid_reid[small_idx].unsqueeze(0)
        large_reid = vid_reid[large_inx].unsqueeze(0)
        reid_features = torch.cat([small_reid, large_reid], dim=0)

        middle_inx = int((small_idx + large_inx)/2)
        middle_gt = video[:, middle_inx, :, :]

        if return_name:
            return small_frames, middle_gt, large_frames, video_path
        else:
            return small_frames, middle_gt, large_frames, reid_features

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1, reid_features = self.get_frames(index, self.return_key)
        img0 = np.transpose(img0, (1, 2, 0))
        gt = np.transpose(gt, (1, 2, 0))
        img1 = np.transpose(img1, (1, 2, 0))

        if 'TRAIN' in self.split:
            # img0, gt, img1 = self.aug(img0, gt, img1, 112, 112) do not need crop
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).unsqueeze(0)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).unsqueeze(0)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0)

        img_set = torch.cat((img0, img1, gt), 0)

        return img_set, reid_features

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class EchoNetFlow(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, reid_path=None, flow_path=None, split="train", return_type=None,
                 target_type="SmallFrame", length=32, period=2, max_length=250):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.length = length
        self.max_length = max_length
        self.period = period
        self.return_type = return_type

        self.reid_path = os.path.join(reid_path, self.split)
        self.flow_path = flow_path
        # self.all_reid = glob(os.path.join(self.reid_path, f"*.pt"))

        self.fnames, self.outcome = [], []
        self.return_key = False

        # Load video-level labels
        with open(os.path.join(self.root, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        if data["Split"].dtype == np.int64:
            offset = int(os.environ.get('S_OFFSET', '0'))
            print(f"Warning: Split is kfold (0-9) instead of train/val/test; converting to train/val/test with offset {offset}.")
            sref = ["TRAIN"] * 8 + ["VAL"] + ["TEST"]
            smap = {(i+offset)%10:sref[i] for i in range(10)}
            data["Split"] = data["Split"].map(smap)
                # {0: "TRAIN", 1: "TRAIN", 2: "TRAIN", 3: "TRAIN", 4: "TRAIN", 5: "TRAIN", 6: "TRAIN", 7: "TRAIN", 8: "VAL", 9: "TEST"})
        data["Split"].map(lambda x: x.upper())
        print(data["Split"].unique())

        if self.split != "ALL":
            data = data[data["Split"] == self.split]

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()
        print(len(self.fnames))
        self.fnames = [fn + ".avi" if os.path.splitext(fn)[1] == "" else fn for fn in self.fnames]  # Assume avi if no suffix
        self.outcome = data.values.tolist()

        # Check that files are present
        missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing) != 0:
            print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

        # Load traces
        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        if os.path.exists(os.path.join(self.root, "VolumeTracings.csv")):
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                # assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                if header != ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
                    header = "BAD_HEADER"
                if not header == "BAD_HEADER":
                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        filename = filename + ".avi" if os.path.splitext(filename)[1] == "" else filename
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))

            if not header == "BAD_HEADER":
                for filename in self.frames:
                    for frame in self.frames[filename]:
                        self.trace[filename][frame] = np.array(self.trace[filename][frame])

                # A small number of videos are missing traces; remove these videos
                keep = [len(self.frames[f]) >= 2 for f in self.fnames]
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                # for f, k in zip(self.fnames, keep):
                #     if k:
                #         continue
                #     else:
                #         print('fnames: ', f)
                #         temp = self.frames[f]
                #         print('frames len < 2: ', temp)

                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        else:
            print("Warning: VolumeTracings.csv not found; no traces will be loaded.")

        assert len(self.fnames) > 0, "No videos found."

    def get_length(self):
        return len(self.fnames)

    def set_return_key(self, flag):
        self.return_key = flag

    def get_frames(self, index):
        # Find filename of video
        vid_name = self.fnames[index]
        video_path = os.path.join(self.root, "Videos", vid_name)

        reid_name = os.path.join(self.reid_path, vid_name.split('.')[0] + '.pt')
        vid_reid = torch.load(reid_name, map_location="cpu", weights_only=True)

        # Load video into np.array
        video = echonet.utils.loadvideo(video_path)  #  C T H W
        video = video.astype(np.float32)

        # Gather targets
        key = self.fnames[index]
        two_idx = self.frames[key]
        small_idx = int(min(two_idx))  ## SmallIndex
        large_inx = int(max(two_idx))  ## LargeIndex
        small_frames = video[:, small_idx, :, :]
        large_frames = video[:, large_inx, :, :]

        middle_inx = int((small_idx + large_inx) / 2)
        middle_gt = video[:, middle_inx, :, :]

        # Load reid features
        small_reid = vid_reid[small_idx].unsqueeze(0)
        large_reid = vid_reid[large_inx].unsqueeze(0)
        reid_features = torch.cat([small_reid, large_reid], dim=0)

        # Load optical flow images
        flow_name = os.path.join(self.flow_path, vid_name.split('.')[0] + '.npy')
        flow = np.load(flow_name)

        return small_frames, middle_gt, large_frames, reid_features, flow, video_path

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1, reid_features, flow, video_path = self.get_frames(index)
        img0 = np.transpose(img0, (1, 2, 0))
        gt = np.transpose(gt, (1, 2, 0))
        img1 = np.transpose(img1, (1, 2, 0))

        if 'TRAIN' in self.split:
            # img0, gt, img1 = self.aug(img0, gt, img1, 112, 112) do not need crop
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).unsqueeze(0)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).unsqueeze(0)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0)

        img_set = torch.cat((img0, img1, gt), 0)
        flow = np.transpose(flow, (0, 3, 1, 2))
        data = {'imgs': img_set, 'reid': reid_features, 'flow': flow}

        return data

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm
    import imageio
    from PIL import Image
    import torch.utils.data as Data
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    # parser.add_argument("--data-path", type=str, default="/nvme/share_data/datasets/UCF101/videos")
    parser.add_argument("--data_path", type=str, default="/vol/ideadata/at70emic/projects/EchoSynExt/datasets/EchoNet-Dynamic")
    parser.add_argument('--syn_path', type=str,
                        default='/vol/ideadata/at70emic/projects/TMI23/samples/dynamic_gen_full_synth_ckpt160k_64',
                        help='synthetic dataset path')
    parser.add_argument('--motion_path', type=str,
                        default='/vol/idea_longterm/ot70igyn/EchoNet-Synthetic/samples/ef_motion_60000',
                        help='synthetic dataset path')
    args = parser.parse_args()

    # mean, std = np.array([32.660564, 32.79394, 33.07615]), np.array([49.802944, 49.89014, 50.17571])
    real_kwargs = {"target_type": ['SmallIndex', 'LargeIndex'], "length": 32, "period": 2}
    dataset_train = EchoNetOne(root=args.motion_path, split="train", **real_kwargs)
    # dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=5, num_workers=4, shuffle=False, pin_memory=True,)

    for idx in tqdm(range(len(dataset_train))):
        small_frames, middle_gt, large_frames, next_middle_gt, next_frames = dataset_train.get_frames(idx)
        print('load data')
        # small_frames = small_frames.transpose(1, 2, 0)
        # small_img = Image.fromarray(small_frames.astype('uint8')).convert('RGB')
        # small_img.save(f'./example/{idx}_small' + '.jpg')
        #
        # middle_gt = middle_gt.transpose(1, 2, 0)
        # middle_gt = Image.fromarray(middle_gt.astype('uint8')).convert('RGB')
        # middle_gt.save(f'./example/{idx}_middle' + '.jpg')
        #
        # large_frames = large_frames.transpose(1, 2, 0)
        # large_img = Image.fromarray(large_frames.astype('uint8')).convert('RGB')
        # large_img.save(f'./example/{idx}_large' + '.jpg')

        # video_clip, middle_gt = dataset_train.get_videos(idx)
        # video_clip = video_clip.transpose(1, 2, 3, 0)
        # video_save_path = f'./example/videoclip{idx}' + '.mp4'
        # print(f"video saved in {video_save_path}")
        # print(f"video shape is {video_clip.shape}")
        #
        # imageio.mimwrite(video_save_path, video_clip, fps=30, quality=9)
        #
        # middle_gt = middle_gt.transpose(1, 2, 0)
        # img = Image.fromarray(middle_gt.astype('uint8')).convert('RGB')
        # img.save(f'./example/middle{idx}' + '.jpg')

    print("Read All")

