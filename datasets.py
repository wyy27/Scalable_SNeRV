"""
Datasets & loaders
"""
import math
from utils import *
import torch_dct as dct
     
class MultiVideoDataset(torch.utils.data.Dataset):
    """
    The dataset class for loading videos. Each dataset instance loads all video frames in a folder.
    It optionally performs frame cropping and resizing, and loading video frames in 3D patches.
    It also implemented caching for fast random patch loading. Caching can be set to 'image'/'patch' level.
    """
    def __init__(self, logger, root, name, crop=[-1, -1], resize=[-1, -1], patch_size=[1, -1, -1], cached='none'):
        self.logger = logger
        self.root = os.path.expanduser(root)
        self.name = name
        self.input_img_paths = sorted([f for f in os.listdir(os.path.join(self.root, self.name)) if not f.startswith(".")])

        self.raw_size = self.get_raw_size()
        self.crop = tuple(crop[d] if crop[d] != -1 else self.raw_size[d] for d in range(2))
        self.resize = tuple(resize[d] if resize[d] != -1 else self.crop[d] for d in range(2))

        self.video_size = (len(self.input_img_paths), self.resize[0], self.resize[1])
        self.patch_size = tuple(patch_size[d] if patch_size[d] != -1 else self.video_size[d] for d in range(3))

        h_4x = self.resize[0]//4
        w_4x = self.resize[1]//4    
        self.gt_270p_root = f'./Datasets/UVG/{w_4x}x{h_4x}/'
        self.gt_270p_img_paths = sorted([f for f in os.listdir(os.path.join(self.gt_270p_root, self.name)) if not f.startswith(".")])
        assert len(self.input_img_paths) == len(self.gt_270p_img_paths)

        h_2x = self.resize[0]//2
        w_2x = self.resize[1]//2
        self.gt_540p_root = f'./Datasets/UVG/{w_2x}x{h_2x}/'
        self.gt_540p_img_paths = sorted([f for f in os.listdir(os.path.join(self.gt_540p_root, self.name)) if not f.startswith(".")])
        assert len(self.input_img_paths) == len(self.gt_540p_img_paths)

        assert all(self.video_size[d] % self.patch_size[d] == 0 for d in range(3))
        self.num_patches = tuple(self.video_size[d] // self.patch_size[d] for d in range(3))

        assert cached in ['none', 'image', 'patch']
        self.cached = cached
        self.load_cache()

        self.logger.info(f'VideoDataset:')
        self.logger.info(f'     root: {self.root}    name: {self.name}    number of images: {len(self.input_img_paths)}')
        self.logger.info(f'     video_size: {self.video_size}    patch_size: {self.patch_size}    num_patches: {self.num_patches}')
        self.logger.info(f'     cached: {self.cached}')

    def load_cache(self):
        """
        Caching the images/patches.
        """
        if self.cached == 'image' or self.cached == 'patch':
            self.in_image_cached, self.gt_270p_image_cached, self.gt_540p_image_cached = self.load_all_images()
        else:
            self.in_image_cached, self.gt_270p_image_cached, self.gt_540p_image_cached = None, None, None

        if self.cached == 'patch':
            self.in_patch_cached, self.gt_270p_patch_cached, self.gt_540p_patch_cached = self.load_all_patches()
            self.in_image_cached, self.gt_270p_image_cached, self.gt_540p_image_cached = None, None, None
        else:
            self.in_patch_cached, self.gt_270p_patch_cached, self.gt_540p_patch_cached = None, None, None

        if self.cached == 'patch':
            self.in_image_cached, self.gt_270p_image_cached, self.gt_540p_image_cached = None, None, None

    def get_raw_size(self):
        """
        Get the original video frame size, i.e., before cropping/resizing.
        This assume that all frames have the same size.
        """
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.input_img_paths[0]))
        return img.shape[1:3]

    def load_in_image(self, idx):
        """
        For loading single image (not cached).
        """
        assert isinstance(idx, int)
        in_img = torchvision.io.read_image(os.path.join(self.root, self.name, self.input_img_paths[idx]))
        in_img = torchvision.transforms.functional.center_crop(in_img, self.crop)
        in_img = torchvision.transforms.functional.resize(in_img, self.resize, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)

        return in_img
    
    def load_gt_270p_image(self, idx):
        """
        For loading single image (not cached).
        """
        assert isinstance(idx, int)
        gt_270p_img = torchvision.io.read_image(os.path.join(self.gt_270p_root, self.name, self.gt_270p_img_paths[idx]))
        return gt_270p_img
    
    def load_gt_540p_image(self, idx):
        """
        For loading single image (not cached).
        """
        assert isinstance(idx, int)
        gt_540p_img = torchvision.io.read_image(os.path.join(self.gt_540p_root, self.name, self.gt_540p_img_paths[idx]))
        return gt_540p_img

    def load_in_patch(self, idx):
        """
        For loading single 3D patch (not cached).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        patches = []
        h = idx[1] * self.patch_size[1]
        w = idx[2] * self.patch_size[2]
        for dt in range(self.patch_size[0]):
            t = idx[0] * self.patch_size[0] + dt
            image = self.in_image_cached[t] if self.in_image_cached is not None else self.load_in_image(t)
            patch = image[:, None, h: h + self.patch_size[1], w: w + self.patch_size[2]]
            patches.append(patch)
        return torch.concatenate(patches, dim=1)
    
    def load_gt_270p_patch(self, idx):
        """
        For loading single 3D patch (not cached).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        patches = []
        upscale_factor = 4  # upscaling
        # new_patch_size  = 30
        patch_size = (int(self.patch_size[1]/upscale_factor), int(self.patch_size[2]/upscale_factor))
        h = idx[1] * patch_size[0]
        w = idx[2] * patch_size[1]
        for dt in range(self.patch_size[0]):
            t = idx[0] * self.patch_size[0] + dt
            image = self.gt_270p_image_cached[t] if self.gt_270p_image_cached is not None else self.load_gt_270p_image(t)
            patch = image[:, None, h: h + patch_size[0], w: w + patch_size[1]]
            patches.append(patch)
        return torch.concatenate(patches, dim=1)
    
    def load_gt_540p_patch(self, idx):
        """
        For loading single 3D patch (not cached).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        patches = []
        upscale_factor = 2  # upscaling
        patch_size = (int(self.patch_size[1]/upscale_factor), int(self.patch_size[2]/upscale_factor))
        # new_patch_size  = 30
        h = idx[1] * patch_size[0]
        w = idx[2] * patch_size[1]
        for dt in range(self.patch_size[0]):
            t = idx[0] * self.patch_size[0] + dt
            image = self.gt_540p_image_cached[t] if self.gt_540p_image_cached is not None else self.load_gt_540p_image(t)
            patch = image[:, None, h: h + patch_size[0], w: w + patch_size[1]]
            patches.append(patch)
        return torch.concatenate(patches, dim=1)

    def load_all_images(self):
        in_images = {}
        gt_270p_images = {}
        gt_540p_images = {}
        for t in range(self.video_size[0]):
            in_images[t] = self.load_in_image(t)
            gt_270p_images[t] = self.load_gt_270p_image(t)
            gt_540p_images[t] = self.load_gt_540p_image(t)
        return in_images, gt_270p_images, gt_540p_images

    def load_all_patches(self):
        in_patches = {}
        gt_270p_patches = {}
        gt_540p_patches = {}
        for t in range(self.num_patches[0]):
            for h in range(self.num_patches[1]):
                for w in range(self.num_patches[2]):
                    in_patches[(t, h, w)] = self.load_in_patch((t, h, w))
                    gt_270p_patches[(t, h, w)] = self.load_gt_270p_patch((t, h, w))
                    gt_540p_patches[(t, h, w)] = self.load_gt_540p_patch((t, h, w))
        return in_patches, gt_270p_patches, gt_540p_patches

    def get_image(self, idx):
        """
        For getting single image (either cached or not).
        """
        assert isinstance(idx, int)
        return self.in_image_cached[idx] if self.in_image_cached is not None else self.load_image(idx)

    def get_patch(self, idx):
        """
        For getting single 3D patch (either cached or not).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        return self.in_patch_cached[idx] if self.in_patch_cached is not None else self.load_in_patch(idx), \
               self.gt_270p_patch_cached[idx] if self.gt_270p_patch_cached is not None else self.load_gt_270p_patch(idx), \
               self.gt_540p_patch_cached[idx] if self.gt_540p_patch_cached is not None else self.load_gt_540p_patch(idx),

    def __len__(self):
        return math.prod(self.num_patches)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx_thw = (idx // (self.num_patches[1] * self.num_patches[2]),
                    (idx % (self.num_patches[1] * self.num_patches[2])) // self.num_patches[2],
                    (idx % (self.num_patches[1] * self.num_patches[2])) % self.num_patches[2])
        in_patch, gt_270p_patch, gt_540p_patch = self.get_patch(idx_thw)
        return torch.tensor(idx_thw, dtype=int), torch.clone(in_patch).float() / 255., torch.clone(gt_270p_patch).float() / 255., torch.clone(gt_540p_patch).float() / 255.


def create_multivideo_dataset(args, logger, training):
    """
    Create the dataset instance. Only apply the patch configuration for trainset.
    """
    return MultiVideoDataset(logger, root=args.dataset, name=args.dataset_name,
                        crop=args.crop_size, resize=args.input_size,
                        patch_size=args.patch_size if training else [args.patch_size[0], -1, -1],
                        cached=args.cached)


def create_loader(args, logger, training, dataset):
    """
    Create the dataset loader.
    """
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.eval_batch_size if not training else args.batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2
    )
    return loader


def set_dataset_args(parser):
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--dataset', default='~/Datasets/UVG/1920x1080/', type=str, help='root path of datasets')
    group.add_argument('--dataset-name', default='Beauty', type=str, help='dataset name. dataset/dataset_name should be the path for image folder.')
    group.add_argument('--crop-size', default=[-1, -1], type=int, nargs='+', help='crop size (before resizing to the input)')    
    group.add_argument('--input-size', default=[-1, -1], type=int, nargs='+', help='input size (scaling that apply after cropping)')
    group.add_argument('--patch-size', default=[1, -1, -1], type=int, nargs='+', help='patch size (apply after cropping and scaling)')
    group.add_argument('--cached', type=str, default='none', help='cache setting for the datasets')
    group.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    group.add_argument('--eval-batch-size', type=int, default=1, help='Evaluation batch size')
    group.add_argument('--workers', type=int, default=2, help='Number of workers for dataloader (default: 2)')
    group.add_argument('--pin-mem', type=str_to_bool, default=True, help='Pin memory for dataloader')