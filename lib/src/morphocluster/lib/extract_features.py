"""Extract features for a EcoTaxa-formatted dataset.
"""
import threading
from typing import Optional
import zipfile
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from torch.nn.functional import avg_pool2d
from torchvision import models
from torchvision.transforms import Resize, ToTensor, Normalize
from tqdm import tqdm


def _check_img_type(img):
    if not isinstance(img, Image.Image):
        raise TypeError("img should be PIL.Image. Got {}".format(type(img)))


def _get_image_info(image):
    return {
        attr: getattr(image, attr)
        for attr in ("shape", "size", "dtype")
        if getattr(image, attr, None) is not None
    }


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` imgects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, verbose=False):
        self.transforms = [t for t in transforms if t is not None]
        self.verbose = verbose

    def __call__(self, img):
        for i, t in enumerate(self.transforms):
            try:
                if self.verbose:
                    print("img before {}:".format(t), _get_image_info(img))
                img = t(img)
            except Exception:
                print("Transformations performed so far:", [self.transforms[:i]])
                raise
        return img

    def __str__(self):
        return "Compose([" + ", ".join(str(t) for t in self.transforms) + "])"


class Crop:
    """
    Crop an image.
    """

    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        _check_img_type(img)

        (w, h) = img.size

        right = w - self.right
        bottom = h - self.bottom

        return img.crop((self.left, self.top, right, bottom))

    def __repr__(self):
        return self.__class__.__name__ + "({!r}, {!r}, {!r}, {!r})".format(
            self.top, self.bottom, self.left, self.right
        )


class Invert:
    """
    Invert the given PIL.Image.
    """

    def __call__(self, img):
        _check_img_type(img)

        return ImageOps.invert(img)

    def __str__(self):
        return "Invert()"


class MinimalCrop:
    """Crop the given PIL.Image to its bounding box.

    Args:
        inverted (boolean): objects are dark, background is light (Default: True).
    """

    def __init__(self, inverted=True):
        self.inverted = inverted

    def __call__(self, img):
        _check_img_type(img)

        img_c = img
        if self.inverted:
            img_c = ImageOps.invert(img_c)

        bbox = img_c.getbbox()

        return img.crop(bbox)

    def __str__(self):
        return "MinimalCrop(inverted={!r})".format(self.inverted)


def pad(img, top, bottom, left, right, mode, value=0, **kwargs):
    if mode == "constant":
        # Use Pillow
        return ImageOps.expand(img, (left, top, right, bottom), value)

    img = np.array(img)

    if mode == "maximum":
        img = np.pad(img, ((top, bottom), (left, right)), mode="maximum")
    else:
        img = np.pad(
            img,
            ((top, bottom), (left, right)),
            mode=mode,
            constant_values=value,
            end_values=value,
        )
    return Image.fromarray(img)


class PadQuadratic:
    """Pads the given PIL.Image so that its a square.

    Args:
        value: Color of the padding
    """

    def __init__(self, min_size=0, value=0, mode="constant"):
        self.value = value
        self.min_size = min_size
        self.mode = mode

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """

        _check_img_type(img)

        w, h = img.size

        long_edge = max(w, h, self.min_size)

        # Borders
        delta_w, delta_h = long_edge - w, long_edge - h
        left, top = delta_w // 2, delta_h // 2
        right, bottom = delta_w - left, delta_h - top

        return pad(img, top, bottom, left, right, self.mode, self.value)

    def __str__(self):
        return "PadQuadratic(min_size={!r}, value={!r})".format(
            self.min_size, self.value
        )


class RandomRot90:
    """Rotate the given np.ndarray a random number of times by 90 degrees."""

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be rotated.

        Returns:
            np.ndarray: Randomly flipped image.
        """

        _check_img_type(img)

        k = np.random.randint(4)

        if k == 0:
            return img
        elif k == 1:
            return img.transpose(Image.ROTATE_90)
        elif k == 2:
            return img.transpose(Image.ROTATE_180)
        elif k == 3:
            return img.transpose(Image.ROTATE_270)
        else:
            raise Exception("k has to be in (0,1,2,3)")

    def __str__(self):
        return "RandomRot90"


class TensorGaussianNoise:
    """Adds gaussian noise to the tensor.

    Args:
        mean, std: Parameters of the Gaussian
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W)

        Returns:
            tensor: Tensor with noise
        """

        # @UndefinedVariable
        return tensor + torch.torch.randn_like(tensor) * self.std + self.mean

    def __str__(self):
        return "TensorGaussianNoise(mean={!r}, std={!r})".format(self.mean, self.std)


class ArchiveDataset(torch.utils.data.Dataset):
    def __init__(self, archive_fn: str, transform=None):
        super().__init__()

        self.transform = transform

        print("Reading archive... ", end="")
        self.archive = zipfile.ZipFile(archive_fn)
        self.lock = threading.Lock()

        with self.archive.open("index.csv") as fp:
            self.dataframe = pd.read_csv(fp, dtype=str, usecols=["object_id", "path"])

        print("Done.")

    def __getitem__(self, index):
        object_id, path = self.dataframe.iloc[index][["object_id", "path"]] # type: ignore

        with self.lock:
            with self.archive.open(path) as fp:
                img = PIL.Image.open(fp)
                img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return object_id, img

    def __len__(self):
        return len(self.dataframe)


class Model(nn.Module):
    def __init__(
        self, architecture, pretrained=False, in_channels=None, num_classes=None
    ):
        super(Model, self).__init__()

        self.architecture = architecture

        if architecture == "resnet18":
            orig_model = models.resnet18(pretrained=pretrained)
            orig_children = list(orig_model.named_children())

            self.features = nn.Sequential(OrderedDict(orig_children[:-2]))

            self.pool = None

            self.classifier = orig_children[-1][1]

            self.in_channels = orig_model.conv1.in_channels
            self.num_features = self.classifier.in_features
            self.num_classes = self.classifier.out_features

            self.input_scale = 256
            self.input_crop = 224

            if in_channels is not None:
                self.in_channels = in_channels
                self.features[0] = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

            if num_classes is not None:
                self.reset_classifier(num_classes)
        elif architecture == "sparse_resnet18":
            # TODO
            # scn.networkArchitectures.SparseResNet(dimension, nInputPlanes, layers)
            raise NotImplementedError("Architecture {} is not implemented.")
        else:
            raise NotImplementedError(
                "Architecture {} is not implemented.".format(architecture)
            )

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.num_features, num_classes)

    def add_feature_bottleneck(self, size):
        self.features.add_module(
            "feature_bottleneck", nn.Conv2d(self.num_features, size, kernel_size=1)
        )
        self.num_features = size
        self.reset_classifier(self.num_classes)

    def freeze_features(self, freeze=True):
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        # Apply feature extractor
        x = self.features(x)

        if self.pool is None:
            self.pool = nn.AvgPool2d(kernel_size=x.size()[2:])

        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply classifier
        return self.classifier(x)

    def flat_features(self, x, last_layer=None):
        # Calculate features
        for name, module in self.features._modules.items():
            x = module(x)

            if name == last_layer:
                break

        # Global pooling
        x = avg_pool2d(x, kernel_size=x.size()[2:])

        # Flatten
        x = x.view(x.size(0), -1)

        return x


def extract_features(
    archive_fn: str,
    features_fn: str,
    parameters_fn: Optional[str],
    normalize=True,
    batch_size=1024,
    # num_workers=0, # ERROR: Parallelization does not work with zip files.
    cuda=True,
    input_mean=(0, 0, 0),
    input_std=(1, 1, 1),
):
    use_cuda = cuda and torch.cuda.is_available()

    if use_cuda:
        map_location = torch.device("cuda")
        print("Using CUDA.")
    else:
        map_location = torch.device("cpu")

    if parameters_fn is not None:
        parameters = torch.load(parameters_fn, map_location=map_location)
        in_channels = parameters["features.conv1.weight"].shape[1]
        n_classes = parameters["classifier.weight"].shape[0]
        pretrained = False
    else:
        print("Using pretrained model.")
        parameters = None
        in_channels = None # Leave model unchanged
        pretrained = True # Use pretrained weights
        n_classes = None # Leave model unchanged

    model = Model(
        "resnet18", pretrained=pretrained, in_channels=in_channels, num_classes=n_classes
    )

    if parameters is not None:
        if "features.feature_bottleneck.weight" in parameters:
            bottleneck = parameters["features.feature_bottleneck.weight"].shape[0]
            model.add_feature_bottleneck(bottleneck)

        model.load_state_dict(parameters)

    if use_cuda:
        model = model.cuda()

    transform = Compose(
        [
            MinimalCrop(),
            PadQuadratic(128, value=(255, 255, 255)),
            Resize(128),
            ToTensor(),
            Normalize(input_mean, input_std)
        ]
    )


    dataset = ArchiveDataset(archive_fn, transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # num_workers has to be 0 because ArchiveDataset is not threadsafe
        pin_memory=True,
    )

    model.eval()

    print("Calculating features...")
    with torch.no_grad(), h5py.File(features_fn, "w") as f_features:
        n_objects = len(dataset)
        n_features = model.num_features

        h5_vstr = h5py.special_dtype(vlen=str)
        h5_objids = f_features.create_dataset("object_id", (n_objects,), dtype=h5_vstr)
        h5_features = f_features.create_dataset(
            "features", (n_objects, n_features), dtype="float32"
        )
        h5_targets = f_features.create_dataset("targets", (n_objects,), dtype="int8")

        offset = 0
        for objids, inputs in tqdm(data_loader, unit="batch"):
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)

            # Run batch through the model
            features = model.flat_features(inputs)

            if normalize:
                length = torch.norm(features, p=2, dim=1, keepdim=True)
                features = features.div(length)

            features = features.data.cpu().numpy()

            batch_size = len(features)

            h5_objids[offset : offset + batch_size] = objids
            h5_features[offset : offset + batch_size] = features
            h5_targets[offset : offset + batch_size] = -1

            offset += batch_size

    print("Done.")
