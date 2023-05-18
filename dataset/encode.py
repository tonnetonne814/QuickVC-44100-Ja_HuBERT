import argparse
from pathlib import Path
from tqdm import tqdm
import os

import torch
import torchaudio
from torchaudio.functional import resample

### add ###
import torch
from transformers import HubertModel
import random

import os
import glob
import argparse
import logging
import numpy 
from scipy.io.wavfile import read
import torch
MATPLOTLIB_FLAG = False

from scipy.io.wavfile import read
import torch
from torch.nn import functional as F

import pyworld

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def encode_dataset(args):

    filelist = glob.glob(f"./{args.in_dir}/**/*{args.extension}", recursive=True)
    out_files_list = list()

    if args.model == "japanese-hubert-base":
        model = HubertModel.from_pretrained("rinna/japanese-hubert-base")
        model.eval()
        model.to(device)
        
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            #out_path = args.out_dir / in_path.relative_to(args.in_dir)
            #if True:#not os.path.exists(out_path.with_suffix(".npy")):
            try:
                wav, sr = torchaudio.load(in_path)
            except:
                continue
            wav = resample(wav, sr, 16000).to(device)
            with torch.inference_mode():
                units = model(wav)[0].squeeze().cpu().numpy()
            out_files_list.append(out_path)
            numpy.save(out_path.replace(args.extension,".content.npy"), units)

    else:
        print(f"Loading hubert checkpoint")
        hubert = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(device).eval()
        print(f"Encoding dataset at {args.in_dir}")
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            #if True:#not os.path.exists(out_path.with_suffix(".npy")):
            try:
                wav, sr = torchaudio.load(in_path)
            except:
                continue
            wav = resample(wav, sr, 16000)
            wav = wav.unsqueeze(0).to(device)
            with torch.inference_mode():
                units = hubert.units(wav) #[Batch, Frame, Hidden]
            out_files_list.append(out_path)
            numpy.save(out_path.replace(args.extension,".content.npy"), units.squeeze().cpu().numpy())
    
    if args.f0 == "dio":
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            if True:#not os.path.exists(out_path.with_suffix(".npy")):
                try:
                    wav, sr = torchaudio.load(in_path)
                except:
                    continue
                wav = wav[0].to('cpu').detach().numpy().copy()
                f0_dio = compute_f0_dio(
                    wav, sampling_rate=sr, hop_length=512)
                numpy.save(out_path.replace(args.extension,".f0.npy"), f0_dio)

    elif args.f0 == "harvest":
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            if True:#not os.path.exists(out_path.with_suffix(".npy")):
                try:
                    wav, sr = torchaudio.load(in_path)
                except:
                    continue
                wav = wav[0].to('cpu').detach().numpy().copy()
                f0_harvest = compute_f0_harvest(
                    wav, sampling_rate=sr, hop_length=512)
                numpy.save(out_path.replace(args.extension,".f0.npy"), f0_harvest)

    elif args.f0 == "parselmouth":
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            if True:#not os.path.exists(out_path.with_suffix(".npy")):
                try:
                    wav, sr = torchaudio.load(in_path)
                except:
                    continue
                wav = wav[0].to('cpu').detach().numpy().copy()
                f0_parselmouth = compute_f0_parselmouth(
                    wav, sampling_rate=sr, hop_length=512)
                numpy.save(out_path.replace(args.extension,".f0.npy"), f0_parselmouth)

    elif args.f0 == "crepe":
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            if True:#not os.path.exists(out_path.with_suffix(".npy")):
                try:
                    wav, sr = torchaudio.load(in_path)
                except:
                    continue
                wav = wav[0].to('cpu').detach().numpy().copy()
                f0_crepe, _= compute_f0_torchcrepe(
                    wav, sampling_rate=sr, hop_length=512)
                numpy.save(out_path.replace(args.extension,".f0.npy"), f0_crepe)

    
    elif args.f0 == "check_f0_method":
        for in_path in tqdm(filelist):
            out_path = in_path.replace(f"{args.in_dir}", f"{args.out_dir}")
            out_dir = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)
            if True:#not os.path.exists(out_path.with_suffix(".npy")):
                try:
                    wav, sr = torchaudio.load(in_path)
                except:
                    continue
                wav = wav[0].to('cpu').detach().numpy().copy()
                f0_crepe, _= compute_f0_torchcrepe(
                    wav, sampling_rate=sr, hop_length=512)
                f0_harvest = compute_f0_harvest(
                    wav, sampling_rate=sr, hop_length=512)
                f0_dio = compute_f0_dio(
                    wav, sampling_rate=sr, hop_length=512)
                f0_parselmouth = compute_f0_parselmouth(
                    wav, sampling_rate=sr, hop_length=512)
                import matplotlib.pyplot as plt
                import numpy as np
                x = numpy.linspace(0, 1, len(f0_parselmouth))
                plt.plot(x, f0_crepe, label="crepe")
                plt.plot(x, f0_harvest, label="harvest")
                plt.plot(x, f0_dio, label="dio")
                plt.plot(x, f0_parselmouth, label="pm")
                plt.legend()
                plt.show()
                plt.close()


    n_files = len(out_files_list) 
    
    test_list = list()
    for idx in range(int(n_files*0.05)):        # 5% of all files are used for test
        target_idx = random.randint(a=0, b=int(n_files-idx-1))
        path = out_files_list.pop(target_idx)
        path_str =  str(path) + "\n"
        test_list.append(path_str)
    
    train_list = list()
    for path in out_files_list:       
        path_str =  str(path)+ "\n"
        train_list.append(path_str)
    
    os.makedirs("./filelist/", exist_ok=True)
    with open("./filelist/train.txt", mode="w", encoding="utf-8") as f:
        f.writelines(train_list)
    with open("./filelist/test.txt", mode="w", encoding="utf-8") as f:
        f.writelines(test_list)

    


################################################################

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * numpy.log(1 + f0_min / 700)
f0_mel_max = 1127 * numpy.log(1 + f0_max / 700)

def normalize_f0(f0, x_mask, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask

def compute_f0_torchcrepe(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512,device=None,cr_threshold=0.05):
    x = wav_numpy
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 4, "pad length error"
    
    x = torch.from_numpy(x.astype(numpy.float32)).clone()
    F0Creper = CrepePitchExtractor(hop_length=hop_length,f0_min=f0_min,f0_max=f0_max,device=device,threshold=cr_threshold)
    f0,uv = F0Creper(x[None,:].float(),sampling_rate,pad_to=p_len)
    f0[uv<0.5] = 0
    return f0,uv

def compute_f0_harvest(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    x = wav_numpy
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 4, "pad length error"
    
    f0, t = pyworld.harvest(
        x.astype(numpy.double),
        fs=sampling_rate,
        f0_ceil=f0_max,
        f0_floor=f0_min,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pyworld.stonemask(x.astype(numpy.double), f0, t, sampling_rate)
    return resize_f0(f0, p_len)

def compute_f0_parselmouth(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    import parselmouth
    x = wav_numpy
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 4, "pad length error"
    time_step = hop_length / sampling_rate * 1000
    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = numpy.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
    return f0

def resize_f0(x, target_len):
    source = numpy.array(x)
    source[source<0.001] = numpy.nan
    target = numpy.interp(numpy.arange(0, len(source)*target_len, len(source))/ target_len, numpy.arange(0, len(source)), source)
    res = numpy.nan_to_num(target)
    return res

def compute_f0_dio(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    import pyworld
    if p_len is None:
        p_len = wav_numpy.shape[0]//hop_length
    f0, t = pyworld.dio(
        wav_numpy.astype(numpy.double),
        fs=sampling_rate,
        f0_ceil=f0_max,
        f0_floor=f0_min,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pyworld.stonemask(wav_numpy.astype(numpy.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)

def f0_to_coarse(f0):
  is_torch = isinstance(f0, torch.Tensor)
  f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * numpy.log(1 + f0 / 700)
  f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

  f0_mel[f0_mel <= 1] = 1
  f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
  f0_coarse = (f0_mel + 0.5).int() if is_torch else numpy.rint(f0_mel).astype(numpy.int)
  assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
  return f0_coarse


def interpolate_f0(f0):

    data = numpy.reshape(f0, (f0.size, 1))

    vuv_vector = numpy.zeros((data.size, 1), dtype=numpy.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i] # this may not be necessary
            last_value = data[i]

    return ip_data[:,0], vuv_vector[:,0]

from typing import Optional,Union
try:
    from typing import Literal
except Exception as e:
    from typing_extensions import Literal
import numpy as np
import torch
import torchcrepe
from torch import nn
from torch.nn import functional as F
import scipy

#from:https://github.com/fishaudio/fish-diffusion

def repeat_expand(
    content: Union[torch.Tensor, numpy.ndarray], target_len: int, mode: str = "nearest"
):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """

    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, numpy.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]


class BasePitchExtractor:
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        keep_zeros: bool = True,
    ):
        """Base pitch extractor.

        Args:
            hop_length (int, optional): Hop length. Defaults to 512.
            f0_min (float, optional): Minimum f0. Defaults to 50.0.
            f0_max (float, optional): Maximum f0. Defaults to 1100.0.
            keep_zeros (bool, optional): Whether keep zeros in pitch. Defaults to True.
        """

        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError("BasePitchExtractor is not callable.")

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, numpy.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = repeat_expand(f0, pad_to)

        if self.keep_zeros:
            return f0
        
        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # Remove 0 frequency and apply linear interpolation
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = numpy.arange(pad_to) * self.hop_length / sampling_rate

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device),torch.zeros(pad_to, dtype=torch.float, device=x.device)

        if f0.shape[0] == 1:
            return torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0],torch.ones(pad_to, dtype=torch.float, device=x.device)
    
        # Probably can be rewritten with torch?
        f0 = numpy.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        vuv_vector = vuv_vector.cpu().numpy()
        vuv_vector = numpy.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        return f0,vuv_vector


class MaskedAvgPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        # Apply the mask by setting masked elements to zero, or make NaNs zero
        if mask is None:
            mask = ~torch.isnan(x)

        # Ensure mask has the same shape as the input tensor
        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))
        # Create a ones kernel with the same number of channels as the input tensor
        ones_kernel = torch.ones(x.size(1), 1, self.kernel_size, device=x.device)

        # Perform sum pooling
        sum_pooled = nn.functional.conv1d(
            masked_x,
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )

        # Count the non-masked (valid) elements in each pooling window
        valid_count = nn.functional.conv1d(
            mask.float(),
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )
        valid_count = valid_count.clamp(min=1)  # Avoid division by zero

        # Perform masked average pooling
        avg_pooled = sum_pooled / valid_count

        # Fill zero values with NaNs
        avg_pooled[avg_pooled == 0] = float("nan")

        if ndim == 2:
            return avg_pooled.squeeze(1)

        return avg_pooled


class MaskedMedianPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedMedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        if mask is None:
            mask = ~torch.isnan(x)

        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))

        x = F.pad(masked_x, (self.padding, self.padding), mode="reflect")
        mask = F.pad(
            mask.float(), (self.padding, self.padding), mode="constant", value=0
        )

        x = x.unfold(2, self.kernel_size, self.stride)
        mask = mask.unfold(2, self.kernel_size, self.stride)

        x = x.contiguous().view(x.size()[:3] + (-1,))
        mask = mask.contiguous().view(mask.size()[:3] + (-1,)).to(x.device)

        # Combine the mask with the input tensor
        #x_masked = torch.where(mask.bool(), x, torch.fill_(torch.zeros_like(x),float("inf")))
        x_masked = torch.where(mask.bool(), x, torch.FloatTensor([float("inf")]).to(x.device))

        # Sort the masked tensor along the last dimension
        x_sorted, _ = torch.sort(x_masked, dim=-1)

        # Compute the count of non-masked (valid) values
        valid_count = mask.sum(dim=-1)

        # Calculate the index of the median value for each pooling window
        median_idx = (torch.div((valid_count - 1), 2, rounding_mode='trunc')).clamp(min=0)

        # Gather the median values using the calculated indices
        median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

        # Fill infinite values with NaNs
        median_pooled[torch.isinf(median_pooled)] = float("nan")
        
        if ndim == 2:
            return median_pooled.squeeze(1)

        return median_pooled


class CrepePitchExtractor(BasePitchExtractor):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        threshold: float = 0.05,
        keep_zeros: bool = False,
        device = None,
        model: Literal["full", "tiny"] = "full",
        use_fast_filters: bool = True,
    ):
        super().__init__(hop_length, f0_min, f0_max, keep_zeros)

        self.threshold = threshold
        self.model = model
        self.use_fast_filters = use_fast_filters
        self.hop_length = hop_length
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        if self.use_fast_filters:
            self.median_filter = MaskedMedianPool1d(3, 1, 1).to(device)
            self.mean_filter = MaskedAvgPool1d(3, 1, 1).to(device)

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using crepe.


        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        x = x.to(self.dev)
        f0, pd = torchcrepe.predict(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            pad=True,
            model=self.model,
            batch_size=1024,
            device=x.device,
            return_periodicity=True,
        )

        # Filter, remove silence, set uv threshold, refer to the original warehouse readme
        if self.use_fast_filters:
            pd = self.median_filter(pd)
        else:
            pd = torchcrepe.filter.median(pd, 3)

        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, sampling_rate, 512)
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)
        
        if self.use_fast_filters:
            f0 = self.mean_filter(f0)
        else:
            f0 = torchcrepe.filter.mean(f0, 3)

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)[0]

        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if pad_to==None else numpy.zeros(pad_to)
            return rtn,rtn
        
        return self.post_process(x, sampling_rate, f0, pad_to)

###     ###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "--model",
        # help="available models (HuBERT-Soft or HuBERT-Discrete)",
        help="available models (HuBERT-Soft or HuBERT-Discrete or japanese-hubert-base)",
        choices=["soft", "soft", "japanese-hubert-base"],
        default="japanese-hubert-base"
    )
    parser.add_argument(
        "--f0",
        # help="available models (HuBERT-Soft or HuBERT-Discrete)",
        help="available F0 extractor",
        choices=["dio", "parselmouth", "harvest", "crepe"],
        default="harvest"
    )
    parser.add_argument(
        "--in_dir",
        help="path to the dataset directory.",
        default="./dataset/",       ### add ###
        type=Path,
    )
    parser.add_argument(
        "--out_dir",
        help="path to the output directory.",
        default="./dataset/",       ### add ###
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".wav",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
    """
    wav_path = "./dataset/jvs_ver1/jvs001/falset10/wav24kHz16bit/BASIC5000_0235.wav"

    model = HubertModel.from_pretrained("rinna/japanese-hubert-base")
    model.eval()
    model.to(device)
    
    wav, sr = torchaudio.load(wav_path)
    wav_16k = resample(wav, sr, 16000).to(device)
    #wav = wav.unsqueeze(0)
    with torch.inference_mode():
        units = model(wav_16k)
    f0_harvest = compute_f0_harvest(wav[0].to('cpu').detach().numpy().copy(), sampling_rate=sr, hop_length=int(512))
    print("")
    """