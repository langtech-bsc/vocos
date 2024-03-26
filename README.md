# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

[Audio samples](https://gemelo-ai.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

Vocos is a fast neural vocoder designed to synthesize audio waveforms from acoustic features. Trained using a Generative
Adversarial Network (GAN) objective, Vocos can generate waveforms in a single forward pass. Unlike other typical
GAN-based vocoders, Vocos does not model audio samples in the time domain. Instead, it generates spectral
coefficients, facilitating rapid audio reconstruction through inverse Fourier transform.

This is a version trained to be compatible with widespread mel spectrogram representation implemented in Hifigan.


## Installation

To use Vocos only in inference mode, install it using:

```bash
pip install git+https://github.com/langtech-bsc/vocos.git@matcha
```

If you wish to train the model, install it with additional dependencies:

```bash
pip install vocos[train]
```

## Usage

### Reconstruct audio from mel-spectrogram

```python
import torch

from vocos import Vocos

vocos = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")

mel = torch.randn(1, 80, 256)  # B, C, T
audio = vocos.decode(mel)
```

Copy-synthesis from a file:

```python
import torchaudio

y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=22050)
y_hat = vocos(y)
```


### Integrate with [Matcha](https://github.com/shivammehta25/Matcha-TTS) text-to-speech model

See [example notebook](notebooks/matcha_inference.ipynb).

## Pre-trained models

| Model Name                                                             | Dataset       | Training Iterations | Parameters 
|------------------------------------------------------------------------|---------------|-------------------|------------|
| [BSC-LT/vocos-mel-22khz](https://huggingface.co/BSC-LT/vocos-mel-22khz)| LibriTTS + LJSpeech + openslr69 + festcat | 1.8M | 13.5M |
| [BSC-LT/vocos-mel-22khz-cat](https://huggingface.co/BSC-LT/vocos-mel-22khz-cat)| openslr69 + festcat lafresca | 1.5M | 13.5M

## Training

Prepare a filelist of audio files for the training and validation set:

```bash
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

Fill a config file, e.g. [vocos-matcha.yaml](configs%2Fvocos-matcha.yaml), with your filelist paths and start training with:

```bash
python train.py -c configs/vocos-matcha.yaml
```

Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

## Citation

If this code contributes to your research, please cite our work:

```
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```

## Funding

This work has been promoted and financed by the Generalitat de Catalunya through the [Aina project](https://projecteaina.cat/).

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
