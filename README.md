# HiFiGAN

Implementation of HifiGAN. Due to computational constraints, the final model has been learning for <=25 hours.

## Load model

[Link](https://drive.google.com/uc?export=download&id=1Ymw4vR--v7uiWNcz2zstzZEoVYLrGJnp) to final model
```python
import gdown
base_model_path = 'https://drive.google.com/uc?export=download&id=1Ymw4vR--v7uiWNcz2zstzZEoVYLrGJnp'
best_tuned_path = 'https://drive.google.com/uc?export=download&id=1oA-1fm27kx80KH5URJ0DV4s5cOj1MrdS'

gdown.download(<choose your path>)
```

## Test model

```bash
test.py --ckpt-path ckpt-tuned.tar -t <test audio folder> -o <your existed output dir>
```

## Wandb report
The [Link](https://wandb.ai/diddone/neural_vocoder/reports/Hifigan--VmlldzozMjA4NTkz?accessToken=3lbe5ocv6py02q6dr7smy1rakr01erhn23q00ihzmklpg29rgslkvmhugh6n120t) to wandb report



