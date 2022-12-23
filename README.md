# HiFiGAN

Implementation of HifiGAN. Due to computational constraints, the final model has been learning for <=25 hours.

## Load model

[Link](https://drive.google.com/uc?export=download&id=1Ymw4vR--v7uiWNcz2zstzZEoVYLrGJnp) to final model
```python
import gdown
model_path = 'https://drive.google.com/uc?export=download&id=1Ymw4vR--v7uiWNcz2zstzZEoVYLrGJnp'

gdown.download(model_path)
```

## Test model

```python
test.py --ckpt-path ckpt-tuned.tar -o <your existed ouput dir>
```

## Wandb report
The [Link](https://wandb.ai/diddone/neural_vocoder/reports/Hifigan--VmlldzozMjA4NTkz) to wandb report



