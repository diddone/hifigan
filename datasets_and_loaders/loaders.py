from torch.utils.data import DataLoader
from .ljspeech_dataset import LJspeechDataset
from .collator import collate_fn

def get_training_loader(dataset, params):

    return DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
