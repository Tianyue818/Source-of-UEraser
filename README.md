# UEraser

This repository contains the code of our paper
"Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks".

## Quick start

We provide an example of UEraser training
on CIFAR-10 poisons generated
with [EM](https://github.com/HanxunH/Unlearnable-Examples)
and [LSP](https://github.com/dayu11/Availability-Attacks-Create-Shortcuts)
for testing.
Requires python 3.10,
and please first install the following packages:
```bash
pip install torch, numpy, kornia
```

### EM
From [the offical repository of EM](https://github.com/HanxunH/Unlearnable-Examples),
generate noise for unlearnable examples:
```bash
python3 perturbation.py \
    --config_path             configs/cifar10                \
    --exp_name                path/to/your/experiment/folder \
    --version                 resnet18                       \
    --train_data_type         CIFAR10                        \
    --noise_shape             50000 3 32 32                  \
    --epsilon                 8                              \
    --num_steps               20                             \
    --step_size               0.8                            \
    --attack_type             min-min                        \
    --perturb_type            samplewise                     \
    --universal_stop_error    0.01
```
Replace `main.py` and `trainer.py` in the official code with `main.py` and `trainer.py`
in the EM folder and modify the `lr` (to be 0.01) and `epochs` (e.g. 200) in the respective `.yaml` config file,
for example [this one here](https://github.com/HanxunH/Unlearnable-Examples/blob/main/configs/cifar10/resnet18.yaml).

Train on unlearnable examples with UEraser and evaluate on the clean test set:
```bash
python3 -u main.py \
    --version                 resnet18                       \
    --exp_name                path/to/your/experiment/folder \
    --config_path             configs/cifar10                \
    --train_data_type         PoisonCIFAR10                  \
    --poison_rate             1.0                            \
    --perturb_type            samplewise                     \
    --perturb_tensor_filepath path/to/your/experiment/folder/perturbation.pt \
    --train                                                  \
    --k                       <k>
```
- `<k>`: The number of repeated error-maximizing augmentation sampling.


### LSP
Go to the LSP subfolder and install `scikit-learn`:
```bash
cd LSP
pip install scikit-learn
```
Here are some example commands to train with UEraser on LSP poisons.
UEraser training on LSP-unlearnable CIFAR-10 with ResNet18:
```bash
python cifar_train.py --model <model> --dataset <dataset> --mode <mode> --type <type>
```

The parameter choices for the above commands are as follows:
- Dataset name `<dataset>`: `c10` , `c100`, `svhn`,
  respectively for CIFAR-10, CIFAR-100 and SVHN.
- Model `<model>`: `resnet18`, `resnet50`, `densenet`.
- UEraser variants `<mode>`: `fast`, `standard`, `em`.
- Toggles unlearnable dataset `<clean>`: `unlearn`, `clean`.


## Using UEraser in your training project

```python
from UEraser_func import UEraser
...
# UEraser
for i, (images, labels) in enumerate(data_loader):
    images, labels = images.to(device), labels.to(device)
    if epoch < <EM_epoch>:
        # UEraser-EM
        loss_bar = torch.empty((K, batch_size))
        for j in range(<K>):
            images_tmp = UEraser(images)
            logits_tmp = model(images_tmp)
            loss_tmp = criterion(logits_tmp, labels, reduction='none')
            loss_bar[j] = loss_tmp
        max_loss, _ = torch.max(loss_bar, dim=0)
        loss = torch.mean(max_loss)
    else:
        # UEraser-fast
        images = UEraser(images)
        logits = model(images)
        loss = criterion(logits, labels)
    loss.backward()
...
```
- `<EM_epoch>`: The number of error-maximizing augmentation epochs.
- `<K>`: The number of repetitions.

## Acknowledgement
Training code adapted
from [EM](https://github.com/HanxunH/Unlearnable-Examples)
and [LSP](https://github.com/dayu11/Availability-Attacks-Create-Shortcuts).
