# Traffic4cast 2020
by Team Deadlock: [Ye Hong](https://www.researchgate.net/profile/Ye_Hong9) and [Shengyu Huang]()

## Documentation

## Howto
### Train a model
- Copy the competition raw data into a folder and enter the root in 'util/create_mask.py' and 'Unet/config.py' files. 
- Run 'util/create_mask.py' to create a mask for each city
- Run 'Unet/training.py' or 'multiLSTM/training.py' to start training.
- Run `util/create_submission.py` to create submission files using the trained model

### Create submission with the pretrained models
Note that due to time restrictions we only train our proposed model on Moscow, the Berlin model is a simplified network with depth 5, and the Istanbul model is the original UNet as reported from [1].
- Download the pretrained model and copy the 'checkpoint.pt' file to the respective folder 'runs'
- open 'validate_Berlin.py', 'validate_Istanbul.py' or 'validate_Moscow.py', and fill in the 'source_root' and 'submission_root' placeholders.
- Run 'validate_Berlin.py', 'validate_Istanbul.py' or 'validate_Moscow.py'.




### References
[1] Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
