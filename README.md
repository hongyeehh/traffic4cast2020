# Traffic4cast 2020
by Team Deadlock: [Ye Hong](https://www.researchgate.net/profile/Ye_Hong9) and [Shengyu Huang]()

## Documentation

## Howto
### Train a model
- Copy the competition raw data into a folder and enter the root in 'util/create_mask.py' and 'Unet/config.py' files. 
- Run 'util/create_mask.py' to create a mask for each city
- Run 'Unet/training.py' or 'multiLSTM/training.py' to start training.
- Run `util/create_submission.py` to create submission files using the trained model

