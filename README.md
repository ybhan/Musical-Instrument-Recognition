### Musical Instrument Recognition in Isolated Notes and Solo Phrases ###

By Yuanbo Han and Wenyu Zhang, November 2018.

#### Methodology ####
- Gammatone Filtering
- CNN
	![CNN Architecture](cnn/CNN Architecture.png)

#### Dataset ####
[RWC Music Database: Musical Instrument Sound](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html)

#### Environment ####
- MATLAB R2018a
- Python 3.6
	- scipy 1.1
	- numpy 1.14
	- torch 1.0

Note that these are only tested versions, not the least requirements.

#### Usage ####
Run `gammatone/main.m` to perform Gammatone filtering, but please first check the dataset path in line 5 and line 8. To get a grasp of the dependent functions, see `gammatone_demo.m` for a simple example.

After preprocessing, run `cnn/read_data.py` to read .MAT files as .NPY format. Then execute `CNN.py`to train the neural network. There is an instance (`mat_files/XY.mat`) for testing `CNN.py` without above steps.
