# crfill
code for paper

## Usage
1. Install dependencies:
```
conda env create -f environment.yml
```
or manually install these packages in a Python 3.6 enviroment: 

```pytorch=1.3.1```, ```opencv=3.4.2```, ```tqdm```

2. Use the code:

with GPU:
```
python test.py --image path/to/images --mask path/to/hole/masks --output path/to/save/results
```
without GPU:
```
python test.py --image path/to/images --mask path/to/hole/masks --output path/to/save/results --nogpu
```
```path/to/images``` is the path to the folder of input images; ```path/to/masks``` is the path to the folder of hole masks; ```path/to/save/results``` is where the results will be saved. 

Hole masks are grayscale images where pixel values> 0 indicates the pixel at the corresponding position is missing and will be replaced with generated new content. 

:mega: :mega: The white area of a hole mask should fully cover all pixels in the missing regions. :mega: :mega:
