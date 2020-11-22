####Install

`git clone --recurse-submodules https://github.com/AlexandrGrents/determining-vehicle-speed.git`

`pip install -r requirements.txt`

`pip install -r ./sort/requirements.txt`

`python -m pip install -e detectron2`

####Run

`python run.py`

**Options:**

`--video-file`: path to the requested .mp4 video. Default: input.mp4

`--save-to`: path where to save results. Default: output.mp4

`--image-mask`: path to the image mask. Default: mask.png