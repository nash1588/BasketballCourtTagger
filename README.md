# BasketballCourtTagger
Efficient and easy GUI to tag and find homography transformations between courts from a basketball game to a 2d basketball court

In case you want to tag different (sports) courts, replace you can easily replace the given court images with yours.

<img src="example/main_screen1.JPG" width="900px">
<img src="example/tagged_screen1.JPG" width="1200px">


## Requirements

```bash
pip install pickle-mixin
pip install tkintertable
```

## Running
```bash
python ManualCourtEdgesTaggerGUI.py --games_images_path YOUR_PATH
```

Where YOUR_PATH is the directory where all the game frames are located.

Make sure you have all 3 images: nba_court_gray_bitmap.bmp, nba_court_phase1_bitmap.bmp and nba_court_white.jpg in YOUR_PATH where all the 

## Tagged data
The tagged courts will be inside a new directory in YOUR_PATH named tagged_data, 
each frame that has been tagged will have it's own directory consisting of dick.pickle file that has a dictonary with these keys:
"game_pts" - points tagged in the game frame
"court_pts" - points tagged in the 2d nba court
"game_res", "court_res" - resolution of the images when they were tagged on your machine
"h_mat" - homography matrix

In addition there are images I used for a generative adversarial network.
