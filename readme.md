# How to run this

## Step 1: Prerequisites

CUDA 12.1 (for pytorch)

install pytorch with cuda support

install some caffeine into your body, cuz ur gunna need it

prepare for headaches and errors to no avail

(not needed, but possible: opencv with gpu support
CUDA 12.5, cuDNN 9.1.1, Nvidia Video Codec SDK 12.2
Note: take a look at the opencv-python-cuda-wheels submodule for 
some releases that might fit your machine - don't install in
the same env as pytorch)


Beware list:
- don't use OS's native screen capture - it injects metadata we don't need, aspect ratio won't be 1:1. This is why we use opencv to get data - it doesn't add ICC color profiles by default. This is good for greyscale image collection. Data collection for a game window should be pretty consistant. I heard that having some background randomness is good for it, which is a plus in this situation. if you convert images to greyscale using opencv but they were taken in color, then you might get a warning from torch saying that greyscale images aren't allowed to have an ICC color profile.
- building opencv with CUDA and cuDNN is hell. It's actual hell. I've done it before, I did it again, and again, and again. Not because I had to, but because I wanted to. Everything is tedius and the slightest misake can cause you to rebuild - which take hours sometimes. Messing with cmake is gross. Thank god a user on github showed me that we can use pre-built wheel files. The ones he made match my software and system specs, which was perfect. It works, but don't install torch with cuda in the same env. 
- Make sure to use conda environments. Before, I was just doing the python -m venv venv technique and I wasn't able to switch versions. Plus it didn't come with some library directory that I needed. Just use conda, it makes life easier.
- i messed up with a whole dataset. it was crap. i spent days labeling 100 images with in total, around 1900 individual annotations. I took screenshots on mac but each screenshot had a completely random size and aspect ratio. yeah, they were all resized to 1024x1024, but the content in them was stretched in weird ways. don't do this. just use opencv to get images with the correct aspect ratio so you don't have to worry about this augmentation. sometimes augmentation is good, but we don't want that much when we only have 100 images. also, the game that is running isn't going to be augmented, so there's not really a point to teaching it how to detect with this level of robustness.
- if torch is unable to locate your labels because it thinks that they're background files, then your file names are messed up. learned that mistake. took screenshots on mac and they got weird space characters in them. spent a while tryna figure this one out.



