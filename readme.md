# Note: [an article](https://medium.com/@evankolberg/dominating-a-skill-based-casino-game-with-deep-learning-83dee3c7fd78) published on Medium.com enhanced this readme


# Why was this made?

This project was made in hopes of earning money from Fanduel Faceoff. Honestly, that is the truth. I didn't create this with the goal of advancing my "python proficiency" or whatever fahdoodle, gurry, bull durham, bushwa, bottlewash, donkey dust, flubdub, flapdoodle, flummadiddle, flabberdegaz, bologna scraps you think could fit here. I heard about this app from a friend and decided to take a look. When I glaced at the available games, I saw that Boggle was an option. Immediately, my mind thought about all the ways I could automate this. From that day, I worked around 8-10 hours a day for nearly 2 weeks until it was finished. It was a rough journey. I had to scrap tons of ideas and files multiple times. I wanted to use my skills to actually make money!

The Journey:

OCR Method:
I attempted to use Optical Character Recognition in order to identify the letters on the board. I used models from pytesseract, paddleOCR, easyOCR, and even Apple's native OCR feature baked into macOS. OCR didn't work too well. It struggled with understanding individual characters in a non-standard font. 

Fully hard coded method:
I collected all the standard game pieces (A-Z) and tried to perform morphological procedures to them in order to compare their similarity. I tried Mean Square Error, overlap ratios, HOG descriptors, etc. I peaked when I had around 50-80% accuracy per board. Clearly, this wasn't enough. Not good enough for me. I felt like these methods weren't adaptive enought. They felt underkill.

Object Detection Method:
First, I collected 100 game board. Then, I spent DAYS and DAYS annotating them. I also spent DAYS and DAYS trying to build Opencv myself until I realized that I could use a pre-built wheel that fit my GPU's specs. Spent a little while working through some Pytorch issues (which were really my fault, not torch's) and trained the model. Yay! It was crap! Complete garbage! It thought everything was the letter J. It did not work at all. I was about to annotate another dataset I collected, but stopped to reasses because annotating another 100 images would take another 3-4 days. I already have the structure of the game pieces, why did I need object detection? Truthfull, I didn't. This method is comletely overkill and yieled a lot of overhead, wasting resources and time.

CLIP Method:
I invested a lot of time and energy by this point. I was ready to give up. But, that's just not what I do. So, I went running for answers. When I was just randomly surfing online, I found [this article](https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3#:~:text=Structural%20Similarity%20Index%20(SSIM),Python%20offers%20an%20SSIM%20implementation.). This article was my savior. It described all the popular ways image similarity is done in python, and at the bottom there was a deep learning based approach. Since I had already collected all the A-Z tiles, I gave it shot. I used what I learned from my other attemps, including the need for pre-processesing of the images before they are compared. To my surprise, it was able to accuratly identify the tiles each time with 100% accuracy! The target letter was always over 90% similar and the others were far below 80%, showing how robust this method was. Using color quantization, I was able to accuratly identify the special tiles.

All while working on the identification methods, I had found a boggle solving library made in python and tailored it to fit my specific needs. I also made computation more efficient by introducing a trie utilized when going through the dictionary of words. 




# What does it do?

When the script is run, it loads a vision transformer contrastive language-image pre-training model into memory, then prompts the user to move their mouse to the top left and bottom right points of a certain region on the screen. At each point, press Enter to record the point. What region of the screen? Great question! If you have an android phone, you can use scrcpy to stream and control your phone, and if not, then you can use Bluestacks emulator. Run the Fanduel Faceoff application and start a boggle game. The top left point should be at the top left of the tile box, and the same idea for the bottom right. Wait to place that second point until after the letter pieces have dropped in the board and have stopped shaking. Once the second point is recorded, then the rectangular region between the two points will be captured. Then, it goes by itself! It will identify each letter on the board, recognize each bonus tile, calculate the possible words, and then the highest scoring words. It will then enter the best words using your mouse. In the code, you can turn glide to True if you want the mouse to follow a Catmull Rom spline to mimic a more human-like path. The mouse dragging movements will be simulated as finger dragging on your phone. 

It was hard, but it actually works. I'm proud of myself.

![real_android](https://github.com/evan-kolberg/fanduel_faceoff_boggle_ML/blob/main/proof/20240620_023628000_iOS.png?raw=true)
![it_running](https://github.com/evan-kolberg/fanduel_faceoff_boggle_ML/blob/main/proof/20240620_030933000_iOS.png?raw=true)
![using_bluestacks](https://github.com/evan-kolberg/fanduel_faceoff_boggle_ML/blob/main/proof/Screenshot%202024-06-18%20125100.png?raw=true)
![high_score](https://github.com/evan-kolberg/fanduel_faceoff_boggle_ML/blob/main/proof/Screenshot%202024-06-18%20143659.png?raw=true)

Videos:
- https://1drv.ms/u/s!AmLvGBmIAALl_j0rwvSKmn85bb3E?e=gloGHJ
- https://1drv.ms/v/s!AmLvGBmIAALl_j9Jm4VVKOTMIvnr?e=8u0qRO
- https://1drv.ms/v/s!AmLvGBmIAALl_j6gcCZ2WCeX0jPg?e=eaPyq5
- https://1drv.ms/v/s!AmLvGBmIAALl_kC4dN-yf_HkrhEm?e=McAm4a


# How to run this

## Step 1: Prerequisites

*** in this repo, main.py was made to work with a windows machine ***

Install [Anaconda](https://www.anaconda.com/download/success) if you don't have it already

Install CUDA 12.1 (for pytorch) if you have a fairly modern Nvidia GPU (I used a 1660 Ti)

Create a conda environment using: `conda create -n envname python=x.x anaconda`

Then, install pytorch (with CUDA support) with:
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

If you don't have an Nvidia card or a supported GPU, then refer to
[this link](https://pytorch.org/) to find out how to install pytorch with your specific requirements

Lastly, pip install the remaining dependencies from main.py

If you have an android phone >= android OS 5, then you can stream and control your phone with your PC. Refer to [here](https://github.com/Genymobile/scrcpy/blob/master/doc/windows.md) for the installation instructions. If you don't have one, then you can use Bluestacks to emulate an android phone. The one caveat is that you won't be able to play games with monetary entry fees. It needs to verify your location. The location spoofer in Bluestacks doesn't work- I've tried, trust me. The FREE entry game works, but you can't earn any prizes. The maximum account money I could accumulate just by playing the FREE entry head-to-head games was $1.60. Reguardless, download the Fanduel Faceoff .apk [here](https://www.fanduel.com/android). 


# Step 2:

Run main.py in the root directory. Pray it works!
