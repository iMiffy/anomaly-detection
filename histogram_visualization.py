import cv2
from matplotlib import pyplot as plt
import argparse 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(ap.parse_args())


img = cv2.imread(args['input'])
colors = ('b', 'g', 'r')
chans = cv2.split(img)

plt.figure()
plt.title('Color Histogram')
plt.ylabel('% of pixels')
plt.xlabel('Bins')

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0,256])
    hist /= hist.sum()
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
    
plt.legend(colors)
plt.savefig(args['output'])