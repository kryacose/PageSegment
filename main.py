from pdf2image import convert_from_path
import lineSegment as lsg
import wordSegment as wsg
import cv2
import os
import numpy as np

# Get the document and specify output directory
_input = os.path.join('data', 'NOFORM_1.pdf')
output = os.path.join(os.getcwd(), 'output1')
if not os.path.exists(output):
	os.mkdir(output)

# Convert document to jpeg
pages = convert_from_path(_input, 200)
pages[0].save(os.path.join(output, 'original.jpg'),'JPEG')


img = cv2.imread(os.path.join(output, 'original.jpg'))
# cv2.imshow('i', img)
# cv2.waitKey(0)
#-------------------------------------------------------------------------------
# SEGMENT LINES OF TEXT
# lineSegment parameters:
# mode 0: only free lines,    mode 1: only straight lines,   mode 2: Both
# saveRes: Write result to file
res = lsg.lineSegment(img, out = output, mode = 0, saveRes = True)
#-------------------------------------------------------------------------------
# SEGMENT WORDS FROM LINES
output = os.path.join(output, 'words')
if not os.path.exists(output):
	os.mkdir(output)
# read image, prepare it by resizing it to fixed height
# execute segmentation with given parameters
# -kernelSize: size of filter kernel (odd integer)
# -sigma: standard deviation of Gaussian function used for filter kernel
# -theta: approximated width/height ratio of words, filter function is distorted by this factor
# - minArea: ignore word candidates smaller than specified area

i=0
words = 0
for pic in res[0]:
	pic = wsg.prepareImg(pic, 50)
	Res = wsg.wordSegmentation(pic, kernelSize=25, sigma=11, theta=7, minArea=100)
	for (j, w) in enumerate(Res):
		words+=1
		(wordBox, wordImg) = w 
		(x, y, w, h) = wordBox
		cv2.imwrite(os.path.join(output,str(i)+'_'+str(j)+'.png'), wordImg) # save word
	i+=1

print("LINES DETECTED: " + str(len(res[0])))
print("WORDS DETECTED: " + str(words))