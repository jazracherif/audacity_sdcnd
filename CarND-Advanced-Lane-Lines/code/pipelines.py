import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import gradient_threshold

def pipeline(img, s_thresh=(0.5, 1), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Use a combination of thresholds
    sxbinary = gradient_threshold.combined_thresh(img)
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    l_binary = np.zeros_like(s_channel)
    l_binary[(l_channel >= 160) & (l_channel <= 255)] = 1

    yellow_binary = np.zeros_like(s_channel)
    yellow_binary[(h_channel >= 19) & (h_channel <= 19)] = 1
    
    # Stack each channel
    color_binary = np.uint8(np.dstack((np.zeros_like(sxbinary),  sxbinary, s_binary))) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    sxbinary_right = sxbinary.copy()
    sxbinary_right[:,:640] = 0
    
    combined_binary[ (yellow_binary==1) | ( (l_binary==1) & (sxbinary_right==1))] = 1

    return color_binary, combined_binary


if __name__ == '__main__':
    image = mpimg.imread('signs_vehicles_xygrad.png')

    color_binary, combined_binary = pipeline(image)

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(color_binary)
    ax2.set_title('Color Binary', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    ax3.imshow(combined_binary, cmap="gray")
    ax3.set_title('Combined', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.savefig("out_color_thresh.png")
