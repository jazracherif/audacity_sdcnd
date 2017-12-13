import matplotlib
import matplotlib.image as mpimg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.externals import joblib
from feature_extraction_utilities import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from keras.models import load_model



def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat=False, hist_feat=False, cells_per_step=2, name=""):
    """
        This function generates looks for cars in the current frame.
        1) Calculare the HOG feature for the whole frame
        2) Break down HOG Features into grid representing a 64x64 image and aply the SVM classfied
        3) Also apply a trainined ConvNet on the raw 64x64 pixels image
        4) Pick those windows which are chosen by the SVM classifier and the NN classifier with high confidence.
    """
    
    draw_img = np.copy(img)
#    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
#    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
#    mpimg.imsave('./out/'+name+'-scale-'+str(scale)+'-cells-'+str(cells_per_step)+'-ctrans.jpg', ctrans_tosearch)

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps for overlapping blocks.
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False,feature_vec=False)

    # For each patch, record the box bounds and the input feature,
    # and then do the prediciton in block
    bbox_list = []
    box_nn = []
    box_svm = []
    box_info = []
    
    num_box = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            num_box +=1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the corresponding raw image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
#            spatial_features = bin_spatial(subimg, size=spatial_size)

            # Get color histogram features
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Get the SVM input features
            svm_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))
            box_svm.append(svm_features)

            #svm_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
#            svm_prediction = svc.predict_proba(svm_features)
#            svm_label = np.argmax(svm_prediction)
#            svm_prob = svm_prediction[0][svm_label]

            # Get the input for the NN classifier
            subimg_rgb_unscaled = cv2.cvtColor(subimg, cv2.COLOR_HLS2RGB)
            subimg_rgb = np.copy(subimg_rgb_unscaled) / 255
            subimg_rgb = subimg_rgb.reshape(1,64,64,3).astype('float32')
            box_nn.append(subimg_rgb)

#            nn_prediction = nn_vehicle_model.predict(subimg_rgb)
#            nn_label = np.argmax(nn_prediction)
#            nn_prob = nn_prediction[0][nn_label]

            # Save the current box bounds
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            box_bound = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))
            box_info.append(box_bound)


    ## Do the SVM and NN predictions in block

#    start = time.time()
    nn_prediction = nn_vehicle_model.predict(np.vstack(box_nn))
#    end = time.time()
#    print(round(end-start, 2), 'Seconds to predict NN')

    nn_label = np.argmax(nn_prediction, axis=1)
    nn_prob = nn_prediction[np.arange(len(nn_prediction)), list(nn_label)]

#    start = time.time()
    no_svm_proba = True
    if no_svm_proba:
        svm_prediction = svc.predict(np.vstack(box_svm))
    #    end = time.time()
    #    print(round(end-start, 2), 'Seconds to predict SVM')
        good_boxes = (np.asarray(box_info)[((nn_label==1) & (nn_prob>0.8)) | (svm_prediction==1)]).tolist()
    else:
        svm_prediction = svc.predict_proba(np.vstack(box_svm))
    #    end = time.time()
    #    print(round(end-start, 2), 'Seconds to predict SVM')

        svm_label = np.argmax(svm_prediction, axis=1)
        svm_prob = svm_prediction[np.arange(len(svm_prediction)), list(svm_label)]

        good_boxes = (np.asarray(box_info)[((nn_label==1) & (nn_prob>0.8)) | ((svm_label==1) & (svm_prob>0.8))]).tolist()

    if DEBUG:
        print ("scale: {} - #Box: {} - tried:{}".format(scale, len(good_boxes), num_box))

    return good_boxes

###### False Positive Detection
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars and draw the boxes
    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
#        print ("Draw", bbox[0], bbox[1])
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    
    # Return the image
    return img

HEATMAP_QUEUE_SIZE = 20
threholds = np.arange(HEATMAP_QUEUE_SIZE)
def filter_bbox(name, image, box_list, PLOT=True):
    """
        - Maintain a cache of the last HEATMAP_QUEUE_SIZE heatmaps detected
        - Apply a filtering to remove false positive
        - clop the heatmap to be between 0 and 255
    """
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, box_list)

    last_n_heaps.append(heat)
    heatmap_size = len(last_n_heaps)
    if ( heatmap_size >= HEATMAP_QUEUE_SIZE-1):
        last_n_heaps.pop(0)

    # Combine Heaps from previous
    for h in last_n_heaps:
        heat = np.add(heat, h)

    # Apply threshold to help remove false positives
    
#    heat = apply_threshold(heat, threholds[heatmap_size])
    heat = apply_threshold(heat, 30)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)


    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    print ("#Cars Detected", labels[1])

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    if PLOT:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar()

        plt.title('Heat Map')
        fig.tight_layout()

        plt.savefig('./out/'+name+'-out2.jpg')

    return draw_img

### PARAMETERS.

index = 0
scale = 1.5

def process_image(image):
    global index
    index +=1
    name = "image"+str(index)

#    if index < 1018:
#        return image
    print ("process Image", name)
    start = time.time()


    box_list = find_cars(img=image, ystart=370, ystop=600, scale=1.25, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=2,name=name)

    box_list += find_cars(img=image, ystart=370, ystop=500, scale=0.85, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=3,name=name)

#    box_list += find_cars(img=image, ystart=400, ystop=600, scale=1., svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=5,name=name)
#
    box_list += find_cars(img=image, ystart=400, ystop=600, scale=1.5, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=3,name=name)
#
    box_list += find_cars(img=image, ystart=390, ystop=650, scale=2.25, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=2,name=name)

#
#    box_list += find_cars(img=image, ystart=300, ystop=500, scale=1.8, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=7, name=name)

#    box_list += find_cars(img=image, ystart=300, ystop=700, scale=1.8, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=13, name=name)
#
#    box_list += find_cars(img=image, ystart=300, ystop=700, scale=1.8, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, cells_per_step=19, name=name)

    

    draw_img = filter_bbox(name, image, box_list, PLOT=False)

    end = time.time()
    print(round(end-start, 2), 'Seconds to process 1 frame')

    return draw_img


### INIT PARAMETERS

# Video Location
#FILE = '../test_video.mp4'
#FILE_OUT = './out/test_video_out.mp4'
FILE = '../project_video.mp4'
FILE_OUT = './out/project_video_out.mp4'
TEST = False   # Whether to run the logic on  ./test_images/*.jpg

color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [370, 720] # Min and max in y to search in slide_window()

# Load model and scaler
svc = joblib.load('./vehicle_classifier.pkl')
X_scaler = joblib.load('./vehicle_classifier_scaler.pkl')
nn_vehicle_model = load_model('keras_cifar10_trained_model.h5')

print ("model", svc)

# Keep a stack of the last n heaps
last_n_heaps = []


##### MAIN LOOP BELOW

if TEST:
    test_image_file = glob.glob('../test_images/*.jpg')
    for f in test_image_file:
        name = f.split('/')[-1].split('.jpg')[0]
        print ("file:",name)

        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        process_image(image)
else:
    clip = VideoFileClip(FILE)
    processed_clip = clip.fl_image(process_image)
    processed_clip.write_videofile(FILE_OUT, audio=False)

