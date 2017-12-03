from moviepy.editor import VideoFileClip
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import pipelines
import gradient_threshold
import numpy as np
import lane_detection

class movingAvg():
    """
        An Implementation of exponential moving average for the
        line equation a x^2 + b x + x.
        
        Each coefficient a,b,c will be smoothly updated separately
        according to the beta smoothing factor.
    """

    def __init__(self, beta, name):
        self.a = 0
        self.b = 0
        self.c = 0
        
        self.t = 0
        self.beta_a = beta
        self.beta_b = beta
        self.beta_c = beta

        self.name = name
    
    def val(self):
        return np.asarray([self.a, self.b, self.c])

    def reset(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.t = 0

    def update(self, fit):
        """
            Update the line's coefficient with a new datapoint.
        """
        a = fit[0]
        b = fit[1]
        c = fit[2]

        if self.t == 0:
            self.a = a
            self.b = b
            self.c = c
            self.t = 1
            return np.asarray([self.a, self.b, self.c])
    
#        delta_a = ((a-self.a)/self.a)
#        delta_b = ((b-self.b)/self.b)
#        delta_c = ((c-self.c)/self.c)

        self.t +=1

        self.a = (self.beta_a * self.a + (1 - self.beta_a) * a )
        self.b = (self.beta_b * self.b + (1 - self.beta_b) * b )
        self.c = (self.beta_c * self.c + (1 - self.beta_c) * c )
        
        return np.asarray([self.a, self.b, self.c])


#        # Large changes to the lane's equation are not expected.
#        if delta_a > 10 or delta_b > 10 or delta_c > 10:
#            print ("a={}->{}, b={}->{}, c={}->{}".format(self.a,a,self.b, b,self.c,c))
#            print ("{} lane, Very Large Change, disregard - delta_a:{} delta_b:{} delta_c:{}".format(self.name, delta_a, delta_b, delta_c))
#            return np.asarray([self.a, self.b, self.c])
#        elif delta_a > 1 or delta_b > 1 or delta_c > 1:
#            print ("Large {} Line Change - delta_a:{} delta_b:{} delta_c:{}".format(self.name, delta_a, delta_b, delta_c))
#            beta_a = beta_b = beta_c = 0.99
#
#            self.a = (beta_a * self.a + (1 - beta_a) * a )
#            self.b = (beta_b * self.b + (1 - beta_b) * b )
#            self.c = (beta_c * self.c + (1 - beta_c) * c )
#
#            return np.asarray([self.a, self.b, self.c])
#        else:
#            self.a = (self.beta_a * self.a + (1 - self.beta_a) * a )
#            self.b = (self.beta_b * self.b + (1 - self.beta_b) * b )
#            self.c = (self.beta_c * self.c + (1 - self.beta_c) * c )
#            return np.asarray([self.a, self.b, self.c])


class Line():
    """
        Maintain state information for each line
    """
    def __init__(self, name):
        self.name = name
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # x values of the last n fits of the line
        self.recent_yfitted = []
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = movingAvg(beta = 0.5, name=name)
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def __repr__(self):
        print ("~~~ Line Info: {} ~~~".format(self.name))
        print ("detected {}".format(self.detected ))
        print ("radius_of_curvature {:0.2f}m".format(self.radius_of_curvature))
        print ("current_fit (A.x^2+B.x+C) A={:0.6f}, B={:0.6f}, C={:0.6f}".format(self.current_fit.a,
                                                                                  self.current_fit.b,
                                                                                  self.current_fit.c))
        
        return ""


index = 0
ploty = None

def process_image(image):
    global ploty
    global left_line
    global right_line
    global TEST
    global index
    index += 1

    h, w, c = image.shape
    
    PLOT = True
#    if index < 110:
#        return image

    if not TEST:
        # For testing purpose, don't make this transformation
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        PLOT = True

    print ("~~~ Process Image #~~~", index)
    
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    undist_rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)

    if PLOT:
        plt.figure()
        plt.imshow(undist_rgb)
        plt.savefig('./out/picture-'+str(index)+'-undistorted.png')


    """
        Run the Thresholding algorithm
    """
    color_binary, combined_binary = pipelines.pipeline(undist_rgb)

    # Plot the result
    if PLOT:
        plt.figure()
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        f.suptitle("Color and Gradient Thresholding Results", fontsize=30)

        ax1.imshow(undist_rgb)
        ax1.set_title('Undistorted Image', fontsize=20)

#        ax2.imshow(color_binary)
#        ax2.set_title('Color Binary', fontsize=40)
#        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        ax2.imshow(combined_binary, cmap="gray")
        ax2.set_title('Pipeline Output', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.savefig('./out/picture-'+str(index)+'-pipeline-out.png')

    """
        Generate Perspective
    """
#    src = np.float32([ [600, 470], [800, 470], [1150, 720],  [200, 720]])
    src = np.float32([ [550, 470], [800, 470], [1150, 710],  [180, 710]])

#    dst = np.float32([[0, 0],[1280, 0], [1280, 720], [0, 720]])
    dst = np.float32([[0, 0],[w, 0], [w, h], [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    img_size = combined_binary.shape[1], combined_binary.shape[0]
    warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

    if PLOT:
        plt.figure()
        plt.imshow(warped)
        plt.savefig('./out/picture-'+str(index)+'-warped-out.png')

    """
        Detect Lanes
    """
    if ploty is None:
        ploty, left_line, right_line, offset = lane_detection.find_lanes(warped, index=index, left_line=left_line, right_line=right_line, PLOT=PLOT)
    else:
        ploty, left_line, right_line, offset = lane_detection.process_next_image(warped, ploty, left_line, right_line, index=index, PLOT=PLOT)

        # Sanity Checks
        if not left_line.detected or not right_line.detected:

            print ("Sanity Check Fail", left_line.detected, left_line.detected, left_line.radius_of_curvature, right_line.radius_of_curvature)
            
            ploty, left_line, right_line, offset = lane_detection.find_lanes(warped, index=index, left_line=left_line, right_line=right_line, PLOT=True)


    """
        Draw lane on the image
    """
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fitx = left_line.fit_pts
    right_fitx = right_line.fit_pts

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist_rgb, 1, newwarp, 0.3, 0)

    cv2.putText(result,
                "Radius of Curvature={:0.2f}m".format(left_line.radius_of_curvature),
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(255,255,255),
                thickness=3,
                lineType=8)
                
#    cv2.putText(result,
#            "right_roc={:0.2f}m".format(right_line.radius_of_curvature),
#            (900, 600),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            fontScale=1,
#            color=(0,0,255),
#            thickness=3,
#            lineType=8)

    if offset > 0:
        off="left"
    else:
        off="right"

    cv2.putText(result,
            "Vehicle is {:0.2f}m {} of center".format(abs(offset), off),
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255,255,255),
            thickness=3,
            lineType=8)
#    plt.text(700,-30, "right_radius={:0.2f}m".format(right_radius))

    if True:
        plt.figure()
        plt.imshow(result)
        plt.savefig('./out/picture-'+str(index)+'-final-out.png')

    return result

FILE = "../videos/project_video.mp4"
FILE_OUT = "./out/project_video.mp4"

dist_pickle = pickle.load( open('./wide_dist_pickle.p', 'rb'))

mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

left_line = Line("LEFT")
right_line = Line("RIGHT")

TEST = False
if TEST:
    image = cv2.imread('./frames/picture-370.png')
    process_image(image)
    image = cv2.imread('./frames/picture-371.png')
    process_image(image)

else:
    clip = VideoFileClip(FILE)
    white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(FILE_OUT, audio=False)



