import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def update_line_fit(line, lefty, leftx):
    """
        Run a polyfit on the last 10 batches of data that
        form the detected lines.
    """
    line.recent_xfitted.append(leftx)
    line.recent_yfitted.append(lefty)
    
    if len(line.recent_xfitted) > 10:
        line.recent_xfitted.pop(0)
        line.recent_yfitted.pop(0)

    x = np.concatenate(line.recent_xfitted)
    y = np.concatenate(line.recent_yfitted)

    fit = np.polyfit(y, x, 2)

    updated_fit = line.current_fit.update(fit)
    return updated_fit

def update_line_info(line,
                     detected,
                     x,
                     y,
                     fit,
                     fit_pts,
                     roc,
                     ):
    """
        update Left Line info
    """
    
    MAX_ITEMS = 10
    line.detected = detected
    
    line.diffs = fit - line.current_fit.val()
    line.fit_pts = fit_pts
    line.radius_of_curvature = roc
    line.allx = x
    line.ally = y
    print (line)

    return line


def get_distance_from_center(left_lane, right_lane):
    xm_per_pix = 3.7/1100 # meters per pixel in x dimension

    left_fit = left_lane.current_fit.val()
    right_fit = right_lane.current_fit.val()
    
    y = 719
    left_pt = left_fit[0]* y**2 + left_fit[1]* y + left_fit[2]
    right_pt = right_fit[0]* y**2 + right_fit[1]* y + right_fit[2]

    center = (left_pt + right_pt) / 2
    
    # Return offset from center of the lane
    # if offset < 0 -> car is on the left of the center
    # if offset > 0 -> car is on the right of the center
    offset = (1280/2 - center) * xm_per_pix
    
    return offset

def get_radius(left_line, right_line):
    """
        Radius of Curvature
    """
    
    left_line_fit = left_line.current_fit.val()
    right_line_fit = right_line.current_fit.val()
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/1100 # meters per pixel in x dimension
    
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    
    leftx = np.array([(y**2)*left_line_fit[0] + y*left_line_fit[1] + left_line_fit[2] for y in ploty])
    rightx = np.array([(y**2)*right_line_fit[0] + y*right_line_fit[1] + right_line_fit[2] for y in ploty])

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = 719
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    
    return left_curverad, right_curverad

def overlap_current_lane(binary_warped, lane):
    current_fit = lane.current_fit
    
    if current_fit.t > 0 and lane.detected:
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        fitx = (current_fit.a*ploty**2 + current_fit.b*ploty + current_fit.c).astype(int)
        
        print("Add previous line")
        
        for y,x in zip(ploty.astype(int),fitx):
            if x>=1280:
                continue
            
            binary_warped[y,x] = 1
            binary_warped[y,x-1] = 1
            binary_warped[y,x-2] = 1
        
    return binary_warped

def find_lanes(binary_warped, left_line, right_line, PLOT=False, index=0):
    print ("find_lanes()")
    binary_warped = overlap_current_lane(binary_warped, right_line)
    binary_warped = overlap_current_lane(binary_warped, left_line)

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    if PLOT:
        plt.figure()
        plt.plot(histogram)
        plt.savefig('./out/picture-'+str(index)+'-lane-histogram')

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255


    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint-100])
    rightx_base = np.argmax(histogram[midpoint+100:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 120
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
#        print ("leftx current rightx current", leftx_current, rightx_current)
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        if PLOT:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    if PLOT:
        cv2.imwrite('./out/picture-'+str(index)+'-out_img.png', out_img)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial and update the line data structure
    left_fit = update_line_fit(left_line, lefty, leftx)
    right_fit = update_line_fit(right_line, righty, rightx)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Get Radius of Curvature
    left_radius, right_radius = get_radius(left_line, right_line)

    # Get Distance from center:
    offset = get_distance_from_center(left_line, right_line)
    print ("Offset from center {:0.02f}m".format(offset))
    
    if PLOT:
#        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        plt.figure()
        plt.imshow(out_img)
        plt.text(0,-30, "left_radius={:0.2f}m".format(left_radius))
        plt.text(700,-30, "right_radius={:0.2f}m".format(right_radius))

        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('./out/picture-'+str(index)+'-lanes.png')


    left_line = update_line_info(left_line,
                 detected=True,
                 x = leftx,
                 y = lefty,
                 fit = left_fit,
                 fit_pts = left_fitx,
                 roc = left_radius,
                 )
                 
    right_line = update_line_info(right_line,
                 detected=True,
                 x = rightx,
                 y = righty,
                 fit = right_fit,
                 fit_pts = right_fitx,
                 roc = right_radius,
                 )
                 
    return ploty, left_line, right_line, offset

def process_next_image(binary_warped, ploty, left_line, right_line, index=0, PLOT=False):
    """
        Process a new image using the result already
        compiled from the function find_lanes()
        
        binary_warped: The warped image after change of perspective
        
    """
    print ("process_next_image()")
    
    left_fit = left_line.current_fit.val()
    right_fit = right_line.current_fit.val()
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    
    # Add previous fit line for robusteness
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    curr_left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    curr_right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    leftx = np.concatenate([leftx, curr_left_fitx])
    lefty = np.concatenate([lefty, ploty])

    rightx = np.concatenate([rightx, curr_right_fitx])
    righty = np.concatenate([righty, ploty])

    try:
        left_fit = update_line_fit(left_line, lefty, leftx)

    except:
        print ("Error:Can't fit (lefty,leftx)" )

        plt.figure()
        plt.scatter(leftx, lefty, s=1, c="y", marker="o")
        plt.scatter(rightx, righty, s=1, c="b", marker="+")
        plt.gca().invert_yaxis()
        plt.savefig('./out/picture-'+str(index)+'-scatter.png')

        left_line.detected = False
        return None, left_line, right_line, 0

    try:
        right_fit = update_line_fit(right_line, righty, rightx)

    except:
        print ("Error:Can't fit (righty, rightx)" )
        plt.figure()
        plt.scatter(leftx, lefty, s=1, c="y", marker="o")
        plt.scatter(rightx, righty, s=1, c="b", marker="+")
        plt.gca().invert_yaxis()
        plt.savefig('./out/picture-'+str(index)+'-scatter.png')

        right_line.detected = False
        print (right_line)
        return None, left_line, right_line, 0


    if PLOT:
        plt.figure()
        _leftx = np.concatenate(left_line.recent_xfitted)
        _lefty = np.concatenate(left_line.recent_yfitted)
        _rightx = np.concatenate(right_line.recent_xfitted)
        _righty = np.concatenate(right_line.recent_yfitted)

        plt.scatter(_leftx, _lefty, s=1, c="y", marker="o")
        plt.scatter(_rightx, _righty, s=1, c="b", marker="+")
        plt.gca().invert_yaxis()
        plt.savefig('./out/picture-'+str(index)+'-scatter.png')

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Get Radius of Curvature
    left_radius, right_radius = get_radius(left_line, right_line)

    # Get Distance from center:
    offset = get_distance_from_center(left_line, right_line)
    print ("Offset from center {:0.02f}m".format(offset))
    
    if PLOT:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        plt.figure()
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('./out/picture-'+str(index)+'-next_lane.png')

    left_line = update_line_info(left_line,
                             detected=True,
                             x = leftx,
                             y = lefty,
                             fit = left_fit,
                             fit_pts = left_fitx,
                             roc = left_radius,
                             )
    
    right_line = update_line_info(right_line,
                                  detected=True,
                                  x = rightx,
                                  y = righty,
                                  fit = right_fit,
                                  fit_pts = right_fitx,
                                  roc = right_radius,
                                  )

    return ploty, left_line, right_line, offset

if __name__ == '__main__':
    binary_warped = mpimg.imread('warped-example.jpg')

    left_fitx, right_fitx = find_lanes(binary_warped)

    next_binary_warped = mpimg.imread('warped-example.jpg')

    #process_next_image(binary_warped, left_fitx, right_fitx)

