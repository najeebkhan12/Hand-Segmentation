

import thread
from pykinect import nui
import numpy as np
import cv2
import imutils
import pygame
import scipy.misc
DEPTH_WINSIZE = 640, 480

screen_lock = thread.allocate()
tmp_s = pygame.Surface(DEPTH_WINSIZE, 0, 16)


def depth_frame_ready(frame):
  with screen_lock:
    frame.image.copy_bits(tmp_s._pixels_address)
    arr2d = (pygame.surfarray.array2d(tmp_s) >> 7 & 255)
    new_image = arr2d.astype(np.uint8)

    u = (new_image[new_image>0].min()+10)
    l = (new_image[new_image>0].min()+0)

    th1 = cv2.inRange(new_image, l , u)
    rotated = cv2.GaussianBlur(th1, (5,5), 0)
    rotated = imutils.rotate_bound(th1, 90)
    img = rotated
    _, contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    #print contours
    for cnt in contours:
    # get convex hull
        if cv2.contourArea(cnt) < 100:
            continue

        hull = cv2.convexHull(cnt, returnPoints = False)

        defects = cv2.convexityDefects(cnt, hull)
        if defects is None:
            continue
        moments = cv2.moments(cnt)
        centre = ((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        cv2.circle(img, centre, 3, (0, 0, 0), -1)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            cv2.line(img,start,end,[255,100,0],2)
            cv2.circle(img,far,5,[100, 255, 204],-1)
            if far[1] > centre[1]:
                continue
            cv2.line(img, centre, far, [120,0,0],2)

    cv2.imshow('KINECT Hand Segmentation', img)



kinect = nui.Runtime()
def main():
        kinect.depth_frame_ready += depth_frame_ready
        kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)
        cv2.namedWindow('KINECT Hand Segmentation', cv2.WINDOW_AUTOSIZE)


        while True:
            key = cv2.waitKey(1)
            if key == 27: break

        kinect.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()