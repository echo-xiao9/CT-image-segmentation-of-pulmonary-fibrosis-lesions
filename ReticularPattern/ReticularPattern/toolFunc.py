#!/usr/bin/env python

import cv2 as cv
import numpy as np
 
def reverseImg(image):
    height, width, channels = image.shape
    print("width:%s,height:%s,channels:%s" % (width, height, channels))
 
    for row in range(height):
        for list in range(width):
            for c in range(channels):
                pv = image[row, list, c]
                image[row, list, c] = 255 - pv
    cv.imshow("AfterDeal", image)

def areaFilter(img):
  # # 加载图片
  # img = cv.imread(path)
  # # 灰度化
  # # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # # 二值化
  # ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
  # cv.imshow('thresh',thresh)
  cv.imshow('inputImg',img)

  # 寻找轮廓
  contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  drawing = np.zeros(img.shape, np.uint8)
  # 计算平均面积
  areas = list()
  for i in range(len(contours)):
      area = cv.contourArea(contours[i], False)
      areas.append(area)
      # print("轮廓%d的面积:%d" % (i, area))

  area_avg = np.average(areas)
  print("轮廓平均面积:", area_avg)
  area_median = np.median(areas)
  print("轮廓中位数面积:", area_median)

  # 筛选超过平均面积的轮廓
  # img_contours 

  for i in range(len(contours)):
      # img_temp = np.zeros(img.shape, np.uint8)
      # img_contours.append(img_temp)

      area = cv.contourArea(contours[i], False)
      if area <  800 and area > area_median/10:
        print("轮廓%d的面积是: %d" % (i, area))
        cv.drawContours(drawing, contours, i,  (255,255,255) , thickness=-1)
      # cv.imshow("contours %d" % i, img_contours[i])
  cv.imshow("contours ", drawing)
  return drawing

