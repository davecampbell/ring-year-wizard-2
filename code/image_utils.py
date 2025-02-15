import cv2
import math
import numpy as np

def masked_ellipse(img, ellipse):
  mask = np.zeros(img.shape[:2], dtype=np.uint8)
  cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
  masked_image = cv2.bitwise_and(img, img, mask=mask)
  return masked_image

def align_ellipse_vertically(gray):
    
    edge_method = "thresh"

    if edge_method == "canny":
      edge_img = cv2.Canny(gray, 50, 150)
    elif edge_method == "sobel":
      edge_img = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    elif edge_method == "thresh":
      _, edge_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.info("No contours found!")
        return None, image

    use_contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse
    if len(use_contour) >= 5:  # cv2.fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(use_contour)
        (x, y), (width, height), angle = ellipse

        good_ellipse = True
        # check ratio major:minor
        if max([width, height]) / min([width, height]) > 4.2:
          good_ellipse = False
          logger.info("BAD ELLIPSE - too skinny")

        if width / gray.shape[0] > 1.2 or height / gray.shape[0] > 1.2:
          good_ellipse = False
          logger.info("BAD ELLIPSE - one dimension out of frame > 20%")

        if good_ellipse:

          cv2.ellipse(gray, ellipse, (0, 255, 0), 2)

          # mask that ellipse on image
          e_image = masked_ellipse(gray, ellipse)

          # affine transform directly using triangle translation
          H, W = e_image.shape[:2]
          (x, y), (w, h), angle = ellipse
          if angle > 90:
            angle -= 180
          M = int(max(h,w))
          m = int(min(h,w))
          scale = H / M

          srcTri = np.array([[x,y],
                  [x+math.sin(math.radians(angle))*M/2, y-math.cos(math.radians(angle))*M/2],
                  [x - math.cos(math.radians(angle)) * m/2, y - math.sin(math.radians(angle)) * m/2]]
                  ).astype(np.float32)
        
          srcA = int(srcTri[0][0]), int(srcTri[0][1])
          srcB = int(srcTri[1][0]), int(srcTri[1][1])
          srcC = int(srcTri[2][0]), int(srcTri[2][1])

          dstTri = np.array([[W//2, H//2],
                          [W//2, 0],
                          [W//2 - m/2 * scale, H//2]]).astype(np.float32)

          Mat = cv2.getAffineTransform(srcTri, dstTri)

          trans_image = cv2.warpAffine(e_image, Mat, (W, H))
          
          return ellipse, trans_image
        else:
          return None, image
    else:
      return None, gray

    logger.info("No valid ellipse found!")
    return None, gray

def ellipsify(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, aligned_image =  align_ellipse_vertically(gray)

  return aligned_image

def flip_vertically(image):
  return cv2.flip(image, -1)

