import cv2
import numpy as np

cursor_template = cv2.imread("cursor_temp.jpg", 0)
# Compare the template patch with pos and neg images
test_images = cv2.imread("pos_1.jpg", 0)



w, h = cursor_template.shape[::-1]

# Applying Gaussian blur and lapacian transform for the template
cursor_blur = cv2.GaussianBlur(cursor_template, (3, 3), sigmaX=0)
cursor_laplacian = cv2.Laplacian(cursor_blur, cv2.CV_8U)

# Applying Gaussian blur and laplacian transform for the test image
image_gaussian_blur = cv2.GaussianBlur(test_images, (5, 5), sigmaX=0)
image_laplacian = cv2.Laplacian(image_gaussian_blur, cv2.CV_8U)

matching_cursor = cv2.matchTemplate(image_laplacian, cursor_laplacian, cv2.TM_CCOEFF_NORMED)

location = np.where(matching_cursor >= 0.67)


for cursor in zip(*location[::-1]):
    test_images = cv2.rectangle(test_images, cursor, (cursor[0] + w, cursor[1] + h), (255, 0, 0), 2)


cv2.namedWindow('new_window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('new_window', 720, 600)
cv2.imshow('new_window', test_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
