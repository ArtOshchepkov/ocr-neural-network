import functools

import numpy as np

from main import y_to_letter, train_network, load_training_samples

import cv2

TARGET_WIDTH = 16
TARGET_HEIGHT = 16


def for_each_image_segmentation(im_path, model):
    image = cv2.imread(im_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 15)

    _, labels = cv2.connectedComponents(thresh)
    cv2.imshow('Thresh', thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 2900  # heuristic param
    upper = total_pixels // 150  # heuristic param

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if lower < numPixels < upper:
            mask = cv2.add(mask, labelMask)
    print('Masks', mask)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    print('BBOXES', boundingBoxes)
    for rect in boundingBoxes:
        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = mask[y:y + h, x:x + w]
        crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        # Prepare data for prediction
        crop = crop.astype("float") / 255.0
        # crop = img_to_array(crop)
        b, g, r = cv2.split(crop)
        # cv2.imshow('Cropped', crop)
        # Make prediction
        pred = model.predict((1 - b).reshape(16*16))

        # Show bounding box and prediction on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, y_to_letter(pred), (x, y + 15), 0, 0.8, (200, 170, 5), 2)
    cv2.imshow('OCR', image)
    cv2.waitKey(0)
    cv2.imshow('OCR', image)
    cv2.waitKey(0)
    cv2.imshow('OCR', image)
    cv2.waitKey(0)
    cv2.imshow('OCR', image)
    cv2.waitKey(0)


def test_page_recognition():
    xs, ys = load_training_samples(1000)
    model = train_network(xs, ys, epochs = 30, learn_speed = 0.0000000000001)
    model.plot_loss(from_epoch=5)

    for_each_image_segmentation("validation_data/ocr-full-2.jpg", model)
