# -*- coding: utf-8 -*-
'''
测试车型识别模型

用法：
python test_vehicle_type.py
'''

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

# 全局变量
model_path = 'models/vehicle_type.hdf5'
test_img_path = "tests/768.jpg"
# 全局常量
VEHICLE_WIDTH = 32
VEHICLE_HEIGHT = 32

# load the  CNN model
model = load_model(model_path)
test_img = cv2.imread(test_img_path)
gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
roi = cv2.resize(gray, (VEHICLE_WIDTH, VEHICLE_HEIGHT))
roi = roi.astype("float") / 255.0
roi = img_to_array(roi)
roi = np.expand_dims(roi, axis=0)
(bus,car,minibus,truck ) = model.predict(roi)[0]
result = {"bus":bus,"car": car, "minibus":minibus, "truck":truck}
label = max(result, key=result.get)
    
cv2.putText(test_img, label, (50, 50),
cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

cv2.imshow("Vehicle Type recognition", test_img)
k = cv2.waitKey()
cv2.destroyAllWindows()