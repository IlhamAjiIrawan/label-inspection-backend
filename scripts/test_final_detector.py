import sys
sys.path.insert(0, 'label-inspection-backend')
import cv2
from app.vision.label_detector import detect_labels

cam = cv2.imread('cam_actual.png')
print('cam shape:', cam.shape)

res = detect_labels(cam)
print('Success:', res['success'], 'Slots:', len(res['slots']))
for s in res['slots']:
    print(f"  {s['label_name']}: crop shape={s['cropped_image'].shape}, rect={s['rect']}")
    cv2.imwrite(f"test_out_{s['label_name']}.png", s['cropped_image'])

print('\nTesting rotated horizontal:')
cam_h = cv2.rotate(cam, cv2.ROTATE_90_COUNTERCLOCKWISE)
res_h = detect_labels(cam_h)
print('Success:', res_h['success'], 'Slots:', len(res_h['slots']))
for s in res_h['slots']:
    print(f"  {s['label_name']}: crop shape={s['cropped_image'].shape}, rect={s['rect']}")

