import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = [
        '../dataset/mnist_fashion/image_0.jpg',
        '../dataset/mnist_fashion/image_1.jpg',
        '../dataset/mnist_fashion/image_2.jpg',
        '../dataset/mnist_fashion/image_3.jpg',
        '../dataset/mnist_fashion/image_4.jpg',
        '../dataset/mnist_fashion/image_5.jpg',
        '../dataset/mnist_fashion/image_6.jpg',
        '../dataset/mnist_fashion/image_7.jpg',
        '../dataset/mnist_fashion/image_8.jpg',
        '../dataset/mnist_fashion/image_9.jpg',
        ]

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]