from ultralytics import YOLO
def train_model():
    model = YOLO("yolov8.yaml").load('yolov8m.pt')
    model.train(data="mydata/mydata.yaml", epochs=200)

if __name__ == '__main__':
    train_model()




