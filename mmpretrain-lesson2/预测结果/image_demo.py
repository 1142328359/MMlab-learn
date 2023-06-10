from mmpretrain import ImageClassificationInferencer

inferencer = ImageClassificationInferencer('./resnet50_fruit.py', pretrained='exp/best_accuracy_top1_epoch_5.pth')

result = inferencer("./banana.jpg")

print(result)