from vision_inference import extract_and_classify


sources = ["video_files/fire0.mp4","video_files/fire1.mp4", "video_files/fire2.mp4", "video_files/fire3.mp4", "video_files/fire4.mp4", 
           "video_files/cc0.mp4", "video_files/cc1.mp4", "video_files/cc2.mp4", "video_files/cc3.mp4", "video_files/cc4.mp4", 
           "video_files/none0.mp4", "video_files/none1.mp4", "video_files/none2.mp4", "video_files/none3.mp4", "video_files/none4.mp4"]
labels = ["fire", "fire", "fire", "fire", "fire", "cc", "cc", "cc", "cc", "cc", "none", "none", "none", "none", "none"]
correct = 0


for idx in range(len(sources)):
    returned = extract_and_classify(source=sources[idx])
    if (labels[idx] in returned):
        correct += 1
        print("yup!")
    else:
        print("nope!")

total = 15
print("Testing complete             Accuracy: " + str(correct/total))