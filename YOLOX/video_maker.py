import sys
import cv2

#画像の保管場所のパス変更
file_path='./image/image_sunny20221111/sunny/training/image/'
#保存する動画の名前注意
video_name='sunny20221111.mp4'
#使用する画像の枚数変更
photo_number=10001


print("START")
# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter(video_name,fourcc, 20.0, (720, 360))

if not video.isOpened():
    print("can't be opened")
    sys.exit()

for i in range(0, photo_number):
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread(file_path+'%06d.png' % i)

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    if i%500==0:
        print(i)

video.release()
print('written')

