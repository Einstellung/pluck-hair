import gxipy as gx
import numpy as np
import cv2

def main():
    # 初始化设备管理器
    dev_mgr = gx.DeviceManager()
    num, dev_info_list = dev_mgr.update_device_list()
    if num == 0:
        print("No Daheng camera found")
        return

    # 打开第一个设备（索引从 1 开始）
    cam = dev_mgr.open_device_by_index(1)

    # 可选：关闭自动曝光／自动增益
    cam.ExposureAuto.set(False)
    cam.GainAuto.set(False)

    # 启动流
    cam.stream_on()

    # 获取数据流，第一个通道
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("Failed to get image")
    else:
        # 转为 numpy 数组
        np_img = raw_image.get_numpy_array()
        print("Captured image shape:", np_img.shape)

        # 使用 OpenCV 显示图像
        cv2.imshow("Camera Image", np_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()

    # 停止流和关闭设备
    cam.stream_off()
    cam.close_device()

if __name__ == '__main__':
    main()
