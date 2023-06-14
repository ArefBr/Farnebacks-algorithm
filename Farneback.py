import time

import cv2
import numpy as np


def farnebacks(video, device):
    FRAME_SKIP = 5

    # init dict to track time for every stage at each iteration
    timers = {
        "full pipeline": [],
        "reading": [],
        "pre-process": [],
        "optical flow": [],
        "post-process": [],
    }

    # init video capture with video
    cap = cv2.VideoCapture(video)

    # get default video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get total number of video frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read the first frame
    ret, previous_frame = cap.read()

    if ret is False: return

    if device == "cpu":

        # proceed if frame reading was successful
        if ret:
            # resize frame
            frame = cv2.resize(previous_frame, (960, 540))

            # convert to gray
            previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # create hsv output for optical flow
            hsv = np.zeros_like(frame, np.float32)

            # set saturation to 1
            hsv[..., 1] = 1.0

            for i in range(num_frames):

                if not i%5==0: continue

                # start full pipeline timer
                start_full_time = time.time()

                # start reading timer
                start_read_time = time.time()

                # capture frame-by-frame
                ret, frame = cap.read()

                # end reading timer
                end_read_time = time.time()

                # add elapsed iteration time
                timers["reading"].append(end_read_time - start_read_time)

                # if frame reading was not successful, break
                if not ret:
                    break

                # start pre-process timer
                start_pre_time = time.time()
                # resize frame
                frame = cv2.resize(frame, (960, 540))

                # convert to gray
                current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # end pre-process timer
                end_pre_time = time.time()

                # add elapsed iteration time
                timers["pre-process"].append(end_pre_time - start_pre_time)

                # start optical flow timer
                start_of = time.time()

                # calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    previous_frame, current_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,
                )
                # end of timer
                end_of = time.time()

                # add elapsed iteration time
                timers["optical flow"].append(end_of - start_of)

                # start post-process timer
                start_post_time = time.time()

                # Calculate the magnitude of optical flow vectors
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

                # Calculate the average magnitude for this pair of frames
                average_magnitude = np.mean(magnitude)
                print("Average Magnitude:", average_magnitude)

                # Accumulate the total movement
                total_movement += average_magnitude
                
                # update previous_frame value
                previous_frame = current_frame

                # end post-process timer
                end_post_time = time.time()

                # add elapsed iteration time
                timers["post-process"].append(end_post_time - start_post_time)

                # end full pipeline timer
                end_full_time = time.time()

                # add elapsed iteration time
                timers["full pipeline"].append(end_full_time - start_full_time)

                # visualization
                cv2.imshow("original", frame)
                # cv2.imshow("result", bgr)
                k = cv2.waitKey(1)
                if k == 27:
                    break

    else:

        # proceed if frame reading was successful
        if ret:
            # resize frame
            frame = cv2.resize(previous_frame, (0, 0), fx=0.7, fy=0.7)

            # upload resized frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # convert to gray
            previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # upload pre-processed frame to GPU
            gpu_previous = cv2.cuda_GpuMat()
            gpu_previous.upload(previous_frame)

            # create gpu_hsv output for optical flow
            gpu_hsv = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
            gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)

            gpu_h = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
            gpu_s = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
            gpu_v = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)

            # set saturation to 1
            gpu_s.upload(np.ones_like(previous_frame, np.float32))

            for i in range(num_frames):

                if not i%5==0: continue
                # start full pipeline timer
                start_full_time = time.time()

                # start reading timer
                start_read_time = time.time()

                # capture frame-by-frame
                ret, frame = cap.read()

                # upload frame to GPU
                gpu_frame.upload(frame)

                # end reading timer
                end_read_time = time.time()

                # add elapsed iteration time
                timers["reading"].append(end_read_time - start_read_time)

                # if frame reading was not successful, break
                if not ret:
                    break

                # start pre-process timer
                start_pre_time = time.time()

                # resize frame
                gpu_frame = cv2.cuda.resize(gpu_frame, (0, 0), fx=0.7, fy=0.7)

                # convert to gray
                gpu_current = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

                # end pre-process timer
                end_pre_time = time.time()

                # add elapsed iteration time
                timers["pre-process"].append(end_pre_time - start_pre_time)

                # start optical flow timer
                start_of = time.time()

                # create optical flow instance
                gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
                    5, 0.5, False, 15, 3, 5, 1.2, 0,
                )
                # calculate optical flow
                gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
                    gpu_flow, gpu_previous, gpu_current, None,
                )

                # end of timer
                end_of = time.time()

                # add elapsed iteration time
                timers["optical flow"].append(end_of - start_of)

                # start post-process timer
                start_post_time = time.time()

                gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
                gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
                cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

                # start post-process timer
                start_post_time = time.time()

                # Calculate the magnitude of optical flow vectors
                magnitude = np.sqrt(gpu_flow_x.download() ** 2 + gpu_flow_y.download() ** 2)

                # Calculate the average magnitude for this pair of frames
                average_magnitude = np.mean(magnitude)
                print("Average Magnitude:", average_magnitude)

                # Accumulate the total movement
                total_movement += average_magnitude

                # update previous_frame value
                gpu_previous = gpu_current

                # end post-process timer
                end_post_time = time.time()

                # add elapsed iteration time
                timers["post-process"].append(end_post_time - start_post_time)

                # end full pipeline timer
                end_full_time = time.time()

                # add elapsed iteration time
                timers["full pipeline"].append(end_full_time - start_full_time)

                # visualization
                cv2.imshow("original", frame)
                k = cv2.waitKey(1)
                if k == 27:
                    break

    # release the capture
    cap.release()

    # destroy all windows
    cv2.destroyAllWindows()

    # print results
    print("Number of frames : ", num_frames)

    # elapsed time at each stage
    print("Elapsed time")
    for stage, seconds in timers.items():
        print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

    # calculate frames per second
    print("Default video FPS : {:0.3f}".format(fps))

    of_fps = (num_frames - 1) / sum(timers["optical flow"])
    print("Optical flow FPS : {:0.3f}".format(of_fps))

    full_fps = (num_frames - 1) / sum(timers["full pipeline"])
    print("Full pipeline FPS : {:0.3f}".format(full_fps))


if __name__ == "__main__":
    farnebacks('../IMG_6655.MP4','gpu')
