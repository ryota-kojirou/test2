import argparse
import gc
import logging
import os
import subprocess
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
# from tf_pose.common import CocoPart
from modules.humans_to_array import humans_to_array
from modules.motion_analysis import MotionAnalysis
from modules.track_humans import track_humans

fps_time = 0

# spline
"""
Cubic Spline library on python
author Atsushi Sakai
"""
import bisect
import math

class Spline:
    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calc_d(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calc_dd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B

    def calc_curvature(self, t):
        j = int(math.floor(t))
        if j < 0:
            j = 0
        elif j >= len(self.a):
            j = len(self.a) - 1

        dt = t - j
        df = self.b[j] + 2.0 * self.c[j] * dt + 3.0 * self.d[j] * dt * dt
        ddf = 2.0 * self.c[j] + 6.0 * self.d[j] * dt
        k = ddf / ((1 + df ** 2) ** 1.5)
        return k


class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calc_d(s)
        ddx = self.sx.calc_dd(s)
        dy = self.sy.calc_d(s)
        ddy = self.sy.calc_dd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calc_d(s)
        dy = self.sy.calc_d(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_2d_spline_interpolation(x, y, num=100):
    """
    Calc 2d spline course with interpolation
    :param x: interpolated x positions
    :param y: interpolated y positions
    :param num: number of path points
    :return:
        - x     : x positions
        - y     : y positions
        - yaw   : yaw angle list
        - k     : curvature list
        - s     : Path length from start point
    """
    sp = Spline2D(x, y)
    s = np.linspace(0, sp.s[-1], num+1)[:-1]

    r_x, r_y, r_yaw, r_k = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        r_x.append(ix)
        r_y.append(iy)
        r_yaw.append(sp.calc_yaw(i_s))
        r_k.append(sp.calc_curvature(i_s))

    travel = np.cumsum([np.hypot(dx, dy) for dx, dy in zip(np.diff(r_x), np.diff(r_y))]).tolist()
    travel = np.concatenate([[0.0], travel])

    return r_x, r_y, r_yaw, r_k, travel


def test_spline2d():
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    input_x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    input_y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

    x, y, yaw, k, travel = calc_2d_spline_interpolation(input_x, input_y, num=200)

    plt.subplots(1)
    plt.plot(input_x, input_y, "xb", label="input")
    plt.plot(x, y, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(travel, [math.degrees(i_yaw) for i_yaw in yaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(travel, k, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()


def test_spline():
    print("Spline test")
    import matplotlib.pyplot as plt
    x = [-0.5, 0.0, 0.5, 1.0, 1.5]
    y = [3.2, 2.7, 6, 5, 6.5]

    spline = Spline(x, y)
    rx = np.arange(-2.0, 4, 0.01)
    ry = [spline.calc(i) for i in rx]

    plt.plot(x, y, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    test_spline()
    test_spline2d()

# spline end




# if __name__ == '__main__':
def estimate_trajectory(video, path='', resize='432x368', model='cmu', resize_out_ratio=4.0, orientation='horizontal',
                   cog="skip", cog_color='black', cog_size='M', showBG=True, start_frame=0, debug=False, plot_image=""):
    logger = logging.getLogger('TfPoseEstimator')
    logger.setLevel(logging.DEBUG) if debug else logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG) if debug else ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    # formatter = logging.Formatter('[#(asctime)s] [#(name)s] [#(levelname)s] #(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # data directory
    if path:
        path_movie_src = os.path.join(path, 'movies', video)
    else:
        path_movie_src = video
    path_movie_out = os.path.join(path, 'movies_estimated')
    path_csv_estimated = os.path.join(path, 'data_estimated')
    path_png_estimated = os.path.join(path, 'png_estimated')
    csv_file = os.path.join(path_csv_estimated, video.rsplit('.')[0] + '.csv')
    os.makedirs(path_movie_out, exist_ok=True)
    os.makedirs(path_png_estimated, exist_ok=True)
    os.makedirs(path_csv_estimated, exist_ok=True)

    w, h = model_wh(resize)
    if orientation == 'horizontal':
        if w == 0: w = 432
        if h == 0: h = 368
    else:
        if w == 0: w = 368
        if h == 0: h = 432
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    logger.info('resize: %d,  %d' % (w, h))

    cap = cv2.VideoCapture(path_movie_src)
    logger.info("OPEN: %s" % path_movie_src)
    if cap.isOpened() is False:
        logger.info("ERROR: opening video stream or file")
    caps_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info('MODE: Plot Center of Gravity')
    ma = MotionAnalysis()
    # CSV FILE SETTING
    segments = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "human_id",
                "head_cog", "torso_cog", "r_thigh_cog", "l_thigh_cog", "r_leg_cog", "l_leg_cog", "r_foot_cog",
                "l_foot_cog",
                "r_arm_cog", "l_arm_cog", "r_forearm_cog", "l_forearm_cog", "r_hand_cog", "l_hand_cog"]
    seg_columns = ['frame']
    [seg_columns.extend([x + '_x', x + '_y', x + '_score']) for x in segments]
    df_template = pd.DataFrame(columns=seg_columns)
    df_template.to_csv(csv_file, index=False)

    # change marker size of cog
    if (cog_size == "s") or (cog_size == "S"):
        cog_size = 10000
    else:
        cog_size = 20000

    # processing video
    frame_no = 0
    count = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        if frame_no == 0:
            h_pxl, w_pxl = image.shape[0], image.shape[1]
        if frame_no < start_frame:
            frame_no += 1
            continue

        # estimate pose
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        time_estimation = time.time() - t
        a_humans = humans_to_array(humans)

        # check the time to estimation
        if (frame_no % int(caps_fps)) == 0:
            logger.info("Now estimating at:" + str(int(frame_no / caps_fps)) + "[sec]")
            logger.info('inference in %.4f seconds.' % time_estimation)
            logger.debug('shape of image: ' + str(image.shape))
            logger.debug(str(a_humans))

        # track human
        if frame_no == start_frame:
            # initialize
            humans_id = np.array(range(len(a_humans)))
            np_humans_current = np_humans = np.concatenate((np.c_[np.repeat(frame_no, len(a_humans))],
                                        a_humans.reshape(a_humans.shape[0], a_humans.shape[1] * a_humans.shape[2]),
                                        np.c_[humans_id]), axis=1)
            clm_of_id = np_humans.shape[1] - 1
        else:
            humans_id = track_humans(a_humans, post_humans, humans_id)
            np_humans_current = np.concatenate((np.c_[np.repeat(frame_no, len(a_humans))],
                                             a_humans.reshape(a_humans.shape[0], a_humans.shape[1] * a_humans.shape[2]),
                                             np.c_[humans_id]), axis=1)
            np_humans = np.concatenate((np_humans[np_humans[:, 0] > (frame_no - 30)], np_humans_current))
        post_humans = a_humans

        # calculate center of gravity
        if cog != 'skip':
            t = time.time()
            bodies_cog = ma.multi_bodies_cog(humans=humans)
            bodies_cog[np.isnan(bodies_cog[:, :, :])] = 0
            humans_feature = np.concatenate((np_humans_current,
                                             bodies_cog.reshape(bodies_cog.shape[0],
                                                                bodies_cog.shape[1] * bodies_cog.shape[2])), axis=1)
            df_frame = pd.DataFrame(humans_feature.round(4))
            df_frame.to_csv(csv_file, index=False, header=None, mode='a')
            time_cog = time.time() - t
            if frame_no % int(caps_fps) == 0:
                logger.info('calculation of cog in %.4f seconds.' % time_cog)

        if plot_image != 'skip':
            fig_resize = 100
            plt.figure(figsize=(int(w_pxl / fig_resize), int(h_pxl / fig_resize)))
            if not showBG:
                image = np.zeros(image.shape)
#             image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if cog != 'skip':
                plt.scatter(bodies_cog[:, 14, 0] * w_pxl, bodies_cog[:, 14, 1] * h_pxl, color=cog_color,
                            marker='o', s=cog_size/fig_resize)
                plt.vlines(bodies_cog[:, 6, 0] * w_pxl, ymin=0, ymax=h_pxl, linestyles='dashed')
                plt.vlines(bodies_cog[:, 7, 0] * w_pxl, ymin=0, ymax=h_pxl, linestyles='dashed')

            # plot trajectories r_wrist:4, l_wrist:7
            for i, hum in enumerate(np.sort(humans_id)):
                df_human = np_humans[np_humans[:, clm_of_id] == hum]
                if len(df_human)>15:
                  num = len(df_human)-15
                  df_human = df_human[num:]
                  
                  # spline
                  x = df_human[:, 4 * 3 + 1] * w_pxl
                  y = df_human[:, 4 * 3 + 2] * h_pxl
                  spline = Spline(x, y)
                  rx = np.arange(0, w_pxl, 0.01)
                  ry = [spline.calc(i) for i in rx]
                  
#                 df_human[:, 4 * 3 + 1] = df_human[:, 4 * 3 + 1] * w_pxl
#                 df_human[:, 7 * 3 + 1] = df_human[:, 7 * 3 + 1] * w_pxl  
#                 df_human[:, 4 * 3 + 2] = df_human[:, 4 * 3 + 2] * h_pxl
#                 df_human[:, 7 * 3 + 2] = df_human[:, 7 * 3 + 2] * h_pxl
#                 for n in range(len(df_human[:, 4 * 3 + 1])-1):
#                   plt.plot([df_human[:, 4 * 3 + 1][n], df_human[:, 4 * 3 + 1][n+1]], [df_human[:, 4 * 3 + 2][n], df_human[:, 4 * 3 + 2][n+1]], linewidth=400/fig_resize, alpha=0.6, color="darkorange")
#                   plt.plot([df_human[:, 7 * 3 + 1][n], df_human[:, 7 * 3 + 1][n+1]], [df_human[:, 7 * 3 + 2][n], df_human[:, 7 * 3 + 2][n+1]], linewidth=400/fig_resize, alpha=0.2+(n/14)*0.5, color="darkorange")
                plt.plot(rx, ry, linewidth=400/fig_resize, alpha=0.6, color="darkorange")
                plt.plot(df_human[:, 7 * 3 + 1] * w_pxl, df_human[:, 7 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.6, color="darkorange")
#                 if count<200:
# #                   print(df_human)
# #                   print("\n")
# #                   print(df_human[:, 1 * 3 + 1])
#                   plt.plot(df_human[:, 1 * 3 + 1] * w_pxl, df_human[:, 1 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                   plt.plot(df_human[:, 4 * 3 + 1] * w_pxl, df_human[:, 4 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                   plt.plot(df_human[:, 7 * 3 + 1] * w_pxl, df_human[:, 7 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                 elif count<300:
#                   plt.plot(df_human[:, 2 * 3 + 1] * w_pxl, df_human[:, 2 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                   plt.plot(df_human[:, 5 * 3 + 1] * w_pxl, df_human[:, 5 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                 else:
#                   plt.plot(df_human[:, 2 * 3 + 1] * w_pxl, df_human[:, 2 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                   plt.plot(df_human[:, 5 * 3 + 1] * w_pxl, df_human[:, 5 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                   plt.plot(df_human[:, 4 * 3 + 1] * w_pxl, df_human[:, 4 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                   plt.plot(df_human[:, 7 * 3 + 1] * w_pxl, df_human[:, 7 * 3 + 2] * h_pxl, linewidth=400/fig_resize, alpha=0.7, color="darkorange")
#                 plt.text(400,400,str(count))
#                 count = count+1

            plt.ylim(h_pxl, 0)

            # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
            plt.savefig(os.path.join(path_png_estimated,
                                     video.split('.')[-2] + '{:06d}'.format(frame_no) + ".png"))
            plt.close()
            plt.clf()

        # before increment, renew some args
        frame_no += 1
        gc.collect()
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    logger.info("finish estimation & start encoding")
    cmd = ["ffmpeg", "-r", str(caps_fps), "-start_number", str(start_frame),
           "-i", os.path.join(path_png_estimated, video.split('.')[-2] + "%06d.png"),
           "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           os.path.join(path_movie_out, video.split('.')[-2] + "_track.mp4")]
    subprocess.call(cmd)
    logger.debug('finished+')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resize', type=str, default='0x0', help='network input resize. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--cog', type=str, default="")
    parser.add_argument('--cog_color', type=str, default='black')
    parser.add_argument('--cog_size', type=str, default='M')
    parser.add_argument('--resize_out_ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--orientation', type=str, default="horizontal")
    parser.add_argument('--plot_image', type=str, default="")
    args = parser.parse_args()
    print(str(args.cog))
    estimate_trajectory(video=args.video, path=args.path, resize=args.resize, model=args.model, orientation=args.orientation,
                        resize_out_ratio=args.resize_out_ratio, showBG=args.showBG, plot_image=args.plot_image,
                        cog=args.cog, cog_color=args.cog_color, cog_size=args.cog_size, start_frame=args.start_frame, debug=args.debug)
