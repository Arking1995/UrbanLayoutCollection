import numpy as np
import os
import copy
import cv2
from scipy.optimize import minimize
import glob
from utils import make_rectangle, make_cross, make_ushape, make_hole, IoU, make_lshape, make_obj

vec_length = np.array([4, 12, 6, 8, 8])


############################ 
# 1: rectangle
# 2: cross
# 3: L-shape
# 4: U-shape
# 5: With-hole
###########################
def fit_bldg_features(img_data):

    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    data = None

    # print(img.shape)
    # cv2.imshow('Gray image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    bufferwidth = 50

    shift_minw = []
    shift_minh = []

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print('Find contour: ',len(contours))


    Shift = np.zeros((int(len(contours)),2))
    Relative_contours = copy.deepcopy(contours)
    for i in range(len(contours)): # len(contours)

        if hierarchy[0, i, 3] != -1: # not father contour
            continue

        minw = int(min(contours[i][:, :, 0]))
        minh = int(min(contours[i][:, :, 1]))

        for j in range(len(contours[i])):
            Shift[i,0] = - minw + bufferwidth
            Shift[i,1] = - minh + bufferwidth


    for i in range(len(contours)): # len(contours)
        father = hierarchy[0, i, 3]
        if father != -1: # not father contour
            Shift[i, 0] = Shift[father, 0]
            Shift[i, 1] = Shift[father, 1]


    for i in range(len(contours)): # len(contours)
        for j in range(len(contours[i])):
            Relative_contours[i][j][0][0] = contours[i][j][0][0] + Shift[i,0]
            Relative_contours[i][j][0][1] = contours[i][j][0][1] + Shift[i,1]



    count = 0
    for i in range(len(contours)):  # len(contours)

        # if hierarchy[0, i, 3] != -1: # not father contour
        #     continue

        # if contours[i].shape[0] < 4:
        #     continue

        minw = int(min(contours[i][:, :, 0]))
        minh = int(min(contours[i][:, :, 1]))

        maxw = int(max(contours[i][:, :, 0]))
        maxh = int(max(contours[i][:, :, 1]))

        ## if minw == 0 or minh == 0 or maxw == 1999 or maxh == 1999:
        ##     continue

        # if (maxw-minw) < 2 or (maxh-minh) < 2:
        #     continue

        shift_minw.append(minw-bufferwidth)
        shift_minh.append(minh-bufferwidth)

        height = maxh - minh + 1
        width = maxw - minw + 1
        data = np.zeros((height + bufferwidth*2, width + bufferwidth*2))

        cv2.drawContours(data, Relative_contours, i, (255), thickness=-1, hierarchy = hierarchy)


    # cv2.imwrite('tmp_img.bmp', data) # write image to file, and loaded by each fitting function.
    if data is None:
        return 1, None, None, None, None

    org_data = data

    # print(data.shape)
    # cv2.imshow('Gray image', data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    IoU_list = np.zeros(1)
    rec_para_result = []
    best_fit = []
    IoU_thres = 0.10

    #################################################   rectangle  fit  #########################################################################################
    fit_rec = []
    IoU_rec = []
    # data = cv2.imread('tmp_img.bmp', 0)
    data = np.array(org_data, dtype=np.uint8)

    height, width = data.shape
    args = [data, 'rectangle']

    init_para = [height / 2.0, width / 2.0, (height - bufferwidth * 2), (width - bufferwidth * 2), 0.1]
    res0 = minimize(IoU, init_para, args=args, method='powell')
    fit_rec.append(res0)
    IoU_rec.append(1 - res0.fun)

    if res0.fun > IoU_thres:
        init_para = [height / 2.0, width / 2.0, (height - bufferwidth * 2), (width - bufferwidth * 2),
                     0.7854]
        res1 = minimize(IoU, init_para, args=args, method='powell')
        fit_rec.append(res1)
        IoU_rec.append(1 - res1.fun)

        if res1.fun > IoU_thres:
            init_para = [height / 2.0, width / 2.0, (height - bufferwidth * 2),
                         (width - bufferwidth * 2),
                         1.5708]
            res2 = minimize(IoU, init_para, args=args, method='powell')
            fit_rec.append(res2)
            IoU_rec.append(1 - res2.fun)

            if res2.fun > IoU_thres:
                init_para = [height / 2.0, width / 2.0, (height - bufferwidth * 2),
                             (width - bufferwidth * 2),
                             2.3562]
                res3 = minimize(IoU, init_para, args=args, method='powell')
                fit_rec.append(res3)
                IoU_rec.append(1 - res3.fun)

    IoU_rec = np.array(IoU_rec).reshape(-1)
    maxid = np.argmax(IoU_rec)

    IoU_list[count] = IoU_rec[maxid]
    # print('Rec IoU:' + str(IoU_rec[maxid]))
    rectangle = make_rectangle(fit_rec[maxid].x)
    rec_para_result.append(fit_rec[maxid].x)  # legacy for further cross matching
    best_fit.append([0, fit_rec[maxid]])

    # fit_contours = np.array(rectangle).reshape((4, 1, 2)).astype(
    #     np.int)  # the type should be int, otherwise return "npoints > 0" error
    # fit_contours = [fit_contours]
    #
    # fit_img = np.zeros(data.shape, np.uint8)
    # cv2.drawContours(fit_img, fit_contours, -1, (255), thickness=1)
    # cv2.imwrite('fit_img_recshape.png', fit_img)


    ################################   cross  fit   ###########################################################################################################
    count = 0
    Nonperfectrec_idx = np.array(np.where(IoU_list < (1.0 - IoU_thres))).reshape(-1)
    Crs_IoU_list = np.zeros(Nonperfectrec_idx.shape)
    for id in Nonperfectrec_idx:
        fit_crs = []
        IoU_crs = []
        # data = cv2.imread('tmp_img.bmp', 0)
        data = np.array(org_data, dtype=np.uint8)
        height, width = data.shape
        h1 = height / 10.0
        w1 = width / 10.0
        h2 = height / 5.0
        w2 = width / 5.0
        centery, centerx, leheight, lewidth, theta = rec_para_result[id]
        leh1 = leheight / 10.0
        lew1 = lewidth / 10.0
        leh2 = leheight / 5.0
        lew2 = lewidth / 5.0
        args = [data, 'cross']

        # print('size: '+str(leheight)+',  '+str(lewidth))
        if leheight < 50.0:
            leh1 = 1.0
            leh2 = 5.0
            h1 = 1.0
            h2 = 5.0

        if lewidth < 50.0:
            lew1 = 1.0
            lew2 = 5.0
            w1 = 1.0
            w2 = 5.0

        init_para = [height / 2.0, width / 2.0, (height - bufferwidth * 2), (width - bufferwidth * 2), 0.1,
                     w1, h1, w1, h1, w1, h1, w1, h1]
        res0 = minimize(IoU, init_para, args=args, method='powell')
        fit_crs.append(res0)
        IoU_crs.append(1 - res0.fun)

        if res0.fun > IoU_thres:
            init_para = [height / 2.0, width / 2.0, (height - bufferwidth * 2), (width - bufferwidth * 2),
                         0.1, w2, h2, w2, h2, w2, h2, w2,
                         h2]  # , w1, h1, w1, h1, w1, h1, w1, h1 ##  w2, h2, w2, h2, w2, h2, w2, h2
            res1 = minimize(IoU, init_para, args=args, method='powell')
            fit_crs.append(res1)
            IoU_crs.append(1 - res1.fun)

            if res1.fun > IoU_thres:
                init_para = [centery, centerx, leheight, lewidth, theta, lew1, leh1, lew1, leh1, lew1, leh1,
                             lew1, leh1]  # lew2, leh2, lew2, leh2, lew2, leh2, lew2, leh2
                res2 = minimize(IoU, init_para, args=args, method='powell')
                fit_crs.append(res2)
                IoU_crs.append(1 - res2.fun)

                if res2.fun > IoU_thres:
                    centery, centerx, height, width, theta = rec_para_result[id]
                    init_para = [centery, centerx, leheight, lewidth, theta, lew2, leh2, lew2, leh2, lew2,
                                 leh2, lew2, leh2]  # lew1, leh1, lew1, leh1, lew1, leh1, lew1, leh1
                    res3 = minimize(IoU, init_para, args=args, method='powell')
                    fit_crs.append(res3)
                    IoU_crs.append(1 - res3.fun)

        IoU_crs = np.array(IoU_crs).reshape(-1)
        maxid = np.argmax(IoU_crs)
        # print('Crs IoU:' + str(IoU_crs[maxid]))

        Crs_IoU_list[count] = IoU_crs[maxid]

        if 1 - best_fit[id][1].fun < IoU_crs[maxid]:
            best_fit[id][1] = fit_crs[maxid]
            best_fit[id][0] = 1

        count += 1
        # cross = make_cross(fit_crs[maxid].x)
        #
        # fit_contours = np.array(cross).reshape((12,  # number should change for more variant shapes
        #                                         1, 2)).astype(np.int
        #                                                       )  # the type should be int, otherwise return "npoints > 0" error
        # fit_contours = [fit_contours]
        # fit_img = np.zeros(data.shape, np.uint8)
        # cv2.drawContours(fit_img, fit_contours, -1, (255), thickness=1)
        # cv2.imwrite('fit_img_crsshape.png',fit_img)





    ###############################   L-shape  fit   ############################################################################################################
    count = 0
    Nonperfectcrs_idx = Nonperfectrec_idx[np.array(np.where(Crs_IoU_list < (1.0 - IoU_thres))).reshape(-1)]
    Lshape_IoU_list = np.zeros(Nonperfectcrs_idx.shape)
    for id in Nonperfectcrs_idx:
        fit_lshape = []
        IoU_lshape = []
        # data = cv2.imread('tmp_img.bmp', 0)
        data = np.array(org_data, dtype=np.uint8)
        height, width = data.shape


        # centery, centerx, leheight, lewidth, theta = rec_para_result[id]

        centery = height / 2.0
        centerx = width / 2.0
        height = height-bufferwidth*2
        width = width-bufferwidth*2
        theta = 0.1

        init_para = [centery, centerx, height, width, theta, width / 2.0, height / 2.0]

        args = [data, 'lshape']
        res0 = minimize(IoU, init_para, args=args, method='powell')
        fit_lshape.append(res0)
        IoU_lshape.append(1 - res0.fun)

        if res0.fun > IoU_thres:
            init_para = [centery, centerx, width, height, theta + 0.5*np.pi, height / 2.0, width / 2.0,]
            res1 = minimize(IoU, init_para, args=args, method='powell')
            fit_lshape.append(res1)
            IoU_lshape.append(1 - res1.fun)

            if res1.fun > IoU_thres:
                init_para = [centery, centerx, height, width, theta + np.pi, width / 2.0, height / 2.0]
                res2 = minimize(IoU, init_para, args=args, method='powell')
                fit_lshape.append(res2)
                IoU_lshape.append(1 - res2.fun)

                if res2.fun > IoU_thres:
                    init_para = [centery, centerx,  width, height, theta + 1.5*np.pi,  height / 2.0, width / 2.0]
                    res3 = minimize(IoU, init_para, args=args, method='powell')
                    fit_lshape.append(res3)
                    IoU_lshape.append(1 - res3.fun)


        IoU_lshape = np.array(IoU_lshape).reshape(-1)
        maxid = np.argmax(IoU_lshape)

        Lshape_IoU_list[count] = IoU_lshape[maxid]

        # print('L-shape IoU:' + str(IoU_lshape[maxid]))

        if 1 - best_fit[id][1].fun < IoU_lshape[maxid]:
            best_fit[id][1] = fit_lshape[maxid]
            best_fit[id][0] = 2

        count += 1

        # lshape = make_lshape(fit_lshape[maxid].x)
        #
        # fit_contours = np.array(lshape).reshape((6,  # number should change for more variant shapes
        #                                          1, 2)).astype(np.int
        #                                                        )  # the type should be int, otherwise return "npoints > 0" error
        # fit_contours = [fit_contours]
        # fit_img = np.zeros(data.shape, np.uint8)
        # cv2.drawContours(fit_img, fit_contours, -1, (255), thickness=1)
        # cv2.imwrite('fit_img_lshape.png',fit_img)




    ###############################   ushape  fit   ############################################################################################################
    count = 0
    Nonperfect_lshape_idx = Nonperfectcrs_idx[np.array(np.where(Lshape_IoU_list < (1.0 - IoU_thres))).reshape(-1)]
    Ushape_IoU_list = np.zeros(Nonperfect_lshape_idx.shape)
    for id in Nonperfect_lshape_idx:
        fit_ushape = []
        IoU_ushape = []
        # data = cv2.imread('tmp_img.bmp', 0)
        data = np.array(org_data, dtype=np.uint8)
        height, width = data.shape

        # centery, centerx, leheight, lewidth, theta = rec_para_result[id]

        centery = height / 2.0
        centerx = width / 2.0
        height = height-bufferwidth*2
        width = width-bufferwidth*2
        theta = 0.1


        init_para = [centery, centerx, height, width, theta, width / 4.0, width / 4.0, height / 2.0]
        args = [data, 'ushape']
        res0 = minimize(IoU, init_para, args=args, method='powell')
        fit_ushape.append(res0)
        IoU_ushape.append(1 - res0.fun)

        if res0.fun > IoU_thres:
            init_para = [centery, centerx, width, height, theta + 0.5*np.pi, height / 4.0, height / 4.0, width / 2.0]
            res1 = minimize(IoU, init_para, args=args, method='powell')
            fit_ushape.append(res1)
            IoU_ushape.append(1 - res1.fun)

            if res1.fun > IoU_thres:
                init_para = [centery, centerx, height, width, theta + 1.0*np.pi, width / 4.0, width / 4.0,
                             height / 2.0]
                res2 = minimize(IoU, init_para, args=args, method='powell')
                fit_ushape.append(res2)
                IoU_ushape.append(1 - res2.fun)

                if res2.fun > IoU_thres:
                    init_para = [centery, centerx, width, height, theta + 1.5*np.pi, height / 4.0, height / 4.0,
                                 width / 2.0]
                    res3 = minimize(IoU, init_para, args=args, method='powell')
                    fit_ushape.append(res3)
                    IoU_ushape.append(1 - res3.fun)

        IoU_ushape = np.array(IoU_ushape).reshape(-1)
        maxid = np.argmax(IoU_ushape)

        Ushape_IoU_list[count] = IoU_ushape[maxid]

        # print('U-shape IoU:' + str(IoU_ushape[maxid]))

        if 1 - best_fit[id][1].fun < IoU_ushape[maxid]:
            best_fit[id][1] = fit_ushape[maxid]
            best_fit[id][0] = 3

        count += 1
        # ushape = make_ushape(fit_ushape[maxid].x)
        #
        # fit_contours = np.array(ushape).reshape((8,  # number should change for more variant shapes
        #                                          1, 2)).astype(np.int
        #                                                        )  # the type should be int, otherwise return "npoints > 0" error
        # fit_contours = [fit_contours]
        # fit_img = np.zeros(data.shape, np.uint8)
        # cv2.drawContours(fit_img, fit_contours, -1, (255), thickness=1)
        # cv2.imwrite('fit_img_ushape.png',fit_img)



    ###############################   hole  fit   ############################################################################################################
    count = 0
    Nonperfectushape_idx = Nonperfectcrs_idx[np.array(np.where(Ushape_IoU_list < (1-IoU_thres))).reshape(-1)]
    Hole_IoU_list = np.zeros(Nonperfectushape_idx.shape)
    for id in Nonperfectushape_idx:
        fit_hole = []
        IoU_hole = []
        # data = cv2.imread('tmp_img.bmp', 0)
        data = np.array(copy.deepcopy(org_data), dtype=np.uint8)
        centery, centerx, height, width, theta = rec_para_result[id]
        init_para = [centery, centerx, height, width, theta, width / 8.0, width / 8.0, height / 8.0,
                     height / 8.0]
        args = [data, 'hole']
        res0 = minimize(IoU, init_para, args=args, method='powell')
        fit_hole.append(res0)
        IoU_hole.append(1 - res0.fun)

        if res0.fun > IoU_thres:
            init_para = [centery, centerx, width, height, theta, width / 4.0, width / 4.0, height / 4.0,
                         height / 4.0]
            res1 = minimize(IoU, init_para, args=args, method='powell')
            fit_hole.append(res1)
            IoU_hole.append(1 - res1.fun)

        IoU_hole = np.array(IoU_hole).reshape(-1)
        maxid = np.argmax(IoU_hole)

        Hole_IoU_list[count] = IoU_hole[maxid]
        # print('Hole IoU:' + str(IoU_hole[maxid]))

        if 1 - best_fit[id][1].fun < IoU_hole[maxid]:
            best_fit[id][1] = fit_hole[maxid]
            best_fit[id][0] = 4

        count += 1
        # hole = make_hole(fit_hole[maxid].x)
        #
        # fit_contours = np.array(hole).reshape((8,  # number should change for more variant shapes
        #                                        1, 2)).astype(np.int
        #                                                      )  # the type should be int, otherwise return "npoints > 0" error
        # fit_img = np.zeros(data.shape, np.uint8)
        # inner_contours = [fit_contours[4:8]]
        # out_contours = [fit_contours[0:4]]
        # cv2.drawContours(fit_img, out_contours, -1, (255), thickness=-1)
        # cv2.drawContours(fit_img, inner_contours, -1, (0), thickness=-1)


    Nonperfecthole_idx = Nonperfectushape_idx[np.array(np.where(Hole_IoU_list < 0.9)).reshape(-1)]

    ##########################   best fit   #################################################################################################################
    IoU_best_list = []
    for id in best_fit:
        curr_id = id[0]
        curr_para = id[1].x
        IoU_best_list.append(1.0 - id[1].fun)

        # fit_img = np.zeros(data.shape, np.uint8)
        # curr_shape_func = make_obj(curr_id)
        # shape = curr_shape_func(curr_para)
        # fit_contours = np.array(shape).reshape((vec_length[curr_id],  # number should change for more variant shapes
        #                                         1, 2)).astype(np.int
        #                                                       )  # the type should be int, otherwise return "npoints > 0" error
        # if curr_id == 4:
        #     inner_contours = [fit_contours[4:8]]
        #     out_contours = [fit_contours[0:4]]
        #     cv2.drawContours(fit_img, out_contours, -1, (255), thickness=-1)
        #     cv2.drawContours(fit_img, inner_contours, -1, (0), thickness=-1)
        # else:
        #     fit_contours = [fit_contours]
        #     cv2.drawContours(fit_img, fit_contours, -1, (255), thickness=-1)
        #
        # cv2.imwrite('fit_img.png', fit_img)


        return curr_id + 1, IoU_best_list[0], curr_para[2], curr_para[3], curr_para[4]






