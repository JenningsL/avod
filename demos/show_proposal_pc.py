import sys
import os
import numpy as np
import mayavi.mlab

from avod.core import box_3d_encoder
from wavedata.tools.obj_detection import obj_utils
sys.path.append('./demo')
from viz_util import draw_lidar, draw_gt_boxes3d
import kitti_util


def load_proposals(frame_id):
    rpn_score_threshold = 0.5
    proposals_file_path = './kitti/proposal_120000/{0}.txt'.format(frame_id)
    proposals_and_scores = np.loadtxt(proposals_file_path)
    proposal_boxes_3d = proposals_and_scores[:, 0:7]
    proposal_scores = proposals_and_scores[:, 7]

    # Apply score mask to proposals
    score_mask = proposal_scores > rpn_score_threshold
    # 3D box in the format [x, y, z, l, w, h, ry]
    proposal_boxes_3d = proposal_boxes_3d[score_mask]
    proposal_scores = proposal_scores[score_mask]
    proposal_objs = \
        [box_3d_encoder.box_3d_to_object_label(proposal,
                                               obj_type='Proposal')
         for proposal in proposal_boxes_3d]
    for obj, score in zip(proposal_objs, proposal_scores):
        obj.score = score
    return proposal_objs

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(boxes[:,4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # WARNING: (x1, y1) must be the relatively small point
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

if __name__ == '__main__':
    frame_id = sys.argv[1]

    pointcloud_file_path = './kitti/velodyne/{0}.bin'.format(frame_id)
    with open(pointcloud_file_path, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    points = data_array.reshape(-1, 4)

    calib_filename = os.path.join('./kitti/calib/', '{0}.txt'.format(frame_id))
    calib = kitti_util.Calibration(calib_filename)

    fig = draw_lidar(points)

    # ground truth
    obj_labels = obj_utils.read_labels('./kitti/label_2', int(frame_id))
    gt_boxes = []
    for obj in obj_labels:
        if obj.type not in ['Car']:
            continue
        _, corners = kitti_util.compute_box_3d(obj, calib.P)
        corners_velo = calib.project_rect_to_velo(corners)
        gt_boxes.append(corners_velo)
    fig = draw_gt_boxes3d(gt_boxes, fig, color=(1, 0, 0))

    # proposals
    proposal_objs = load_proposals(frame_id)
    boxes = []
    box_scores = []
    for obj in proposal_objs:
        _, corners = kitti_util.compute_box_3d(obj, calib.P)
        corners_velo = calib.project_rect_to_velo(corners)
        boxes.append(corners_velo)
        box_scores.append(obj.score)

    bev_boxes = list(map(lambda bs: [bs[0][1][0], bs[0][1][1], bs[0][3][0], bs[0][3][1], bs[1]], zip(boxes, box_scores)))
    bev_boxes = np.array(bev_boxes)
    print('before nms: {0}'.format(len(bev_boxes)))
    nms_idxs = non_max_suppression(bev_boxes, 0.3)
    print('after nms: {0}'.format(len(nms_idxs)))
    boxes = [boxes[i] for i in nms_idxs]
    # fig = draw_lidar(np.array(boxes)[:, 1], fig=fig, pts_color=(1, 0, 0), pts_scale=0.1, pts_mode='sphere')
    # fig = draw_lidar(np.array(boxes)[:, 3], fig=fig, pts_color=(0, 1, 0), pts_scale=0.1, pts_mode='sphere')
    fig = draw_gt_boxes3d(boxes, fig, draw_text=False)
    mayavi.mlab.show()
