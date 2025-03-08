# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--frame-skip', type=int, default=3, help='process every Nth frame')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    camera = cv2.VideoCapture(args.camera_id)
    
    # Initialize frame counter and last processed result
    frame_count = 0
    last_result = None
    last_processed_img = None

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        if not ret_val:
            break
            
        # Process only every Nth frame, where N is frame_skip
        if frame_count % args.frame_skip == 0:
            # Process the frame with the model
            result = inference_detector(model, img)
            last_result = result
            
            # Convert image for visualization
            rgb_img = mmcv.imconvert(img, 'bgr', 'rgb')
            visualizer.add_datasample(
                name='result',
                image=rgb_img,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=args.score_thr,
                show=False)
                
            vis_img = visualizer.get_image()
            last_processed_img = mmcv.imconvert(vis_img, 'bgr', 'rgb')
            
        elif last_processed_img is not None:
            # For skipped frames, just show the last processed result
            # This gives a more consistent visualization experience
            pass
        else:
            # If we haven't processed any frames yet, just convert the image
            rgb_img = mmcv.imconvert(img, 'bgr', 'rgb')
            visualizer.add_datasample(
                name='result',
                image=rgb_img,
                data_sample=None,
                draw_gt=False,
                show=False)
            last_processed_img = mmcv.imconvert(visualizer.get_image(), 'bgr', 'rgb')
        
        # Always show the latest processed image
        cv2.imshow('result', last_processed_img)
        
        # Increment frame counter
        frame_count += 1

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()