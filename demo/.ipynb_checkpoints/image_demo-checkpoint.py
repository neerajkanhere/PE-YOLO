import asyncio
from argparse import ArgumentParser
import os
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument(
        '--out-dir', default='vis', help='out dir')
    args = parser.parse_args()
    return args


def main(args):
    if os.path.isfile(args.img):
        # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint, device=args.device)
        # test a single image
        result = inference_detector(model, args.img)
        # show the results
        show_result_pyplot(model, args.img, result, score_thr=args.score_thr, out_dir=os.path.join(args.out_dir, args.img.split('/')[-1]))
    else:
        model = init_detector(args.config, args.checkpoint, device=args.device)
        for i in os.listdir(args.img):
            if not os.path.isfile(args.img):
                continue
            image = os.path.join(args.img, i)
            result = inference_detector(model, image)

            # show the results
            show_result_pyplot(model, image, result, score_thr=args.score_thr, out_dir=os.path.join(args.out_dir, i))


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
   
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
