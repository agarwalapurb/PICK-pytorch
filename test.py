# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            # For easier debug.
            image_names = input_data_item["filenames"]

            output = pick_model(**input_data_item)
            print("\nPrinting output...\n")
            print(output)
            logits = output['logits']  # (B, N*T, out_dim)
            print("\nPrinting logits...\n")
            print(logits)
            new_mask = output['new_mask']
            print("\nPrinting newmask...\n")
            print(new_mask)
            image_indexs = input_data_item['image_indexs']  # (B,)
            print("\nPrinting imageindexs...\n")
            print(image_indexs)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            print("\nPrinting text_segments...\n")
            print(text_segments)
            mask = input_data_item['mask']
            print("\nPrinting mask...\n")
            print(mask)
            mask1 = torch.squeeze(mask)
            plt.imsave("test.png",mask1.cpu(),cmap="gray",vmin=0,vmax=1)
            plt.imshow(mask1.cpu(),cmap="gray",vmin=0,vmax=1)
            plt.show()
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            print("\nPrinting best paths...\n")
            print(best_paths)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)
            print("\nPrinting predicted tags...\n")         
            print(predicted_tags)
            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            print("\nPrinting decoded tags list...\n")
            print(decoded_tags_list)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)
            print("\nPrinting decoded texts list...\n")
            print(decoded_texts_list)
            print("\nPrinting decoded_tags, decoded_texts, image_index ...\n")
            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                print(decoded_tags, decoded_texts, image_index)
                print("\nPrinting spans ...\n")
                spans = bio_tags_to_spans(decoded_tags, [])
                print(spans)
                print("\n")
                spans = sorted(spans, key=lambda x: x[1][0])
                print(spans)
                print("\n")

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)
                print("\nPrinting entities ...\n")
                print(entities)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()
    main(args)
