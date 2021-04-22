import json
import os

split = 'val' # 'train' 'val'
input_path = 'your path'
out_path = 'your path'
if __name__ == '__main__':

    f = json.load(open(input_path + '/{}.json'.format(split)))  # f[]dict_keys(['info', 'licenses', 'videos', 'categories', 'annotations'])
    out = {'images': [], 'annotations': [],
           'categories': [], 'videos': []}

    for cat in f['categories']:
        out['categories'].append({'id': cat['id'], 'name': cat['name']})

    image_cnt = 0
    ann_cnt = 0
    global_track_id = {}
    image_id_map = {}
    for v_id, video in enumerate(f['videos']):
        out['videos'].append({
            'id': v_id+1,
            'file_name': video['file_names'][0].split('/')[0],
            'video_len': video['length']})

        img_width = video['width']
        img_height = video['height']

        for i in range(video['length']):
            image_info = {'file_name': video['file_names'][i],
                          'id': image_cnt + i + 1,
                          'height': img_height,
                          'width': img_width,
                          'frame_id': i + 1,
                          'prev_image_id': image_cnt + i if i > 0 else -1,
                          'next_image_id': image_cnt + i + 2 if i < video['length'] - 1 else -1,
                          'video_id': v_id+1}
            image_id_map['{}_{}'.format(image_info['video_id'], image_info['frame_id'])] = image_info['id']
            out['images'].append(image_info)
        print('{}: {} images'.format(v_id, video['length']))
        image_cnt += video['length']

    if 'annotations' in f:
        for item in f['annotations']:
            track_id = item['id']

            for frame_id, bbox in enumerate(item['bboxes']):
                if bbox is None:
                    continue
                ann_cnt += 1
                ann = {'id': ann_cnt,
                       'category_id': item['category_id'],
                       'image_id': image_id_map['{}_{}'.format(item['video_id'], frame_id+1)],
                       'track_id': track_id,
                       'bbox': bbox,  # x1, y1, w, h
                       'segmentation': item['segmentations'][frame_id],
                       'global_track_id': item['id']}
                out['annotations'].append(ann)

    print('total image #: {}'.format(image_cnt))
    json.dump(out, open(out_path + '{}.json'.format(split), 'w'))

