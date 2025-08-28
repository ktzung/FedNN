import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data


def resize_with_padding(im, desired_size, resample=Image.ANTIALIAS):
    old_size = im.size
    if old_size[0] == desired_size and old_size[1] == desired_size:
        return im
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    if old_size[0] != new_size[0] or old_size[1] != new_size[1]:
        im = im.resize(new_size, resample)
    if new_size[0] == desired_size and new_size[1] == desired_size:
        return im
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


class DatasetObject:
    def __init__(self, dataset, n_client, seed, result_path='', data_dir='.tmp/data', distribution_alpha=0.6):
        self.dataset = dataset
        self.n_client = n_client
        self.seed = seed
        self.data_dir = data_dir
        self.name = "%s_%d_%d" % (self.dataset, self.n_client, self.seed)  # concept drift
        self.result_path = result_path
        self.distribution_alpha = distribution_alpha
        self.set_data()

    def set_data(self):
        if self.dataset in ['CIFAR10-C', 'DIGIT', 'CIFAR10']:
            self.channels = 3
            self.width = 32
            self.height = 32
            self.n_cls = 10
        elif self.dataset == 'FAIRFACE':
            self.channels = 3
            self.width = 224
            self.height = 224
            self.n_cls = 2
        elif self.dataset == 'PACS':
            self.channels = 3
            self.width = 224
            self.height = 224
            self.n_cls = 7
        elif self.dataset == 'OFFICE-HOME':
            self.channels = 3
            self.width = 224
            self.height = 224
            self.n_cls = 65
        elif self.dataset == 'VLCS':
            self.channels = 3
            self.width = 224
            self.height = 224
            self.n_cls = 5
        elif self.dataset == 'DomainNet':
            self.channels = 3
            self.width = 224
            self.height = 224
            self.n_cls = 345

        clnt_tst_x = None
        clnt_tst_y = None

        if self.dataset == 'CIFAR10-C':
            dir_name = 'CIFAR-10-C'
            # Tuy tung loai Anh, nghien cuu phuong phap bien doi khac nhau
            curruption_name_list = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
            label_name = 'labels'
            num_samples = 10000  # with 5 strength
            sample_per_client_and_test = num_samples // (len(curruption_name_list) + 1)
            percent = 1.0  # TODO:

            mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape((1, 1, 1, 3))
            std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape((1, 1, 1, 3))

            if self.n_client != len(curruption_name_list):
                raise ValueError()

            clnt_x = []
            clnt_y = []
            for _ in range(self.n_client):
                clnt_x.append(list())
                clnt_y.append(list())

            clnt_tst_x = []
            clnt_tst_y = []
            for _ in range(self.n_client):
                clnt_tst_x.append(list())
                clnt_tst_y.append(list())

            tst_x = []
            tst_y = []

            labels = np.load(os.path.join(self.data_dir, dir_name, f'{label_name}.npy'))

            for client_idx, curruption_name in enumerate(curruption_name_list):
                npy_path = os.path.join(self.data_dir, dir_name, f'{curruption_name}.npy')
                data = np.load(npy_path)
                data = data.astype(np.float32) / 255
                data = (data - mean) / std
                data = data.transpose((0, 3, 1, 2))
                for strength_idx in [3]:  # TODO: 0 ~ 5
                    # for tst
                    str_idx = num_samples * strength_idx  # client_idx=0
                    tst_x.extend(data[str_idx:str_idx + sample_per_client_and_test])
                    tst_y.extend(labels[str_idx:str_idx + sample_per_client_and_test])
                    clnt_tst_x[client_idx].extend(data[str_idx:str_idx + sample_per_client_and_test])
                    clnt_tst_y[client_idx].extend(labels[str_idx:str_idx + sample_per_client_and_test])

                    # for clnt
                    str_idx = (num_samples * strength_idx) + ((client_idx + 1) * sample_per_client_and_test)
                    clnt_x[client_idx].extend(data[str_idx:str_idx + sample_per_client_and_test])
                    clnt_y[client_idx].extend(labels[str_idx:str_idx + sample_per_client_and_test])

            # reduce samples (based on percent)
            for client_idx in range(self.n_client):
                clnt_x[client_idx] = clnt_x[client_idx][:int(len(clnt_x[client_idx]) * percent)]
                clnt_y[client_idx] = clnt_y[client_idx][:int(len(clnt_y[client_idx]) * percent)]

            for client_idx in range(self.n_client):
                clnt_x[client_idx] = np.asarray(clnt_x[client_idx])
                clnt_y[client_idx] = np.asarray(clnt_y[client_idx])
                clnt_tst_x[client_idx] = np.asarray(clnt_tst_x[client_idx])
                clnt_tst_y[client_idx] = np.asarray(clnt_tst_y[client_idx])

            # clnt_x = np.asarray(clnt_x)
            # clnt_y = np.asarray(clnt_y)
            # clnt_tst_x = np.asarray(clnt_tst_x)
            # clnt_tst_y = np.asarray(clnt_tst_y)
            # tst_x = np.asarray(tst_x)
            # tst_y = np.asarray(tst_y)

        elif self.dataset == 'DIGIT':
            dir_name = 'digit_dataset'
            curruption_name_list = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']

            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 1, 3))
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 1, 3))

            if self.n_client != len(curruption_name_list):
                raise ValueError()

            clnt_x = []
            clnt_y = []
            for _ in range(self.n_client):
                clnt_x.append(list())
                clnt_y.append(list())

            clnt_tst_x = []
            clnt_tst_y = []
            for _ in range(self.n_client):
                clnt_tst_x.append(list())
                clnt_tst_y.append(list())

            tst_x = []
            tst_y = []

            for client_idx, curruption_name in enumerate(curruption_name_list):
                npy_path = os.path.join(self.data_dir, dir_name, curruption_name, 'test.pkl')
                data, label = np.load(npy_path, allow_pickle=True)

                _data = []
                for img in data:
                    img = Image.fromarray(img).convert('RGB').resize((32, 32))
                    img = np.array(img)
                    _data.append(img)
                data = np.asarray(_data)

                data = data.astype(np.float32) / 255
                data = (data - mean) / std
                data = data.transpose((0, 3, 1, 2))
                tst_x.extend(data)
                tst_y.extend(label)
                clnt_tst_x[client_idx].extend(data)
                clnt_tst_y[client_idx].extend(label)

            for client_idx, curruption_name in enumerate(curruption_name_list):
                data, label = [], []
                for i in range(1):  # TODO: 1 ~ 10
                    npy_path = os.path.join(self.data_dir, dir_name, curruption_name, 'partitions', f'train_part{i}.pkl')
                    data_part, label_part = np.load(npy_path, allow_pickle=True)
                    data.extend(data_part)
                    label.extend(label_part)

                data = data[:(len(data) // 2)]
                label = label[:(len(label) // 2)]

                _data = []
                for img in data:
                    img = Image.fromarray(img).convert('RGB').resize((32, 32))
                    img = np.array(img)
                    _data.append(img)
                data = np.asarray(_data)

                data = data.astype(np.float32) / 255
                data = (data - mean) / std
                data = data.transpose((0, 3, 1, 2))
                clnt_x[client_idx].extend(data)
                clnt_y[client_idx].extend(label)

            for client_idx in range(self.n_client):
                clnt_x[client_idx] = np.asarray(clnt_x[client_idx])
                clnt_y[client_idx] = np.asarray(clnt_y[client_idx])
                clnt_tst_x[client_idx] = np.asarray(clnt_tst_x[client_idx])
                clnt_tst_y[client_idx] = np.asarray(clnt_tst_y[client_idx])

            # clnt_x = np.asarray(clnt_x)
            # clnt_y = np.asarray(clnt_y)
            # clnt_tst_x = np.asarray(clnt_tst_x)
            # clnt_tst_y = np.asarray(clnt_tst_y)
            # tst_x = np.asarray(tst_x)
            # tst_y = np.asarray(tst_y)

        elif self.dataset == 'FAIRFACE':
            dir_name = 'fairface'
            curruption_name_list = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
            label_name_list = ['Male', 'Female']
            percent = 0.1  # TODO:

            if self.n_client != len(curruption_name_list):
                raise ValueError()

            clnt_x = []
            clnt_y = []
            for _ in range(self.n_client):
                clnt_x.append(list())
                clnt_y.append(list())

            clnt_tst_x = []
            clnt_tst_y = []
            for _ in range(self.n_client):
                clnt_tst_x.append(list())
                clnt_tst_y.append(list())

            tst_x = []
            tst_y = []

            def read_csv(csv_path):
                data_list = []
                with open(csv_path) as f:
                    f.readline()  # remove head
                    while True:
                        l = f.readline()
                        if len(l) == 0:
                            break
                        l_split = l.strip().split(',')

                        file = l_split[0]
                        gender_idx = label_name_list.index(l_split[2])
                        race_idx = curruption_name_list.index(l_split[3])
                        data_list.append((file, gender_idx, race_idx))
                return data_list

            val_csv_path = os.path.join(self.data_dir, dir_name, 'fairface_label_val.csv')
            train_csv_path = os.path.join(self.data_dir, dir_name, 'fairface_label_train.csv')
            img_dir = os.path.join(self.data_dir, dir_name, 'fairface-img-margin025-trainval')

            val_label_list = read_csv(val_csv_path)
            train_label_list = read_csv(train_csv_path)

            random.Random(1).shuffle(train_label_list)
            train_label_list = train_label_list[:int(len(train_label_list) * percent)]

            for (file, gender, race) in val_label_list:
                img_path = os.path.join(img_dir, file)
                img = Image.open(img_path).convert('RGB')
                tst_x.append(img)
                tst_y.append(gender)
                clnt_tst_x[race].append(img)
                clnt_tst_y[race].append(gender)

            for (file, gender, race) in train_label_list:
                img_path = os.path.join(img_dir, file)
                img = Image.open(img_path).convert('RGB')
                clnt_x[race].append(img)
                clnt_y[race].append(gender)

            for client_idx in range(self.n_client):
                clnt_y[client_idx] = np.asarray(clnt_y[client_idx])
                clnt_tst_y[client_idx] = np.asarray(clnt_tst_y[client_idx])

            # clnt_y = np.asarray(clnt_y)
            # clnt_tst_y = np.asarray(clnt_tst_y)
            # tst_y = np.asarray(tst_y)

        elif self.dataset in ['PACS', 'OFFICE-HOME', 'VLCS', 'DomainNet']:
            if self.dataset == 'PACS':
                dir_name = 'pacs'
                curruption_name_list = ['art_painting', 'cartoon', 'photo', 'sketch']
                label_name_list = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
            elif self.dataset == 'OFFICE-HOME':
                dir_name = 'off_home'
                curruption_name_list = ['Art', 'Clipart', 'Product', 'Real_World']
                label_name_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
            elif self.dataset == 'VLCS':
                dir_name = 'vlcs'
                curruption_name_list = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
                label_name_list = ['bird', 'car', 'chair', 'dog', 'person']
            elif self.dataset == 'DomainNet':
                dir_name = 'domain_net_256'
                curruption_name_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                label_name_list = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon',
                                   'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter',
                                   'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark',
                                   'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

                dataset_path = os.path.join(self.data_dir, dir_name)
                train_txt_path_list = [os.path.join(dataset_path, f) for f in sorted(os.listdir(dataset_path)) if f.endswith('_train.txt')]
                test_txt_path_list = [os.path.join(dataset_path, f) for f in sorted(os.listdir(dataset_path)) if f.endswith('_test.txt')]

                def read_filename_from_txt(txt_path):
                    data = []
                    with open(txt_path) as f:
                        while True:
                            line = f.readline()
                            if len(line) == 0:
                                break
                            data.append(os.path.basename(line.split(' ')[0]))
                    return data

                train_filename_list, test_filename_list = [], []
                for txt_path in train_txt_path_list:
                    train_filename_list.extend(read_filename_from_txt(txt_path))

                for txt_path in test_txt_path_list:
                    test_filename_list.extend(read_filename_from_txt(txt_path))

            val_rate = 0.1

            if self.n_client != len(curruption_name_list):
                raise ValueError()

            clnt_x = []
            clnt_y = []
            for _ in range(self.n_client):
                clnt_x.append(list())
                clnt_y.append(list())

            clnt_tst_x = []
            clnt_tst_y = []
            for _ in range(self.n_client):
                clnt_tst_x.append(list())
                clnt_tst_y.append(list())

            tst_x = []
            tst_y = []

            for client_idx, curruption_name in enumerate(curruption_name_list):
                for label_idx, label in enumerate(label_name_list):
                    img_dir = os.path.join(self.data_dir, dir_name, curruption_name, label)
                    img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.splitext(f)[-1] in ['.jpg', '.png']])
                    random.Random(1).shuffle(img_list)

                    if self.dataset == 'DomainNet':
                        train_img_list, val_img_list = [], []
                        for f in img_list:
                            filename = os.path.basename(f)
                            if filename in test_filename_list:
                                val_img_list.append(f)
                            else:
                                train_img_list.append(f)
                    else:
                        _img_list = []
                        for f in img_list:
                            try:
                                img = Image.open(f).convert('RGB')
                            except:
                                print('error skip:', f)
                                continue
                            _img_list.append(img)
                        img_list = _img_list
                        img_list = [resize_with_padding(im, 256) for im in img_list]

                        train_img_list = img_list[int(len(img_list) * val_rate):]
                        val_img_list = img_list[:int(len(img_list) * val_rate)]

                    clnt_x[client_idx].extend(train_img_list)
                    clnt_y[client_idx].extend([label_idx] * len(train_img_list))
                    tst_x.extend(val_img_list)
                    tst_y.extend([label_idx] * len(val_img_list))
                    clnt_tst_x[client_idx].extend(val_img_list)
                    clnt_tst_y[client_idx].extend([label_idx] * len(val_img_list))

            for client_idx in range(self.n_client):
                clnt_y[client_idx] = np.asarray(clnt_y[client_idx])
                clnt_tst_y[client_idx] = np.asarray(clnt_tst_y[client_idx])

            # clnt_y = np.asarray(clnt_y)
            # clnt_tst_y = np.asarray(clnt_tst_y)
            # tst_y = np.asarray(tst_y)
        elif self.dataset == 'CIFAR10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

            trnset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
            tstset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform)

            trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
            tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)

            trn_itr = trn_load.__iter__()
            tst_itr = tst_load.__iter__()
            # labels are of shape (n_data,)
            trn_x, trn_y = trn_itr.__next__()
            tst_x, tst_y = tst_itr.__next__()

            trn_x = trn_x.numpy()
            trn_y = trn_y.numpy().reshape(-1, 1)
            tst_x = tst_x.numpy()
            tst_y = tst_y.numpy().reshape(-1, 1)

            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            # Draw from lognormal distribution
            clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0.0, size=self.n_client))
            clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)

            # Add/Subtract the excess number starting from first client
            if diff != 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            ###

            # Drichlet
            cls_priors = np.random.dirichlet(alpha=[self.distribution_alpha] * self.n_cls, size=self.n_client)
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
            cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

            clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client)]
            clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

            while (np.sum(clnt_data_list) != 0):
                curr_clnt = np.random.randint(self.n_client)
                # If current node is full resample a client
                # print('Remaining Data: %d' % np.sum(clnt_data_list))
                if clnt_data_list[curr_clnt] <= 0:
                    continue
                clnt_data_list[curr_clnt] -= 1
                curr_prior = prior_cumsum[curr_clnt]
                while True:
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    # Redraw class label if trn_y is out of that class
                    if cls_amount[cls_label] <= 0:
                        continue
                    cls_amount[cls_label] -= 1

                    clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                    clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                    break

            # clnt_x = np.asarray(clnt_x)
            # clnt_y = np.asarray(clnt_y)

            cls_means = np.zeros((self.n_client, self.n_cls))
            for clnt in range(self.n_client):
                for cls in range(self.n_cls):
                    cls_means[clnt, cls] = np.mean(clnt_y[clnt] == cls)
            prior_real_diff = np.abs(cls_means - cls_priors)
            print('--- Max deviation from prior: %.4f' % np.max(prior_real_diff))
            print('--- Min deviation from prior: %.4f' % np.min(prior_real_diff))

        self.clnt_x = clnt_x
        self.clnt_y = clnt_y
        self.clnt_tst_x = clnt_tst_x  # TODO: temporal use for personalization
        self.clnt_tst_y = clnt_tst_y  # TODO: temporal use for personalization
        self.tst_x = tst_x
        self.tst_y = tst_y

        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % len(self.clnt_y[clnt]))
            count += len(self.clnt_y[clnt])

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % len(self.tst_y))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        data_y = np.array(data_y)  # convert to numpy
        if self.name == 'DIGIT':
            self.X_data = torch.tensor(np.array(data_x)).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name in ['CIFAR10-C', 'CIFAR10']:
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        elif self.name == 'FAIRFACE':
            self.train = train

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform_train = transforms.Compose([
                transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            self.transform_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        elif self.name in ['PACS', 'OFFICE-HOME', 'VLCS', 'DomainNet']:
            self.train = train

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            self.transform_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

            self.X_data = data_x
            self.y_data = data_y
            # if not isinstance(data_y, bool):
            #     self.y_data = data_y.astype('float32')

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'DIGIT':
            X = self.X_data[idx, :]
            y = self.y_data[idx]
            return X, y

        elif self.name in ['CIFAR10-C', 'CIFAR10']:
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if (np.random.rand() > .5):
                    # Random cropping
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            y = self.y_data[idx]
            return img, y

        elif self.name in ['FAIRFACE', 'PACS', 'OFFICE-HOME', 'VLCS']:
            img = self.X_data[idx]
            y = self.y_data[idx]
            if self.train:
                img = self.transform_train(img)
            else:
                img = self.transform_val(img)
            return img, y
        elif self.name == 'DomainNet':
            img = Image.open(self.X_data[idx]).convert('RGB')
            y = self.y_data[idx]
            if self.train:
                img = self.transform_train(img)
            else:
                img = self.transform_val(img)
            return img, y
