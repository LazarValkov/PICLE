from typing import Union
import numpy as np
import random
from random import randint
import pickle
import abc
import urllib.request
import zipfile, gzip
import copy
import torchvision

from skimage.transform import rotate
import os

ROOT_IMG_DATA_DIRECTORY = "Experiments/BELL/Data/ImgData"
ROOT_IMG_DATA_CACHE_DIRECTORY = "Experiments/BELL/Data/ImgData/prepared_cache"


class ImageDatasetProvider(object):
    IMG_SIZE = 28
    CH_NUM = 1
    NUM_CLASSES_PER_PROBLEM = 8

    def __init__(self, class_portion: int, get_test=True, normalize_input=True,
                 num_tr_datapoints: Union[int, None] = None,
                 tr_shuffle_seed: Union[int, None] = None,
                 flatten_inputs=False
                 ):
        """
        :param class_portion > 0, decides which portion of classes to use
        :param get_test: If true, loads the test dataset as well as the training one
        """
        total_num_classes = len(self._get_full_class_map().keys())
        total_num_class_portions = total_num_classes // ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM
        assert class_portion > 0
        assert class_portion <= total_num_class_portions
        self.class_portion = class_portion
        self.flatten_inputs = flatten_inputs

        self.ds_images_tr = {
            "sorted_images": None,
            "sorted_labels": None,
            "labels_count": None,
            "labels_indices": None,
            "num_labels": None
        }
        self.ds_images_val = {
            "sorted_images": None,
            "sorted_labels": None,
            "labels_count": None,
            "labels_indices": None,
            "num_labels": None
        }
        self.ds_images_test = {
            "sorted_images": None,
            "sorted_labels": None,
            "labels_count": None,
            "labels_indices": None,
            "num_labels": None
        }
        self.mean = None
        self.std = None

        self.get_test = get_test
        self.normalize_input = normalize_input

        if not self.try_load_prepared_data():
            self.prepare_image_data()
            self.save_prepared_data()

        if self.flatten_inputs:
            flattened_dim = np.prod(self.ds_images_tr["sorted_images"].shape[1:]).item()
            self.ds_images_tr["sorted_images"] = self.ds_images_tr["sorted_images"].reshape((-1, flattened_dim))
            self.ds_images_val["sorted_images"] = self.ds_images_val["sorted_images"].reshape((-1, flattened_dim))
            self.ds_images_test["sorted_images"] = self.ds_images_test["sorted_images"].reshape((-1, flattened_dim))

        if num_tr_datapoints is not None:
            assert num_tr_datapoints <= self.ds_images_tr['sorted_images'].shape[0]
            assert tr_shuffle_seed is not None
            c_random_obj = random.Random(tr_shuffle_seed)

            num_tr_datapoints_per_class = num_tr_datapoints // ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM
            tr_inputs_list, tr_labels_list = [], []

            for c_label, c_indices in self.ds_images_tr['labels_indices'].items():
                c_selected_indices = c_random_obj.sample(range(c_indices['first'], c_indices['last']+1),
                                                   num_tr_datapoints_per_class)
                c_selected_inputs = self.ds_images_tr['sorted_images'][c_selected_indices]
                c_selected_labels = self.ds_images_tr['sorted_labels'][c_selected_indices]

                tr_inputs_list.append(c_selected_inputs)
                tr_labels_list.append(c_selected_labels)

            tr_inputs_np = np.vstack(tr_inputs_list)
            tr_labels_np = np.hstack(tr_labels_list)

            #tr_inputs_np, tr_labels_np = self.ds_images_tr['sorted_images'], self.ds_images_tr['sorted_labels']
            #self.shuffle_in_unison(tr_inputs_np, tr_labels_np, rndm_seed=tr_shuffle_seed)
            #tr_inputs_np = tr_inputs_np[:num_tr_datapoints]
            #tr_labels_np = tr_labels_np[:num_tr_datapoints]

            self.ds_images_tr = {}
            self.sort_data_into_dict(self.ds_images_tr, tr_inputs_np, tr_labels_np)

    def get_num_classes(self):
        return self.ds_images_tr["num_labels"]

    @staticmethod
    def get_image_size():
        return ImageDatasetProvider.IMG_SIZE

    @staticmethod
    def get_image_channels_num():
        return ImageDatasetProvider.CH_NUM

    def try_load_prepared_data(self):
        # return True if loaded, False otherwise

        # load the data
        save_folder = f"{ROOT_IMG_DATA_CACHE_DIRECTORY}/{self.get_name()}"
        # save_folder = f"{ROOT_IMG_DATA_DIRECTORY}/prepared_cache/{self.get_name()}"
        main_dict_filepath = f"{save_folder}/main_dict.pickle"

        if not os.path.exists(main_dict_filepath):
            return False

        with open(main_dict_filepath, 'rb') as fh:
            main_dict = pickle.load(fh)

        try:
            # assert the stored value of normalize input is the same
            assert self.normalize_input == main_dict["normalize_input"]
            self.mean = main_dict["normalize_input"]
            self.std = main_dict["std"]

            self.ds_images_tr = main_dict["ds_images_tr"]
            self.ds_images_tr["sorted_images"] = np.load(f"{save_folder}/ds_images_tr_sorted_images.npy")
            self.ds_images_tr["sorted_labels"] = np.load(f"{save_folder}/ds_images_tr_sorted_labels.npy")

            self.ds_images_val = main_dict["ds_images_val"]
            self.ds_images_val["sorted_images"] = np.load(f"{save_folder}/ds_images_val_sorted_images.npy")
            self.ds_images_val["sorted_labels"] = np.load(f"{save_folder}/ds_images_val_sorted_labels.npy")

            self.ds_images_test = main_dict["ds_images_test"]
            self.ds_images_test["sorted_images"] = np.load(f"{save_folder}/ds_images_test_sorted_images.npy")
            self.ds_images_test["sorted_labels"] = np.load(f"{save_folder}/ds_images_test_sorted_labels.npy")

        except Exception as e:
            raise NotImplementedError()
            # return False

        return True

    def save_prepared_data(self):
        # save the numpy arrays separately
        # save 1 big dictionary to hold everything else

        # save_folder = f"{ROOT_IMG_DATA_DIRECTORY}/prepared_cache/{self.get_name()}"
        save_folder = f"{ROOT_IMG_DATA_CACHE_DIRECTORY}/{self.get_name()}"
        os.makedirs(save_folder, exist_ok=True)

        main_dict_filepath = f"{save_folder}/main_dict.pickle"

        main_dict = {
            "normalize_input": self.normalize_input,
            "mean": self.mean,
            "std": self.std
        }
        # process ds_images_tr
        c_ds_images_tr = copy.copy(self.ds_images_tr)  # creating a shallow copy
        np.save(f"{save_folder}/ds_images_tr_sorted_images.npy", c_ds_images_tr["sorted_images"])
        np.save(f"{save_folder}/ds_images_tr_sorted_labels.npy", c_ds_images_tr["sorted_labels"])
        c_ds_images_tr["sorted_images"] = None
        c_ds_images_tr["sorted_labels"] = None
        main_dict["ds_images_tr"] = c_ds_images_tr

        c_ds_images_val = copy.copy(self.ds_images_val)  # creating a shallow copy
        np.save(f"{save_folder}/ds_images_val_sorted_images.npy", c_ds_images_val["sorted_images"])
        np.save(f"{save_folder}/ds_images_val_sorted_labels.npy", c_ds_images_val["sorted_labels"])
        c_ds_images_val["sorted_images"] = None
        c_ds_images_val["sorted_labels"] = None
        main_dict["ds_images_val"] = c_ds_images_val

        c_ds_images_test = copy.copy(self.ds_images_test)  # creating a shallow copy
        np.save(f"{save_folder}/ds_images_test_sorted_images.npy", c_ds_images_test["sorted_images"])
        np.save(f"{save_folder}/ds_images_test_sorted_labels.npy", c_ds_images_test["sorted_labels"])
        c_ds_images_test["sorted_images"] = None
        c_ds_images_test["sorted_labels"] = None
        main_dict["ds_images_test"] = c_ds_images_test

        # save the main dictionary
        with open(main_dict_filepath, 'wb') as f:
            pickle.dump(main_dict, f, pickle.HIGHEST_PROTOCOL)

    @abc.abstractmethod
    def get_name(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _get_full_class_map():
        # {original_class_id: new_class_id}
        # can be used to omit classes with a few data points
        pass

    @staticmethod
    def shuffle_in_unison(a, b, rndm_seed=123):
        rng_state1 = np.random.RandomState(seed=rndm_seed)
        rng_state1.shuffle(a)
        rng_state2 = np.random.RandomState(seed=rndm_seed)
        rng_state2.shuffle(b)

    def sort_data_into_dict(self, dictionary, np_images, np_labels):
        # split into 10 data-label pairs, 1 pair for each digit
        labels_sorted_order = np.argsort(np_labels, axis=0).reshape((-1,))

        dictionary['sorted_labels'] = np_labels[labels_sorted_order]
        dictionary['sorted_images'] = np_images[labels_sorted_order]

        items_unique, items_counts = np.unique(np_labels, return_counts=True)
        dictionary['num_labels'] = items_unique.size
        dictionary['labels_count'] = dict(zip(items_unique, items_counts))
        dictionary['labels_indices'] = {0: {'first': 0, 'last': items_counts[0] - 1}}

        for i in range(1, dictionary['num_labels']):
            first_indx = dictionary['labels_indices'][i - 1]['last'] + 1
            dictionary['labels_indices'][i] = {'first': first_indx, 'last': first_indx + items_counts[i] - 1}

        return dictionary

    def prepare_image_data(self):
        """
        Prepares the images, used for subsequent tasks.
        :return:
        """
        if self.get_test:
            tr_images, tr_labels, val_images, val_labels, test_images, test_labels = self.load_data()
        else:
            tr_images, tr_labels, val_images, val_labels, = self.load_data()

        # ImageDatasetProvider.shuffle_in_unison(train_all_images, train_all_labels)

        if self.normalise_all_together():
            self.mean = tr_images.mean()
            self.std = tr_images.std()
        else:
            assert len(tr_images.shape) == 2
            self.mean = tr_images.mean(axis=0)
            self.std = tr_images.std(axis=0)
        if self.normalize_input:
            tr_images = (tr_images - self.mean) / self.std
            val_images = (val_images - self.mean) / self.std
            if self.get_test:
                test_images = (test_images - self.mean) / self.std

        self.sort_data_into_dict(self.ds_images_tr, tr_images, tr_labels)
        self.sort_data_into_dict(self.ds_images_val, val_images, val_labels)
        if self.get_test:
            self.sort_data_into_dict(self.ds_images_test, test_images, test_labels)

    @staticmethod
    def get_random_image_w_replacement(class_id, image_dictionary, rndm_obj: Union[None, random.Random]=None):
        first_index = image_dictionary['labels_indices'][class_id]['first']
        last_index = image_dictionary['labels_indices'][class_id]['last']

        if rndm_obj is not None:
            indx = rndm_obj.randint(first_index, last_index)
        else:
            indx = randint(first_index, last_index)

        return image_dictionary['sorted_images'][indx]

    @staticmethod
    def state_for_random_sampling_reset_values_for_class_id(image_dictionary, dict_class_to_indices_left, class_id):
        c_first_index = image_dictionary['labels_indices'][class_id]['first']
        c_last_index = image_dictionary['labels_indices'][class_id]['last']
        dict_class_to_indices_left[class_id] = list(range(c_first_index, c_last_index + 1))

    @classmethod
    def get_random_image_without_replacement(cls, class_id, image_dictionary, dict_class_to_indices_left,
                                             rndm_obj: Union[None, random.Random]):
        assert class_id in image_dictionary['labels_indices'].keys()
        # if there are no choices left, reset the values
        if class_id not in dict_class_to_indices_left.keys() or len(dict_class_to_indices_left[class_id]) == 0:
            cls.state_for_random_sampling_reset_values_for_class_id(image_dictionary, dict_class_to_indices_left, class_id)

        c_available_indices = dict_class_to_indices_left[class_id]
        c_selected_index = c_available_indices.pop(rndm_obj.randrange(len(c_available_indices)))
        return image_dictionary['sorted_images'][c_selected_index]

    @staticmethod
    def download_if_needed(dataset_name, directory, filepath_to_check, urls):
        """
        :param directory: to store the files in
        :param filepath_to_check:
        :param zips: list of urls
        :return:
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(filepath_to_check):
            print("Dataset {} doesn't exist. Will attempt to download it next.".format(dataset_name))
            for c_download_url in urls:
                c_archive_filename = os.path.basename(c_download_url)
                c_archive_filepath = "{}/{}".format(directory, c_archive_filename)

                if not os.path.exists(c_archive_filepath):
                    print("Downloading {}".format(c_archive_filename))
                    urllib.request.urlretrieve(c_download_url, c_archive_filepath)

                if c_archive_filename[-2:] == "gz":
                    new_filename = "{}/{}".format(directory, c_archive_filename[:-3])
                    with gzip.GzipFile(c_archive_filepath, 'rb') as f_in:
                        with open(new_filename, 'wb') as f_out:
                            f_out.write(f_in.read())
                elif c_archive_filename[-4:] == "data":
                    pass
                else:
                    zip_ref = zipfile.ZipFile(c_archive_filepath, 'r')
                    zip_ref.extractall(directory)
                    zip_ref.close()

    def _get_c_class_map(self):
        og_class_map = self._get_full_class_map()
        new_class_map = {}
        class_idx_start = (self.class_portion - 1) * ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM
        class_idx_end_exclusive = class_idx_start + ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM

        for og_idx, new_idx in og_class_map.items():
            #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            if new_idx >= class_idx_start and new_idx < class_idx_end_exclusive:
                new_class_map[og_idx] = new_idx % ImageDatasetProvider.NUM_CLASSES_PER_PROBLEM

        return new_class_map

    def _filter_images_based_on_class(self, all_images, all_labels):
        # all_images.shape = [n_items, n_channel, 28, 28]
        c_class_map = self._get_c_class_map()

        relevant_images = []
        relevant_labels = []
        for i in range(all_images.shape[0]):
            c_img = all_images[i]  # [0]
            c_label = all_labels[i].item()

            if c_label not in c_class_map.keys():
                continue

            new_label = c_class_map[c_label]
            relevant_images.append(c_img)
            relevant_labels.append(new_label)

        relevant_images_np = np.stack(relevant_images, axis=0)
        # relevant_images_np = np.expand_dims(relevant_images_np, 1)
        relevant_labels_np = np.stack(relevant_labels, axis=0)

        return relevant_images_np, relevant_labels_np

    def get_images_for_classification(self):
        # remap_classes = False
        """
        self.ds_images_tr = {
            "sorted_images": None,
            "sorted_labels": None,
            "labels_count": None,
            "labels_indices": None,
            "num_labels": None
        }
        """
        results = {"mean": self.mean, "std": self.std}
        for key, ds_images in [("tr", self.ds_images_tr), ("val", self.ds_images_val), ("test", self.ds_images_test)]:
            c_images = ds_images["sorted_images"].astype(np.float32)
            c_labels_int = ds_images["sorted_labels"]
            # c_labels_one_of_k = np.zeros((c_labels_int.shape[0], ds_images["num_labels"]))
            c_labels_one_of_k = np.zeros((c_labels_int.shape[0], 10))
            for lidx in range(c_labels_int.shape[0]):
                c_labels_one_of_k[lidx, c_labels_int[lidx]] = 1.
            results[key] = {"images": c_images, "labels": c_labels_one_of_k}
            print(f"{key}, images.shape = {c_images.shape}, labels.shape = {c_labels_one_of_k.shape}")

        return results

    @staticmethod
    @abc.abstractmethod
    def get_num_val_images_per_class():
        pass

    @staticmethod
    def _process_images(all_images, all_labels):
        return all_images, all_labels

    def load_data(self):
        """
        image_size = self.get_image_size()
        train_all_images = np array, type=np.float32, shape=(-1, 1, image_size, image_size)
        train_all_labels = np array, type=np.float32, shape=(-1, 1)
        train_all_labels is assumed to be classes {0., 1., 2. etc...}
        :return: returns train_all_labels, train_all_images
        """
        if not os.path.exists("Experiments/BELL"):
            raise FileNotFoundError("Make sure you are running from the project's root directory.")
        if not os.path.exists(ROOT_IMG_DATA_DIRECTORY):
            os.makedirs(ROOT_IMG_DATA_DIRECTORY)

        train_all_images, train_all_labels = self._get_images(train=True)
        train_all_images, train_all_labels = self._filter_images_based_on_class(train_all_images, train_all_labels)
        train_all_images, train_all_labels = self._process_images(train_all_images, train_all_labels)

        self.shuffle_in_unison(train_all_images, train_all_labels)

        num_val_images = self.get_num_val_images_per_class() * self.NUM_CLASSES_PER_PROBLEM

        val_first_idx = train_all_images.shape[0] - num_val_images
        train_images, train_labels = train_all_images[:val_first_idx], train_all_labels[:val_first_idx]
        val_images, val_labels = train_all_images[val_first_idx:], train_all_labels[val_first_idx:]

        if self.get_test:
            test_all_images, test_all_labels = self._get_images(train=False)
            test_all_images, test_all_labels = self._filter_images_based_on_class(test_all_images, test_all_labels)
            test_all_images, test_all_labels = self._process_images(test_all_images, test_all_labels)

            return train_images, train_labels, val_images, val_labels, test_all_images, test_all_labels

        return train_images, train_labels, val_images, val_labels

    @staticmethod
    @abc.abstractmethod
    def _get_images(train=True):
        pass

    @staticmethod
    def normalise_all_together():
        # True for images where we use a single constant for the mean and std
        # if False, we normalise each input dimension separately
        return True


class MNISTDataProvider(ImageDatasetProvider):
    NUM_VAL_IMAGES_PER_CLASS = 600

    def get_name(self):
        return f"MNIST_{self.class_portion}"

    @staticmethod
    def get_num_val_images_per_class():
        return MNISTDataProvider.NUM_VAL_IMAGES_PER_CLASS

    @staticmethod
    def _get_images(train=True):
        mnist_trainset = torchvision.datasets.MNIST(f"{ROOT_IMG_DATA_DIRECTORY}/MNIST", train=train,
                                                           download=True)
        all_images = mnist_trainset.data.numpy().reshape((-1, 1, 28, 28)).astype(np.float32)
        all_labels = np.expand_dims(mnist_trainset.targets.numpy(), axis=1)
        return all_images, all_labels

    @staticmethod
    def _get_full_class_map():
        # must map to a new index starting with 0
        return {i: i for i in range(10)}


class FMNISTDataProvider(MNISTDataProvider):
    def get_name(self):
        return f"FMNIST_{self.class_portion}"

    @staticmethod
    def _get_images(train=True):
        mnist_trainset = torchvision.datasets.FashionMNIST(f"{ROOT_IMG_DATA_DIRECTORY}/FMNIST", train=train,
                                                           download=True)
        all_images = mnist_trainset.data.numpy().reshape((-1, 1, 28, 28)).astype(np.float32)
        all_labels = np.expand_dims(mnist_trainset.targets.numpy(), axis=1)
        return all_images, all_labels


class Kuzushiji49DataProvider(MNISTDataProvider):
    # https://github.com/rois-codh/kmnist
    URLS = [
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz'
    ]

    @staticmethod
    def _get_full_class_map():
        # 33 different classes
        relevant_og_classes = [
            0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21,
            24, 25, 26, 27, 28, 30, 34, 35, 37, 38, 39,
            40, 41, 46, 47,
        ]
        # must map to a new index starting with 0
        full_class_map = {}
        for new_idx, og_idx in enumerate(relevant_og_classes):
            full_class_map[og_idx] = new_idx
        return full_class_map

    def get_name(self):
        return f"K49_{self.class_portion}"

    @staticmethod
    def _get_images(train=True):
        data_folder = f"{ROOT_IMG_DATA_DIRECTORY}/k49"
        tr_imgs_fp = f"{data_folder}/k49-train-imgs.npz"
        tr_labels_fp = f"{data_folder}/k49-train-labels.npz"
        test_imgs_fp = f"{data_folder}/k49-test-imgs.npz"
        test_labels_fp = f"{data_folder}/k49-test-labels.npz"

        Kuzushiji49DataProvider.download_if_needed("Kuzushiji49", data_folder, tr_imgs_fp, Kuzushiji49DataProvider.URLS)

        if train:
            all_images = np.load(tr_imgs_fp)['arr_0']
            all_labels = np.load(tr_labels_fp)['arr_0']
        else:
            all_images = np.load(test_imgs_fp)['arr_0']
            all_labels = np.load(test_labels_fp)['arr_0']

        all_images = all_images.reshape((-1, 1, 28, 28)).astype(np.float32)
        all_labels = all_labels.reshape((-1, 1))

        return all_images, all_labels


class EMNIST_corrected_url(torchvision.datasets.EMNIST):
    # the url from the torchvision version used seems to be outdated, this is a fix
    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"


class EMNISTDataProvider(MNISTDataProvider):
    @staticmethod
    def _get_full_class_map():
        return {i+1:i for i in range(26)}

    def get_name(self):
        return f"EMNIST_{self.class_portion}"

    @staticmethod
    def _get_images(train=True):
        # emnist_trainset = torchvision.datasets.EMNIST(f"{ROOT_IMG_DATA_DIRECTORY}/EMNIST", split="letters", train=train,
        #                                              download=True)
        emnist_trainset = EMNIST_corrected_url(f"{ROOT_IMG_DATA_DIRECTORY}/EMNIST", split="letters", train=train,
                                               download=True)

        all_images = emnist_trainset.data.numpy().reshape((-1, 1, 28, 28)).astype(np.float32)
        all_labels = emnist_trainset.targets.numpy().astype(np.int8).reshape((-1, 1))

        return all_images, all_labels

    @staticmethod
    def _process_images(all_images, all_labels):
        processed_images = []
        for i in range(all_images.shape[0]):
            c_img = all_images[i][0]
            c_img_flipped = np.flip(c_img, axis=0)
            c_img_rotated = rotate(c_img_flipped, angle=-90)
            processed_images.append(c_img_rotated)

        relevant_images_np = np.stack(processed_images, axis=0)
        relevant_images_np = np.expand_dims(relevant_images_np, 1)

        return relevant_images_np, all_labels
