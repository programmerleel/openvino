import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import cv2
import glob
import logging
import numpy as np
import openvino as ov
import os
import shutil
import sqlite3
import sys
from tensorflow import keras
from tqdm import tqdm
import traceback

# 生成日志
def create_logs(filename):
    logging.basicConfig(filename=filename, format='[%(asctime)s] [%(levelname)s] >>> %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %I:%M:%S')


# 删除图片
def remove_files(local_root):
    thread_pool = ThreadPoolExecutor(32)
    tasks = []
    for account in os.listdir(local_root):
        account_path = os.path.join(local_root, account)
        original_path = os.path.join(account_path, "original")
        if os.path.exists(original_path):
            for category in os.listdir(original_path):
                category_path = os.path.join(original_path, category)
                for file in os.listdir(category_path):
                    if file == "cropped":
                        cropped_path = os.path.join(category_path, file)
                        for image in os.listdir(cropped_path):
                            image_path = os.path.join(cropped_path, image)
                            tasks.append(thread_pool.submit(do_delete, image_path))
                    else:
                        file_path = os.path.join(category_path, file)
                        tasks.append(thread_pool.submit(do_delete, file_path))
    for task in tqdm(as_completed(tasks), desc="remove files", total=len(tasks)):
        pass


def do_delete(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())


# 删除文件夹
def remove_folders(local_root):
    accounts = []
    for account in os.listdir(local_root):
        accounts.append(account)
    for account in tqdm(accounts, desc="remove folders", total=len(accounts)):
        account_path = os.path.join(local_root, account)
        original_path = os.path.join(account_path, "original")
        if os.path.exists(original_path):
            try:
                shutil.rmtree(original_path)
            except Exception as e:
                logging.error(e)
                logging.error(traceback.format_exc())


# 移动重叠文件夹并重命名文件夹
# 不要打开文件夹，会导致拒绝访问的error
def do_folders(local_root):
    accounts = []
    for account in os.listdir(local_root):
        accounts.append(account)
    for account in tqdm(accounts, desc="move folders and rename folders", total=len(accounts)):
        account_path = os.path.join(local_root, account)
        classified_path = os.path.join(account_path, "classified")
        try:
            os.rename(classified_path, os.path.join(account_path, "category"))
            for category in os.listdir(os.path.join(account_path, "category")):
                category_path = os.path.join(account_path, "category", category)
                flag = 0
                for file in os.listdir(category_path):
                    src_path = os.path.join(category_path, file)
                    if os.path.isdir(src_path):
                        shutil.move(src_path, os.path.join(account_path, "category"))
                        os.rename(os.path.join(account_path, "category", file),
                                  os.path.join(account_path, "category", category + "#{}".format(flag)))
                        flag = flag + 1
                flag = 0
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())


def rename_files(local_root):
    thread_pool = ThreadPoolExecutor(32)
    tasks = []
    for account in os.listdir(local_root):
        account_path = os.path.join(local_root, account)
        for category in os.listdir(os.path.join(account_path, "category")):
            category_path = os.path.join(account_path, "category", category)
            if len(os.listdir(category_path)) == 0:
                shutil.rmtree(category_path)
                continue
            for file in os.listdir(category_path):
                if "#" in file:
                    file_path = os.path.join(category_path, file)
                    new_path = os.path.join(category_path, file_path.split("#")[1])
                    tasks.append(thread_pool.submit(do_rename, file_path, new_path))
    for task in tqdm(as_completed(tasks), desc="rename files", total=len(tasks)):
        pass


def do_rename(file_path, new_path):
    try:
        os.rename(file_path, new_path)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())


class DBHelper:
    def __init__(self, db_root):
        try:
            self.connection = sqlite3.connect(db_root, check_same_thread=False)
            self.cursor = self.connection.cursor()
            self.create_table()
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())

    def create_table(self):
        sql = """create table if not exists embeddings (file_name TEXT PRIMARY KEY NOT NULL,embedding BLOB);"""
        try:
            self.cursor.execute(sql)
            self.connection.commit()
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())

    def select_all_records(self):
        embeddings = {}
        sql = """select file_name,embedding from embeddings"""
        try:
            rows = self.cursor.execute(sql)
            for row in rows:
                file_name, embedding = row[0], row[1]
                embedding = np.frombuffer(embedding, dtype=np.float32)
                embeddings[file_name] = embedding
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
        return embeddings

    # 需要commit
    def insert_records(self, values):
        sql = """replace into embeddings (file_name,embedding) values (?,?);"""
        try:
            self.cursor.executemany(sql, values)
            self.connection.commit()
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())


class Encoder:
    def __init__(self, model_root,model_type):
        self.model_type = model_type
        if self.model_type == "tf":
            self.encoder = keras.models.load_model(os.path.join(model_root,"emb_model"))
            self.image_size = self.encoder.input_shape[1:3]
        elif self.model_type == "ov":
            model_path = os.path.join(model_root,"emb_model.xml")
            core = ov.Core()
            read_model = core.read_model(model_path)
            pop = ov.preprocess.PrePostProcessor(read_model)
            pop.input().tensor().set_layout(ov.Layout('NHWC'))
            model = pop.build()
            compile_model = core.compile_model(model, "CPU")
            self.infer_request = compile_model.create_infer_request()
            self.image_size = [-1,160,160,3]
            print(self.image_size)
            self.image_size = [self.image_size[1],self.image_size[2]]

    def deal_image(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]
        height, width, _ = image.shape
        if height > width:
            image = np.rot90(image)
            height, width = width, height
        if height != width:
            max_len = max(height, width)
            top = (max_len - height) // 2
            bottom = max_len - height - top
            left = (max_len - width) // 2
            right = max_len - width - left
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image / 255.0
        return image_path, image

    def inference(self, image_paths):
        thread_pool = ThreadPoolExecutor(32)
        tasks = []
        for image_path in image_paths:
            tasks.append(thread_pool.submit(self.deal_image,image_path))
        datas = {}
        for task in as_completed(tasks):
            image_path, image = task.result()
            datas[image_path] = image
        batch_images = [datas[image_path] for image_path in image_paths]
        if self.model_type == "tf":
            batch_results = self.encoder.predict(np.array(batch_images),verbose=None)
            return batch_results
        elif self.model_type == "ov":
            input_tensor = ov.Tensor(np.array(batch_images,dtype=np.float32))
            self.infer_request.set_input_tensor(input_tensor)
            self.infer_request.infer()
            output_tensor = self.infer_request.get_output_tensor()
            batch_results = output_tensor.data
            print(batch_results.shape)
            return batch_results
'''
batch 设置为32 openvino batch 设置为-1 速度快于tf cpu还未满载
由于本机禁用了核显，没有使用openvino CPU进行测试
'''
def generate_embedding(local_root, model_root,model_type, db_root, batch_size=32):
    image_paths = []
    records = []
    db_helper = DBHelper(db_root)
    for account in os.listdir(local_root):
        account_path = os.path.join(local_root, account)
        for category in os.listdir(os.path.join(account_path, "category")):
            category_path = os.path.join(account_path, "category", category)
            for image in os.listdir(category_path):
                image_path = os.path.join(category_path, image)
                image_paths.append(image_path)
    encoder = Encoder(model_root,model_type)
    with tqdm(total=len(image_paths), desc="generate embedding") as pbar:
        for i in range(0, len(image_paths), batch_size):
            pbar.update(batch_size)
            batch_images = image_paths[i:i + batch_size]
            batch_results = encoder.inference(batch_images)
            for image_path, embedding in zip(batch_images, batch_results):
                records.append((os.path.basename(image_path), embedding))
                if len(records) > 10000:
                    db_helper.insert_records(records)
                    records = []
        if len(records) > 0:
            db_helper.insert_records(records)
    del encoder
    pbar.close()


def do_sort(local_root, db_root):
    category_paths = []
    file_paths = []
    embeddings = []
    tasks = []
    thread_pool_1 = ThreadPoolExecutor(20)
    db_helper = DBHelper(db_root)
    records = db_helper.select_all_records()
    for account in os.listdir(local_root):
        account_path = os.path.join(local_root, account)
        for category in os.listdir(os.path.join(account_path, "category")):
            category_path = os.path.join(account_path, "category", category)
            category_paths.append(category_path)
    for category_path in tqdm(category_paths, total=len(category_paths), desc="do_sort"):
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            file_paths.append(file_path)
        for file_path in file_paths:
            embeddings.append(records[os.path.basename(file_path)])
        center_embedding = np.sum(np.array(embeddings), axis=0) / len(embeddings)
        for file_path in file_paths:
            tasks.append(thread_pool_1.submit(do_dis, file_path, records, center_embedding, category_path))
        for task in as_completed(tasks):
            pass
        file_paths = []
        embeddings = []
        tasks = []


def do_dis(file_path, records, center_embedding, category_path):
    try:
        embedding = records[os.path.basename(file_path)]
        dis = euclidean_dis(embedding, center_embedding)
        os.rename(file_path, os.path.join(category_path, "{}#{}".format(dis, os.path.basename(file_path))))
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())


def euclidean_dis(v1, v2):
    return round(float(np.linalg.norm(v1 - v2)), 5)


def main(args):
    local_root = args.local_root
    filename = args.filename
    model_root = args.model_root
    model_type = args.model_type
    db_root = args.db_root
    create_logs(filename)
    # remove_files(local_root)
    # remove_folders(local_root)
    # do_folders(local_root)
    rename_files(local_root)
    generate_embedding(local_root, model_root,model_type, db_root)
    do_sort(local_root, db_root)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_root', type=str, help='', default=r'D:/data/image_classifier/test')
    parser.add_argument('--filename', type=str, help='', default=r'logs.txt')
    parser.add_argument('--model_root', type=str, help='',
                        default=r'D:\data\image_classifier\account_info\models\Encoder\plate\20230529-175908')
    parser.add_argument('--model_type',type=str,help='',default='ov')
    parser.add_argument('--db_root', type=str, help='', default=r'D:/data/image_classifier/embeddings.db')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
