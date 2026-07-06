import os
import pickle

import lmdb
import redis
import torch


class LMDBLatentsDataset(torch.utils.data.Dataset):
    """
    Args:
        lmdb_path (str): LMDB dataset path.
        flip_prob (float): flip or upflip.
    """

    def __init__(self, lmdb_path, flip_prob=0.5, return_raw_data=False):
        self.lmdb_path = lmdb_path
        self.env = None

        self.flip_prob = flip_prob
        self.return_raw_data = return_raw_data
        try:
            env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                length_from_db = txn.get("num_samples".encode())
                if length_from_db is None:
                    raise ValueError(f"Key 'num_samples' not found in LMDB at {lmdb_path}")
                self.length = int(length_from_db.decode())
        finally:
            if "env" in locals() and env is not None:
                env.close()
            del env

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        with self.env.begin() as txn:
            data = txn.get(f"{index}".encode())
            if data is None:
                raise IndexError(f"Index {index} is out of bounds")

            if self.return_raw_data:
                return data
            else:
                data = pickle.loads(data)
                moments = data["moments"]
                moments_flip = data["moments_flip"]
                label = data["label"]

                use_flip = torch.rand(1).item() < self.flip_prob

                moments_to_use = moments_flip if use_flip else moments

                moments_tensor = torch.from_numpy(moments_to_use).float()

                return moments_tensor, label

    def __del__(self):
        if self.env is not None:
            self.env.close()


class RedisCachedImageFolder:
    def __init__(self, redis_ports: list, root: str, flip_prob=0.5, redis_host="localhost"):
        self.root = root
        self.dataset = LMDBLatentsDataset(root, flip_prob=flip_prob, return_raw_data=True)
        self.flip_prob = flip_prob

        # Simplified redis connection
        self.redis_ports = redis_ports
        self.redis_host = redis_host
        self.redis_clients = []
        self.num_shards = len(redis_ports)

        self.redis_clients = None
        # self._init_redis_connection()
        self.cache_misses = 0
        self.cache_hits = 0
        self.dataset_prefix = os.path.basename(root)[0]

    def _init_redis_connection(self):
        try:
            if self.redis_clients is not None:
                for redis_client in self.redis_clients:
                    redis_client.close()

            # Simple Redis connection
            self.redis_clients = []
            for port in self.redis_ports:
                redis_client = redis.StrictRedis(
                    host=self.redis_host, port=port, decode_responses=False, socket_connect_timeout=5, socket_timeout=5
                )
                redis_client.ping()
                self.redis_clients.append(redis_client)

        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_clients = []

    def _get_redis_client(self, key):
        if self.redis_clients is None:
            self._init_redis_connection()

        # 如果连接失败，redis_clients 可能还是 None
        if self.redis_clients:
            return self.redis_clients[key % self.num_shards]
        return None

    def _safe_redis_get(self, key):
        redis_client = self._get_redis_client(key)
        # redis_client = self.redis_clients[key % self.num_shards]
        if redis_client:
            try:
                return redis_client.get(key)
            except:
                return None
        return None

    def _safe_redis_set(self, key, value):
        redis_client = self._get_redis_client(key)
        # redis_client = self.redis_clients[key % self.num_shards]
        if redis_client:
            try:
                return redis_client.set(key, value)
            except:
                return False
        return False

    def __getitem__(self, index):
        # cache_key = f"{self.dataset_prefix}{index}"
        cache_key = index

        file_data = self._safe_redis_get(cache_key)
        if file_data is None:
            self.cache_misses += 1
            try:
                file_data = self.dataset.__getitem__(index)
                self._safe_redis_set(cache_key, file_data)
            except Exception as e:
                print(f"Error reading file {index}: {e}")
                raise
        else:
            self.cache_hits += 1

        try:
            data = pickle.loads(file_data)
            moments = data["moments"]
            moments_flip = data["moments_flip"]
            label = data["label"]

            use_flip = torch.rand(1).item() < self.flip_prob

            moments_to_use = moments_flip if use_flip else moments

            moments_tensor = torch.from_numpy(moments_to_use).float()

        except Exception as e:
            print(f"Error decoding image data for index {index}: {e}")

        # if (self.cache_hits + self.cache_misses) % 1000 == 0:
        #     print(f"Redis Cache stats - hits: {self.cache_hits}, misses: {self.cache_misses}")

        return moments_tensor, label

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        """Destructor to ensure connections are properly closed"""
        if self.redis_clients is not None:
            for redis_client in self.redis_clients:
                try:
                    redis_client.close()
                except:
                    pass
