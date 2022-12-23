import json
import redis

if __name__ == '__main__':
    redis_inst = redis.StrictRedis()

    data = [[1, 2, 3], [2, 1, 4]]
    redis_inst.publish(f'channel-name', json.dumps(data).encode('utf-8'))

