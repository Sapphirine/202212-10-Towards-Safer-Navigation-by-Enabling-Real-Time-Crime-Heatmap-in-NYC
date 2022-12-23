import json
import redis
from threading import Thread


class DataReceiver(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.redis = redis.StrictRedis()
        self.pubsub = self.redis.pubsub()
        self.pubsub.psubscribe('*')  # To subscribe to certain channels, see: https://redis.io/commands/psubscribe/
        self.run()

    def run(self):
        for m in self.pubsub.listen():
            if 'pmessage' != m['type']:
                continue
            ch_name = m["channel"].decode('utf-8')
            ch_data = json.loads(m['data'].decode('utf-8'))
            print(ch_name, ch_data)


if __name__ == '__main__':
    DataReceiver()
