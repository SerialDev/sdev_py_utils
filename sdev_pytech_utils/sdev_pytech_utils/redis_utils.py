"""Python Redis utility library"""
import redis

class redis_utils(object):
    @staticmethod
    def get_db(config):
        db = redis.StrictRedis(host=config['Redis']['ip'], port=int(config['Redis']['port']))
        return db

    @staticmethod
    def get_session_id(message):
        return int(message.partition_key.decode().split(':')[0])

    @staticmethod
    def get_server_info(db, session_id):
        return db.hgetall('s:%s' % session_id)

    @staticmethod
    def get_player_info(db, session_id, player_id):
        return db.hgetall('p:%s:%s' % (session_id, player_id))

    @staticmethod
    def get_all_keys(db):
        return db.keys(pattern='*')
