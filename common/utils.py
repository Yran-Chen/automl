import os
import configparser
import logging
import time

root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..','Logfile')

class LogHandler():
    def __init__(self,name = 'test',setLvl = logging.NOTSET):
        try:
            tmp_dir = self.get_log_route()
        except configparser.NoSectionError:
            try:
                os.listdir(os.path.join(root_dir, 'log'))
            except FileNotFoundError:
                os.mkdir(os.path.join(root_dir, 'log'))
            finally:
                tmp_dir = root_dir

        logging.root.setLevel(setLvl)
        log_file = os.path.join(tmp_dir, 'log', '{}'.format(name))
        _log = logging.getLogger()
        f_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(formatter)
        _log.addHandler(f_handler)
        _log.setLevel(logging.INFO)
        self._log = _log

    def get_log_route(self):
        conf_file = os.path.join(root_dir, "conf", 'application.conf')
        conf = configparser.ConfigParser()
        conf.read(conf_file)
        return conf.get('log', 'route')

def kwargDecorator(decorator):

    def decorator_proxy(f=None, **kwargs):
        if f is not None:
            return decorator(f=f, **kwargs)

        def decorator_proxy(func):
            return decorator(f=f, **kwargs)

        return decorator_proxy

    return decorator_proxy

def log(_log):

    def decorator(f):

        def wrapper(ins, *args):
            _log.info('{} start at {}'.format(f.__name__, str(time.ctime())))
            _log.info('********************************************************************************')
            _log.info('input json:')
            for k, v in ins.param.items():
                _log.info('{}, {}'.format(k, v))
            _log.info('********************************************************************************')
            try:
                a = f(ins, *args)
                _log.info('{} end at {}'.format(f.__name__, str(time.ctime())))
            except Exception as e:
                _log.error(e, exc_info=True)
                raise e
            finally:
                _log.info('********************************************************************************')
                _log.info('********************************************************************************')

        return wrapper

    return decorator

