from contextlib import contextmanager
import time
from utils import format_num, format_time, format_object
import threading
from logger import log, Colors

print_lock = threading.Lock()

@contextmanager
def timer(name:str=None, repetitions:float=None):
    start_time = time.time()  # Record start time
    ret = None
    try:
        yield lambda: ret  # Allow the context to return a value
    finally:
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        with print_lock:
            log(Colors.green('runtime = ') + Colors.red(format_time(elapsed_time)), end='')
            if repetitions is not None:
                log(f' x {format_num(repetitions)} = {format_time(repetitions*elapsed_time)}', end='')
            if name is not None:
                log(' ' + Colors.green(f'[{name}]'), end='')
            log()

        if repetitions is None:    
            ret = elapsed_time
        else:
            ret = elapsed_time * repetitions

def timed(func, repetitions=None):
    def wrapper(*args, **kwargs):
        name = func.__name__ + '(' + ', '.join(format_object(arg) for arg in args) + ')'
        with timer(name=name, repetitions=repetitions):
            return func(*args, **kwargs)
    return wrapper

