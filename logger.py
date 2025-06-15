from utils import *
import time
import re
import os

class Colors:
    @staticmethod
    def red(text: str) -> str:
        return f'\033[91m{text}\033[0m'
    @staticmethod 
    def green(text: str) -> str:
        return f'\033[92m{text}\033[0m'
    @staticmethod
    def yellow(text: str) -> str:
        return f'\033[93m{text}\033[0m'
    @staticmethod
    def blue(text: str) -> str:
        return f'\033[94m{text}\033[0m'

START_TIME = time.time()

def time_since_start():
    return time.time() - START_TIME

def current_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

os.makedirs('logs', exist_ok=True)
LOGS = open(f'logs/{current_time_str()}.log', 'w', encoding='utf-8')
__new_line = True

def log(*values: object, sep: str | None = " ", end: str | None = "\n"):
    global __new_line, LOGS
    if __new_line:
        preffix = f'[{format_time(time_since_start(), force_format=True)}]'
        print(Colors.blue(preffix), end='  ')
        print(preffix, end='  ', file=LOGS)
        __new_line = False
    text = (sep or '').join(map(str, values)) +  (end or '')
    print(text, end='')
    text = re.sub(r'\033\[\d+m', '', text)  # Remove ANSI escape codes
    print(text, end='', file=LOGS, flush=True)
    if text.endswith('\n'):
        __new_line = True


LOOP_START_TIME = None
def start_loop_time():
    global LOOP_START_TIME
    LOOP_START_TIME = time.time()
    log('start_loop_time()')

def stop_loop_time():
    global LOOP_START_TIME
    LOOP_START_TIME = None
    log('stop_loop_time()')

def log_expected_time(ended_iteration, total_iterations):
    global LOOP_START_TIME
    assert LOOP_START_TIME is not None, f'call start_loop_time before log_expected_time'
    it = ended_iteration + 1
    time_loop = time.time() - LOOP_START_TIME
    time_before_loop = LOOP_START_TIME - START_TIME
    log(f'Iteration {format_num(it)}/{format_num(total_iterations)} | {format_time(time_loop/it)}/it'
        f' | Expected end loop: {Colors.yellow(format_time(time_before_loop + time_loop/it*total_iterations))}')