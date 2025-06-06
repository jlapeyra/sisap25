from collections import Counter

def format_num(x:int, base=1000):
    i = 0
    units = ['', 'k', 'M', 'G', 'T']
    while x > base and i < len(units):
        i += 1
        x /= base
    unit = units[i]
    if not unit and isinstance(x, int):
        return str(x)
    else:
        decimals = 2 if x < 10 else 1 if x < 100 else 0
        return f'{round(x, decimals)}{unit}'

def format_bytes(x:int):
    return format_num(x, base=1024) + 'B'

def format_time(sec:float, force_format=False, force_seconds=False):
    if (sec < 60 and not force_format) or force_seconds:
        return f'{sec:.4f}s'
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f'{hours}:{minutes:02}:{seconds:02.0f}'



def __close_unclosed(text:str):
    count = Counter(text)
    return '}'*max(0, count['{'] - count['}']) +  ']'*max(0, count['['] - count[']']) +  ')'*max(0, count['('] - count[')'])


def format_object(obj, max_length=50):
    obj = repr(obj).replace('\n', ' ')
    if len(obj) <= max_length:
        return obj
    else:
        return obj[:max_length] + '...' + __close_unclosed(obj[:max_length])






