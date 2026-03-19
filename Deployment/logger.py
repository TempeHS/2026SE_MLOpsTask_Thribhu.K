_pending = []

def log(msg, data=None):
    _pending.append({"msg": msg, "data": data})

def flush():
    msgs = list(_pending)
    _pending.clear()
    return msgs