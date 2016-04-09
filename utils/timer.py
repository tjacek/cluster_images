import timeit

class Timer(object):
    def __init__(self):
        self.start_time=timeit.default_timer()

    def stop(self):
        self.end_time = timeit.default_timer()
        self.total_time = (self.end_time - self.start_time)

    def show(self):
        print("Training time %d ",self.total_time)

    def show(self,info):
    	print(info)
        print("Training time %d ",self.total_time)

def clock(func):
    def inner_func(arg):
        func_timer=Timer()
        result=func(arg)
        func_timer.stop()
        func_timer.show("Timer "+func.__name__)
        return result
    return inner_func