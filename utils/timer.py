import timeit

class Timer(object):
    def __init__(self):
        self.start_time=timeit.default_timer()

    def stop(self):
        self.end_time = timeit.default_timer()
        self.total_time = (self.end_time - self.start_time)

    def show(self):
        print("Training time %d ",self.total_time)
