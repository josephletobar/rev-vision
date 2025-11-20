import numpy as np

class Trajectory:
    def __init__(self, buffer_size=5, threshold=120):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.seeding_buffer = []
        self.buffer = []

    def push(self, new_point):
        self.buffer.append(new_point)
        self.buffer_size += 1

    def last(self):
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1]
    
    def all(self):
        return self.buffer

    # not using currently
    def update(self, new_point):
        # Use the first x mean for initial acceptance point 
        if len(self.seeding_buffer) == self.buffer_size: 
            self.seeding_buffer.append(new_point)
            mean_point = np.mean(self.seeding_buffer, axis=0) # mean across columns
            self.buffer.append(mean_point)
            return mean_point # first buffer point
        
        elif len(self.seeding_buffer) < self.buffer_size: # Must continue seeding
            self.seeding_buffer.append(new_point)
            return None

        # Enough seeding
        else: 
            # Start checking points (new point compared to the last buffer point)
            last_point = self.buffer[-1]
            dist = np.linalg.norm(np.array(new_point) - np.array(last_point))

            if dist > self.threshold: # too far, reject
                return None
            else:
                self.buffer.append(new_point) 

        return new_point
