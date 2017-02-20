import pyximport
import numpy as np
pyximport.install()
import memview
import time
a = np.array(range(1000000), dtype=np.float64)

print 'parallel'
t = time.time()
b = memview.transf_value(a,2)
print time.time()-t

print 'serial'
t = time.time()
b = memview.transf_value_serial(a,2)
print time.time()-t
