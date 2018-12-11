import tvm
import time
from tvm.contrib import cc
from tvm.contrib import util
import numpy as np

def simple_llvm_save_module():
  n = tvm.var ("n")
  A = tvm.placeholder ((n ,n), name='A', dtype="float32")
  B = tvm.placeholder ((n, n), name='B', dtype="float32")
  C = tvm.compute (A.shape, lambda *i: A(*i) + B(*i), name='C')
  s = tvm.create_schedule (C.op)
  module = tvm.build(s, [A, B, C], "llvm", "llvm")

  temp = util.tempdir()
  module.save (temp.relpath ("myadd.o"))
  cc.create_shared (temp.relpath("myadd.so"), [temp.relpath("myadd.o")])

  ctx = tvm.context ("llvm", 0)
  n = 1024
  a = tvm.nd.array(np.random.uniform(size=(n,n)).astype(A.dtype), ctx)
  b = tvm.nd.array(np.random.uniform(size=(n,n)).astype(A.dtype), ctx)
  c = tvm.nd.array(np.random.uniform(size=(n,n)).astype(A.dtype), ctx)

  myadd = tvm.module.load (temp.relpath ("myadd.so"))
  t0 = time.time()
  myadd(a, b, c)
  t1 = time.time()
  print ("CPU time: %s" %(t1 - t0))

def simple_device_save_module():
  def check_code_gen(device):
    n = tvm.var ("n")
    A = tvm.placeholder ((n ,n), name='A', dtype="float32")
    B = tvm.placeholder ((n, n), name='B', dtype="float32")
    C = tvm.compute (A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule (C.op)

    bx, tx = s[C].split (C.op.axis[0], factor=64)
    s[C].bind (bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind (tx, tvm.thread_axis("threadIdx.x"))
    
    #print (tvm.lower(s, [A, B, C], simple_mode=True))
    module = tvm.build(s, [A, B, C], device, target_host="llvm")
    #print ("Device code %s" %device)
    #print (module.imported_modules[0].get_source())
    #print (module.get_source ("asm"))

    temp = util.tempdir()
    module.save (temp.relpath("myadd.o"))
    # Save device code
    suffix = "vulkan"
    if device == "opencl":
      suffix = "cl"
    module.imported_modules[0].save(temp.relpath("myadd.%s" %suffix))
    # Create shared library
    cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])

    myadd = tvm.module.load (temp.relpath("myadd.so"))
    # Import "deviced" code
    myadd_device = tvm.module.load(temp.relpath("myadd.%s" %suffix))
    # Import module 
    myadd.import_module(myadd_device)

    ctx = tvm.context (device, 0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=(n,n)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(n,n)).astype(A.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=(n,n)).astype(A.dtype), ctx)
    t0 = time.time()
    myadd(a, b, c)
    t1 = time.time()
    print (device)
    print ("GPU time: %s" %(t1 - t0))

  check_code_gen("vulkan")
  check_code_gen("opencl")

if __name__ == "__main__":
  simple_llvm_save_module()
  simple_device_save_module()
