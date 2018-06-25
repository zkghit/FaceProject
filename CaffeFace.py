print("begin to import caffe...")
import caffe
print("import caffe finished")
solver = caffe.SGDSolver('Kaige_solver_phase2.prototxt')
solver.solve()

print ("hello end")
