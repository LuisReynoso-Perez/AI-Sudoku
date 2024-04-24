from utils import *
import csp
import time


easy1 = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
harder1 = '4173698.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'

start_time = time.time()
e = csp.Sudoku(easy1)
print("solving this sudoku:")
e.display(e.infer_assignment())
print("applying AC3")
csp.AC3(e) # Constraint propagation via AC3 algorithm:
e.display(e.infer_assignment())

end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

start_time1 = time.time()
h = csp.Sudoku(harder1)
print("solving this sudoku:")
h.display(h.infer_assignment())
print("after applying AC3")
csp.AC3(h) # Constraint propagation via AC3 algorithm: doesn't fully solve the problem
h.display(h.infer_assignment()) # incomplete
csp.backtracking_search(h, select_unassigned_variable=csp.mrv, inference=csp.forward_checking)
print("backtracking search")
h.display(h.infer_assignment()) # complete

end_time1 = time.time()

elapsed_time1 = end_time1 - start_time1
print("Elapsed time:", elapsed_time1, "seconds")