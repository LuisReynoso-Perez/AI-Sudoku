External libraries used: numpy, sortedcontainers. Be sure these are downloaded as well.

# Read Me

### Sudoku Solver ReadMe
This code provides a Sudoku puzzle solver using Constraint Satisfaction Problems (CSP) and the AC3 algorithm for constraint propagation. The main.py file contains the following steps:

* Create a Sudoku problem instance using the csp.Sudoku class.
* Display the initial state of the Sudoku puzzle.
* Apply the AC3 algorithm for constraint propagation.
* Display the Sudoku puzzle after applying the AC3 algorithm.
* Measure the time taken for the AC3 algorithm to execute.
* Create a harder Sudoku problem instance.
* Apply the AC3 algorithm for constraint propagation, but it doesn't fully solve the problem.
* Perform backtracking search with the MRV heuristic and forward checking for inference.
* Display the harder Sudoku puzzle after applying backtracking search.
* Measure the time taken for backtracking search to execute.

### Requirements
* Python 3.x
*  Make sure all  files are in the same folder
*  Run main.py

### Dependencies
**utilspy**: Contains utility functions used throughout the code.
**csp.py**: Contains the CSP class and related functions for solving constraint satisfaction problems.
To install the required libraries make sure you run the following command: 

* pip **install** numpy

### Usage
* Save the provided code in a file named main.py.
* Run the code using the command:
*   python main.py
The output will display the initial and solved states of the Sudoku puzzles as well as the time taken for the AC3 algorithm and backtracking search to execute.

