
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
* Perform backtracking search with the Minimum Remaining Values heuristic and forward checking for inference.
* Display the harder Sudoku puzzle after applying backtracking search.
* Measure the time taken for backtracking search to execute.

### Requirements
* Python 3.x


### Dependencies


**utilspy**: Contains utility functions used throughout the code.

**csp.py**: Contains the CSP class and related functions for solving constraint satisfaction problems. Sudoku is just one possible type of CSP but the code in CSP.py can be expanded to any type of CSP.

To install the required libraries make sure you run the following commands: 

* pip **install** numpy
* pip **install** sortedcontainers

To install the required libraries for the image recognition, run the following commands:

pip install torch
pip install torchvision
pip install matplotlib

### Usage
* Save the provided code in a file named main.py.
* Run the code using the command:
*   python main.py

The output will display the initial state, post-ac3, and solved states of the Sudoku puzzles as well as the time taken for the AC3 algorithm and backtracking search to execute.

### References
* [Solving Sudoku From Image Using Deep Learning – With Python Code](https://www.analyticsvidhya.com/blog/2021/05/solving-sudoku-from-image-using-deep-learning-with-python-code/)
* [Python implementation of algorithms from Russell and Norvig's "Artificial Intelligence: A Modern Approach"](https://github.com/aimacode/aima-python)
