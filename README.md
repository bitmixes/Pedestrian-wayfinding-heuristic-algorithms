# What are pedestrian wayfinding heuristic algorithms?

Human wayfinding in outdoor spaces involves selecting path segments from an existing real-world network to determine a route between any two given points present in the network. 
Pedestrians, during wayfinding, do not always choose the shortest available route, but apply certain wayfinding strategies or heuristics which minimize their cognitive effort.
These heuristics minimize cognitive effort and usually lead to satisfactory route choices. 
These wayfinding heuristics are not only applied by a navigator in an unfamiliar environment, but also by people having complete spatial knowledge relevant for making wayfinding decisions.
Several studies have explored human wayfinding strategies in outdoor spaces.
A review of existing wayfinding literature reveals the existence of multiple heuristics that are applied by pedestrians.

# What does this code do?

This Python code obtains street network data from OpenStreetMap.
It contains algorithms simualting four wayfinding heuristics Least angle strategy, Longest-leg first (or initial segment) strategy, Shortest-leg first strategy and Fewest turns strategy.
These simulated routes represent actual routes which would be chosen if any one of these heuristics were applied by a pedestrian, consistently during their wayfinding. 

# Reference
If you are using the code for your work, please cite the following papers

@article{bhowmick2019comparing,
  title={Comparing the costs of pedestrian wayfinding heuristics across different urban network morphologies. GeoComputation 2019},
  author={Bhowmick, Debjit and WINTER, STEPHAN and STEVENSON, MARK},
  year={2019},
  publisher={The University of Auckland}
}

@article{bhowmick2020impact,
  title={The impact of urban road network morphology on pedestrian wayfinding behavior},
  author={Bhowmick, Debjit and Winter, Stephan and Stevenson, Mark and Vortisch, Peter},
  journal={Journal of Spatial Information Science},
  volume={2020},
  number={21},
  pages={203--228},
  year={2020}
}

# Contact

If you have any queries, please contact dbhowmick@student.unimelb.edu.au
