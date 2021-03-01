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

# References

Bhowmick, D., WINTER, S. and STEVENSON, M., 2019. Comparing the costs of pedestrian wayfinding heuristics across different urban network morphologies. GeoComputation 2019.

Bhowmick, D., Winter, S., Stevenson, M. and Vortisch, P., 2020. The impact of urban road network morphology on pedestrian wayfinding behavior. Journal of Spatial Information Science, 2020(21), pp.203-228.

# Citations

If you are using the code for your work, please cite the papers using the following BibTeX entries.

@inproceedings{Bhowmick2019,
    author = "Debjit Bhowmick and Stephan Winter and Mark Stevenson",
    title = "{Comparing the costs of pedestrian wayfinding heuristics across different urban network morphologies}",
    booktitle={GeoComputation 2019},
    year = "2019",
    Month = {September},
    url = "\url{https://auckland.figshare.com/articles/Comparing_the_costs_of_pedestrian_wayfinding_heuristics_across_different_urban_network_morphologies/9846137}",
    Doi = {\mydoi{10.17608/k6.auckland.9846137.v1}},
    type = {Conference Proceedings}
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
