Deep Topological Learning (DeepTL)
======================================================

DeepTL is a Python package providing an easy-to-use software
for learning complex topologies with neural networks.

DeepTL networks are based on a novel theory (`the duality theory`)
bridging
two research fields which are usually thought as disjointed:
gradient-based and competitive
neighborhood-based learning.


Examples on benchmark datasets
--------------------------------
.. list-table::

    * - .. figure:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/Spiral_dual.png
            :height: 100px

      - .. image:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/Circles_dual.png
            :height: 100px

      - .. image:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/Moons_dual.png
            :height: 100px



Using DeepTL
---------------

.. code:: python

    from deeptl import DeepTopologicalClustering

    X, y = ... # load dataset

    # load and fit the neural model
    model = DeepTopologicalClustering()
    model.fit(X)

    # compute the final graph and plot the result
    model.compute_graph()
    model.plot_graph(y)


Authors
-------

`Pietro Barbiero <http://www.pietrobarbiero.eu/>`__

Licence
-------

Copyright 2020 Pietro Barbiero.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.