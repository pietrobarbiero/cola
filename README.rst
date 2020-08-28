COLA - Competitive layers for deep learning
======================================================

|Build|
|Coverage|

|PyPI license|
|PyPI-version|



.. |Build| image:: https://img.shields.io/travis/pietrobarbiero/cola?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.org/pietrobarbiero/cola

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/pietrobarbiero/cola?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/pietrobarbiero/cola

.. |PyPI license| image:: https://img.shields.io/pypi/l/deepcola.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/deepcola/

.. |PyPI-version| image:: https://img.shields.io/pypi/v/deepcola?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/deepcola/

COLA (COmpetitive LAyers) is a Python package providing the implementation of
gradient-based competitive layers which can be used on top of deep
learning models for unsupervised tasks.


.. image:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/deep_dual_figure.png
    :height: 300px


Theory
--------
Theoretical foundations can be found in our paper.

If you find COLA useful in your research, please consider citing the following paper::

    @misc{barbiero2020topological,
        title={Topological Gradient-based Competitive Learning},
        author={Pietro Barbiero and Gabriele Ciravegna and Vincenzo Randazzo and Giansalvo Cirrincione},
        year={2020},
        eprint={2008.09477},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }

Examples
----------

Dual Competitive Layer (DCL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

    * - .. figure:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/test/test-results/circles_dynamic_dual.png
            :height: 200px

      - .. image:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/test/test-results/circles_scatter_dual.png
            :height: 200px


Vanilla Competitive Layer (VCL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::

    * - .. figure:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/test/test-results/circles_dynamic_vanilla.png
            :height: 200px

      - .. image:: https://github.com/pietrobarbiero/deep-topological-learning/blob/master/test/test-results/circles_scatter_vanilla.png
            :height: 200px



Using COLA
---------------

.. code:: python

    from cola import DualModel, plot_confusion_matrix, scatterplot, scatterplot_dynamic

    X, y = ... # load dataset

    # load custom tensorflow layers
    inputs = Input(shape=(d,), name='input')
    ...
    outputs = ...

    # instantiate the dual model
    n = X.shape[0] # number of samples
    k = ... # upper bound of the desired number of prototypes
    model = DualModel(n_samples=n, k_prototypes=k, inputs=inputs, outputs=outputs, deep=False)
    model.compile(optimizer=optimizer)
    model.fit(X, y, epochs=epochs)

    # plot prototype dynamics
    plt.figure()
    scatterplot_dynamic(X, model.prototypes_, y, valid=True)
    plt.show()

    # plot confusion matrix
    # considering the prototypes estimated in the last epoch
    plt.figure()
    plot_confusion_matrix(x_pred, model.prototypes[-1], y)
    plt.show()

    # plot estimated topology
    # considering the prototypes estimated in the last epoch
    plt.figure()
    scatterplot(x_pred, model.prototypes[-1], y, valid=True)
    plt.show()



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