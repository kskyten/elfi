API
===
This file describes the classes and methods available in ELFI.

Modelling API
-------------
Below is the API for creating generative models.

.. autosummary::
   elfi.ElfiModel

**General model nodes**

.. autosummary::
   elfi.Constant
   elfi.Operation
   elfi.RandomVariable

**LFI nodes**

.. autosummary::
   elfi.Prior
   elfi.Simulator
   elfi.Summary
   elfi.Discrepancy
   elfi.Distance

**Other**

.. currentmodule:: elfi.model.elfi_model

.. autosummary::
   elfi.get_current_model
   elfi.set_current_model

.. currentmodule:: elfi.visualization.visualization

.. autosummary::
   elfi.draw

Inference API
-------------

Below is a list of inference methods included in ELFI.

.. autosummary::
   elfi.Rejection
   elfi.SMC
   elfi.BayesianOptimization
   elfi.BOLFI

**Result objects**

.. currentmodule:: elfi.methods.results

.. autosummary::
   Result
   ResultSMC
   ResultBOLFI

Other
-----

**Data pools**

.. autosummary::
   elfi.OutputPool
   elfi.ArrayPool


**Module functions**

.. currentmodule:: elfi

.. autosummary::
   elfi.get_client
   elfi.set_client


**Tools**

.. currentmodule:: elfi.model.tools

.. autosummary::
   elfi.tools.vectorize
   elfi.tools.external_operation


Class documentations
--------------------

Modelling API classes
.....................

.. autoclass:: elfi.ElfiModel
   :members:
   :inherited-members:

.. autoclass:: elfi.Constant
   :members:
   :inherited-members:

.. autoclass:: elfi.Operation
   :members:
   :inherited-members:

.. autoclass:: elfi.RandomVariable
   :members:
   :inherited-members:

.. autoclass:: elfi.Prior
   :members:
   :inherited-members:

.. autoclass:: elfi.Simulator
   :members:
   :inherited-members:

.. autoclass:: elfi.Summary
   :members:
   :inherited-members:

.. autoclass:: elfi.Discrepancy
   :members:
   :inherited-members:

.. autoclass:: elfi.Distance
   :members:
   :inherited-members:


**Other**

.. currentmodule:: elfi.model.elfi_model

.. automethod:: elfi.get_current_model

.. automethod:: elfi.set_current_model

.. currentmodule:: elfi.visualization.visualization

.. automethod:: elfi.visualization.visualization.nx_draw

.. This would show undocumented members :undoc-members:


Inference API classes
.....................

.. autoclass:: elfi.Rejection
   :members:
   :inherited-members:

.. autoclass:: elfi.SMC
   :members:
   :inherited-members:

.. autoclass:: elfi.BayesianOptimization
   :members:
   :inherited-members:

.. autoclass:: elfi.BOLFI
   :members:
   :inherited-members:

.. currentmodule:: elfi.methods.results

.. autoclass:: Result
   :members:
   :inherited-members:

.. autoclass:: ResultSMC
   :members:
   :inherited-members:

.. autoclass:: ResultBOLFI
   :members:
   :inherited-members:


Other
.....

**Data pools**

.. autoclass:: elfi.OutputPool
   :members:
   :inherited-members:

.. autoclass:: elfi.ArrayPool
   :members:
   :inherited-members:


**Module functions**

.. currentmodule:: elfi

.. automethod:: elfi.get_client

.. automethod:: elfi.set_client


**Tools**

.. currentmodule:: elfi.model.tools

.. automethod:: elfi.tools.vectorize

.. automethod:: elfi.tools.external_operation