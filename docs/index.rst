:github_url: https://github.com/kevinzakka/mink/tree/main/docs/index.rst

.. title:: Table of Contents

####
mink
####

mink is a library for differential inverse kinematics in Python, based on the `MuJoCo <https://github.com/google-deepmind/mujoco>`_ physics engine.

.. image:: https://github.com/kevinzakka/mink/blob/assets/banner.png?raw=true
   :alt: Banner for mink

------------
Installation
------------

Install from PyPI:

.. code:: bash

   uv add mink

Or clone and run locally:

.. code:: bash

   git clone https://github.com/kevinzakka/mink.git && cd mink
   uv sync

.. toctree::
    :maxdepth: 1

    configuration.rst
    lie.rst
    tasks.rst
    limits.rst
    inverse_kinematics.rst
    utilities.rst
    derivations.rst
    references.rst
