Showroom Demos
==============

The main core interface extension in Isaac Lab ``isaaclab`` provides
the main modules for actuators, objects, robots and sensors. We provide
a list of demo scripts and tutorials. These showcase how to use the provided
interfaces within a code in a minimal way.

A few quick showroom scripts to run and checkout:

-  Spawn different quadrupeds and make robots stand using position commands:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/quadrupeds.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\quadrupeds.py

   .. image:: ../_static/demos/quadrupeds.jpg
      :width: 100%
      :alt: Quadrupeds in Isaac Lab

-  Spawn different arms and apply random joint position commands:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/arms.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\arms.py

   .. image:: ../_static/demos/arms.jpg
      :width: 100%
      :alt: Arms in Isaac Lab

-  Spawn different hands and command them to open and close:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/hands.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\hands.py

   .. image:: ../_static/demos/hands.jpg
      :width: 100%
      :alt: Dexterous hands in Isaac Lab

-  Spawn different deformable (soft) bodies and let them fall from a height:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/deformables.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\deformables.py

   .. image:: ../_static/demos/deformables.jpg
      :width: 100%
      :alt: Deformable primitive-shaped objects in Isaac Lab

-  Use the interactive scene and spawn varying assets in individual environments:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/multi_asset.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\multi_asset.py

   .. image:: ../_static/demos/multi_asset.jpg
      :width: 100%
      :alt: Multiple assets managed through the same simulation handles

-  Create and spawn procedurally generated terrains with different configurations:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/procedural_terrain.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\procedural_terrain.py

   .. image:: ../_static/demos/procedural_terrain.jpg
      :width: 100%
      :alt: Procedural Terrains in Isaac Lab

-  Define multiple markers that are useful for visualizations:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/markers.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\markers.py

   .. image:: ../_static/demos/markers.jpg
      :width: 100%
      :alt: Markers in Isaac Lab

-  Interactive inference of trained H1 rough terrain locomotion policy:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/h1_locomotion.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\h1_locomotion.py

   .. image:: ../_static/demos/h1_locomotion.jpg
      :width: 100%
      :alt: H1 locomotion in Isaac Lab

   This is an interactive demo that can be run using the mouse and keyboard.
   To enter third-person perspective, click on a humanoid character in the scene.
   Once entered into third-person view, the humanoid can be controlled by keyboard using:

   * ``UP``: go forward
   * ``LEFT``: turn left
   * ``RIGHT``: turn right
   * ``DOWN``: stop
   * ``C``: switch between third-person and perspective views
   * ``ESC``: exit current third-person view

   If a misclick happens outside of the humanoid bodies when selecting a humanoid,
   a message is printed to console indicating the error, such as
   ``The selected prim was not a H1 robot`` or
   ``Multiple prims are selected. Please only select one!``.
