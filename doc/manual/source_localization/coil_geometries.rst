:orphan: 

.. _implemented_coil_geometries:

Implemented coil geometries
===========================

This section describes the coil geometries currently implemented
in MNE. The coil types fall in two general categories:

- Axial gradiometers and planar gradiometers
  and

- Planar magnetometers.

For axial sensors, the *z* axis of the
local coordinate system is parallel to the field component detected, *i.e.*,
normal to the coil plane.For circular coils, the orientation of
the *x* and *y* axes on the
plane normal to the z axis is irrelevant. In the square coils employed
in the Vectorview (TM) system the *x* axis
is chosen to be parallel to one of the sides of the magnetometer
coil. For planar sensors, the *z* axis is likewise
normal to the coil plane and the x axis passes through the centerpoints
of the two coil loops so that the detector gives a positive signal
when the normal field component increases along the *x* axis.

:ref:`BGBBHGEC` lists the parameters of the *normal* coil
geometry descriptions :ref:`CHDBDFJE` lists the *accurate* descriptions. For simple accuracy,
please consult the coil definition file, see :ref:`BJECIGEB`.
The columns of the tables contain the following data:

- The number identifying the coil id.
  This number is used in the coil descriptions found in the FIF files.

- Description of the coil.

- Number of integration points used

- The locations of the integration points in sensor coordinates.

- Weights assigned to the field values at the integration points.
  Some formulas are listed instead of the numerical values to demonstrate
  the principle of the calculation. For example, in the normal coil
  descriptions of the planar gradiometers the weights are inverses
  of the baseline of the gradiometer to show that the output is in
  T/m.

.. note:: The coil geometry information is stored in the file $MNE_ROOT/mne/coil_def.dat

.. XXX : table of normal coil description is missing

.. tabularcolumns:: |p{0.1\linewidth}|p{0.3\linewidth}|p{0.1\linewidth}|p{0.25\linewidth}|p{0.2\linewidth}|
.. _BGBBHGEC:
.. table:: Normal coil descriptions.

    +------+-------------------------+----+----------------------------------+----------------------+
    | Id   | Description             | n  | r/mm                             | w                    |
    +======+=========================+====+==================================+======================+
    | 2    | Neuromag-122            | 2  | (+/-8.1, 0, 0) mm                | +/-1 ⁄ 16.2mm        | 
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 2000 | A point magnetometer    | 1  | (0, 0, 0)mm                      | 1                    |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3012 | Vectorview type 1       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3013 | Vectorview type 2       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3022 | Vectorview type 1       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3023 | Vectorview type 2       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3024 | Vectorview type 3       | 4  | (+/-5.25, +/-5.25, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 2000 | An ideal point          | 1  | (0.0, 0.0, 0.0)mm                | 1                    |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4001 | Magnes WH               | 4  | (+/-5.75, +/-5.75, 0.0)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4002 | Magnes WH 3600          | 8  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | axial gradiometer       |    | (+/-4.5, +/-4.5, 50.0)mm         | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4003 | Magnes reference        | 4  | (+/-7.5, +/-7.5, 0.0)mm          | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4004 | Magnes reference        | 8  | (+/-20, +/-20, 0.0)mm            | 1/4                  |
    |      | gradiometer measuring   |    | (+/-20, +/-20, 135)mm            | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4005 | Magnes reference        | 8  | (87.5, +/-20, 0.0)mm             | 1/4                  |
    |      | gradiometer measuring   |    | (47.5, +/-20, 0.0)mm             | -1/4                 |
    |      | off-diagonal gradients  |    | (-87.5, +/-20, 0.0)mm            | 1/4                  |
    |      |                         |    | (-47.5, +/-20, 0.0)mm            | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5001 | CTF 275 axial           | 8  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | gradiometer             |    | (+/-4.5, +/-4.5, 50.0)mm         | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5002 | CTF reference           | 4  | (+/-4, +/-4, 0.0)mm              | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5003 | CTF reference           | 8  | (+/-8.6, +/-8.6, 0.0)mm          | 1/4                  |
    |      | gradiometer measuring   |    | (+/-8.6, +/-8.6, 78.6)mm         | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+

.. note:: If a plus-minus sign occurs in several coordinates, all possible combinations have to be included.

.. tabularcolumns:: |p{0.1\linewidth}|p{0.3\linewidth}|p{0.05\linewidth}|p{0.25\linewidth}|p{0.15\linewidth}|
.. _CHDBDFJE:
.. table:: Accurate coil descriptions

    +------+-------------------------+----+----------------------------------+----------------------+
    | Id   | Description             | n  | r/mm                             | w                    |
    +======+=========================+====+==================================+======================+
    | 2    | Neuromag-122 planar     | 8  | +/-(8.1, 0, 0) mm                | +/-1 ⁄ 16.2mm        |
    |      | gradiometer             |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 2000 | A point magnetometer    | 1  | (0, 0, 0) mm                     | 1                    |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3012 | Vectorview type 1       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3013 | Vectorview type 2       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3022 | Vectorview type 1       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3023 | Vectorview type 2       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3024 | Vectorview type 3       | 4  | (+/-5.25, +/-5.25, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4001 | Magnes WH magnetometer  | 4  | (+/-5.75, +/-5.75, 0.0)mm        | 1/4                  |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4002 | Magnes WH 3600          | 4  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | axial gradiometer       |    | (+/-4.5, +/-4.5, 0.0)mm          | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4004 | Magnes reference        | 8  | (+/-20, +/-20, 0.0)mm            | 1/4                  |
    |      | gradiometer measuring   |    | (+/-20, +/-20, 135)mm            | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4005 | Magnes reference        | 8  | (87.5, +/-20, 0.0)mm             | 1/4                  |
    |      | gradiometer measuring   |    | (47.5, +/-20, 0.0)mm             | -1/4                 |
    |      | off-diagonal gradients  |    | (-87.5, +/-20, 0.0)mm            | 1/4                  |
    |      |                         |    | (-47.5, +/-20, 0.0)mm            | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5001 | CTF 275 axial           | 8  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | gradiometer             |    | (+/-4.5, +/-4.5, 50.0)mm         | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5002 | CTF reference           | 4  | (+/-4, +/-4, 0.0)mm              | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5003 | CTF 275 reference       | 8  | (+/-8.6, +/-8.6, 0.0)mm          | 1/4                  |
    |      | gradiometer measuring   |    | (+/-8.6, +/-8.6, 78.6)mm         | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5004 | CTF 275 reference       | 8  | (47.8, +/-8.5, 0.0)mm            | 1/4                  |
    |      | gradiometer measuring   |    | (30.8, +/-8.5, 0.0)mm            | -1/4                 |
    |      | off-diagonal gradients  |    | (-47.8, +/-8.5, 0.0)mm           | 1/4                  |
    |      |                         |    | (-30.8, +/-8.5, 0.0)mm           | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 6001 | MIT KIT system axial    | 8  | (+/-3.875, +/-3.875, 0.0)mm      | 1/4                  |
    |      | gradiometer             |    | (+/-3.875, +/-3.875, 0.0)mm      | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+


.. _BJECIGEB:

The coil definition file
========================

The coil geometry information is stored in the text file
$MNE_ROOT/share/mne/coil_def.dat. In this file, any lines starting
with the pound sign (#) are comments. A coil definition starts with
a description line containing the following fields:

** <*class*>**

    This is a number indicating class of this coil. Possible values
    are listed in :ref:`BJEFABHA`.

** <*id*>**

    Coil id value. This value is listed in the first column of Tables :ref:`BGBBHGEC` and :ref:`CHDBDFJE`.

** <*accuracy*>**

    The coil representation accuracy. Possible values and their meanings
    are listed in :ref:`BJEHIBJC`.

** <*np*>**

    Number of integration points in this representation.

** <*size/m*>**

    The size of the coil. For circular coils this is the diameter of
    the coil and for square ones the side length of the square. This
    information is mainly included to facilitate drawing of the coil
    geometry. It should not be employed to infer a coil approximation
    for the forward calculations.

** <*baseline/m*>**

    The baseline of a this kind of a coil. This will be zero for magnetometer
    coils. This information is mainly included to facilitate drawing
    of the coil geometry. It should not be employed to infer a coil
    approximation for the forward calculations.

** <*description*>**

    Short description of this kind of a coil. If the description contains several
    words, it is enclosed in quotes.

.. _BJEFABHA:

.. table:: Coil class values

    =======  =======================================================
    Value    Meaning
    =======  =======================================================
    1        magnetometer
    2        first-order axial gradiometer
    3        planar gradiometer
    4        second-order axial gradiometer
    1000     an EEG electrode (used internally in software only).
    =======  =======================================================


.. tabularcolumns:: |p{0.1\linewidth}|p{0.5\linewidth}|
.. _BJEHIBJC:
.. table:: Coil representation accuracies.

    =======  =====================================================================
    Value    Meaning
    =======  =====================================================================
    1        The simplest representation available
    2        The standard or *normal* representation (see :ref:`BGBBHGEC`)
    3        The most *accurate* representation available (see :ref:`CHDBDFJE`)
    =======  =====================================================================

Each coil description line is followed by one or more integration
point lines, consisting of seven numbers:

** <*weight*>**

    Gives the weight for this integration point (last column in Tables :ref:`BGBBHGEC` and :ref:`CHDBDFJE`).

** <*x/m*> <*y/m*> <*z/m*>**

    Indicates the location of the integration point (fourth column in Tables :ref:`BGBBHGEC` and :ref:`CHDBDFJE`).

** <*nx*> <*ny*> <*nz*>**

    Components of a unit vector indicating the field component to be selected.
    Note that listing a separate unit vector for each integration points
    allows the implementation of curved coils and coils with the gradiometer
    loops tilted with respect to each other.
