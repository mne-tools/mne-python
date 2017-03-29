:orphan:

.. _inside_martinos:

Martinos Center setup
---------------------

For people within the MGH/MIT/HMS Martinos Center, MNE is available on the network.

In a terminal do:

.. code-block:: bash

    $ setenv PATH /usr/pubsw/packages/python/anaconda/bin:${PATH}

If you use Bash replace the previous instruction with:

.. code-block:: bash

    $ export PATH=/usr/pubsw/packages/python/anaconda/bin:${PATH}

Then start the python interpreter with:

.. code-block:: bash

    $ ipython

Then type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go.

We encourage all Martinos center Python users to subscribe to the
`Martinos Python mailing list`_.

.. _Martinos Python mailing list: https://mail.nmr.mgh.harvard.edu/mailman/listinfo/martinos-python
