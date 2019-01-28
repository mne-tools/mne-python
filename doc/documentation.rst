:orphan:

.. _documentation:

Documentation
=============

.. raw:: html

    <style class='text/css'>
    .panel-title a {
        display: block;
        padding: 5px;
        text-decoration: none;
    }

    .plus {
        float: right;
        color: #212121;
    }

    .panel {
        margin-bottom: 3px;
    }

    .example_details {
        padding-left: 20px;
        margin-bottom: 10px;
    }
    </style>

    <script type="text/javascript">
    $(document).ready(function () {
        if(location.hash != null && location.hash != ""){
            $('.collapse').removeClass('in');
            $(location.hash + '.collapse').addClass('in');
        }
    });
    </script>

.. toctree::
    :maxdepth: 1

    manual/cookbook.rst
    tutorials.rst
    auto_examples/index.rst
    faq.rst
    python_reference.rst
    generated/commands.rst
    glossary.rst
    auto_tutorials/plot_configuration.rst
    whats_new.rst
    cited.rst

This is where you can learn about all the things you can do with MNE. It contains **background information** and **tutorials** for taking a deep-dive into the techniques that MNE-python covers. You'll find practical information on how to use these methods with your data, and in many cases some high-level concepts underlying these methods.

There are also **examples**, which contain a short use-case to highlight MNE-functionality and provide inspiration for the many things you can do with this package. You can also find a gallery of these examples in the :ref:`examples gallery <sphx_glr_auto_examples>`.


Workflows
==========================


The documentation in mne-python provides information in the form of: 
•	Workflows
•	Tutorials
•	Example codes
•	MNE-study template 

The workflows will help you get you started on defining which type of pipeline you should aim for in your data analysis. 
Each step in the workflows will link to specific tutorials and example codes.

The tutorials provide in-depth review of particular analysis steps (e.g. preprocessing, epoching, decoding). 
The example codes provides you elaborate snippet codes that help you go through raw data to the endpoint.
