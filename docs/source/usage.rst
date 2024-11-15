.. _usage:

Usage
=====

## Importing the Package

Once the package has been installed, you can import its modules and start using them:

.. code-block:: python

   from hermes_rheo.transforms.rheo_analysis import RheoAnalysis

## File Readers

The [file_readers](https://github.com/3mcloud/hermes-rheo/tree/main/src/hermes_rheo/file_readers) directory contains example file readers for rheological data collected using [TA TRIOS software](https://www.tainstruments.com/trios-software).

### Supported Formats
- **TRIOS V4**: Reads `.txt` files exported via the "Export to LIMS" command.
- **TRIOS V5**: Introduces a new `.json` export format. See [new features here](https://www.tainstruments.com/wp-content/uploads/NewFeaturesTRIOS.pdf). A reader for this format is currently in development.

### Custom Readers
For other formats or instruments, users can develop custom readers and still leverage the package's data transforms for analysis. For assistance in developing a reader for your specific data format, contact aperego[at]mmm.com.
