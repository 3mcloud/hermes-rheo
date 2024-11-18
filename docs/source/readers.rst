Readers
==========

The `file_readers <https://github.com/3mcloud/hermes-rheo/tree/main/src/hermes_rheo/file_readers>`_
directory contains an example file reader for rheological data collected using
`TA TRIOS software <https://www.tainstruments.com/trios-software>`_.

This reader was designed to read `.txt` files generated via the "Export to LIMS" command in TRIOS.
Starting with TRIOS V5, a new export format, `.json`, has been introduced.
`See new features here <https://www.tainstruments.com/wp-content/uploads/NewFeaturesTRIOS.pdf>`_.
A reader for this format is currently in development.

For other formats or instruments, users can develop custom readers while still utilizing
the package's data transforms for analysis. For assistance in developing a reader for your data format,
please contact aperego[at]mmm.com.
