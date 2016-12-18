# APL module for Python
A Python module using Numpy for programming in APL.

This module will implement a large subset of the APL language in Python by using the very efficient vectorized procedures from the well known Numpy module. Once mature enough, the project should provide:

  * a Python module for performing APL-oriented operations on Numpy arrays (while using its own extended class, the module is fully compatible with native Numpy arrays);
  * a parser for interpreting pieces of APL code (as strings) from a Python program, allowing both to do some computations on arrays or to define standalone functions in APL for later use in Python;
  * a REPL-based APL interpreter;
  * an APL to Python translator.

Since the module relies on Numpy, it does not attempt to implement the full APL standard. Strings for instance are not at all implemented here. Nested arrays are implemented as long as they have a regular shape and don't mix different types. The philosophy of the module is to get all benefits of using Numpy types, rather than following tightly the standard. People who need all APL features should rather use existing softwares.

Purposes of the module are:

  * allowing Python programmers with no APL background to use some functions in their code when the APL philosophy is more convenient than the Numpy one (of course, the same thing could be achieved with Numpy, but some operations require complicated broadcasting rules in Numpy and may be more natural in APL);
  * allowing APL programmers to make some computations in APL with no significant loss in speed and then access other Python features (for instance `scipy`, `matplotlib`, etc.);

## Looking for contributors

Right now, I am focusing on internal functions. I wrote a (working) lexer but I don't have the time to focus on the parser now.

I would be happy to have people trying the code and report the bugs.

I would be very happy to have some contributors writing some documentation. The Wiki page on Github would probably be a good place. Writing a nice page in docs/ which is displayed at https://baruchel.github.io/apl/ would be nice also. Creating a logo would be helpful also.

A small but useful contribution would be to re-write the `__repr__` method in apl/internal.py in order to have a better display for the arrays.

I created a mailing list at https://groups.io/g/apl/

There is no documentation yet, but the root directory contains some demo files.
