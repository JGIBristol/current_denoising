Integration Tests
====

I haven't been too strict with the definitions here (technically a unit test
should test a single unit of functionality, but most of my "unit tests" test ore
than that), but some of the code here tests larger chunks or has more dependencies than others - I've put those here.

The practical difference is that the unit tests only require a minimal
subset of the python requirements, but some of the tests here require larger 
packages (notably, pytorch). This means I don't want to run them in the CI,
on every branch but they might be useful to run from the CLI or less frequently
(nightly/on push to main/on PR etc.)
