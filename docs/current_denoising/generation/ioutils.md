ioutils
====
Additional documentation for the io utils in `current_denoising/`.


read_clean_currents
----
This function is more complicated that `read_currents` (which just reads in a whole dat file
and is for reading in noisy currents). The dat files containing the (clean) CMIP simulation
hold the data for many different runs, models, and start years, so the correct one needs to be
specified. This is done with the year/model/name parameters; the name is a special string (e.g.
r1i1p1f1_gn).

It is assumed that the currents don't change much within 5 years, so we have model outputs at
a granularity of every 5 years. The provided year in the metadata is the start year of the run.

#### r (realization_index)
    an integer (≥1) distinguishing among members of an ensemble of simulations that
    differ only in their initial conditions (e.g., initialized from different points
    in a control run). Note that if two different simulations were started from the
    same initial conditions, the same realization number should be used for both simulations.
    For example if a historical run with “natural forcing” only and another historical
    run that includes anthropogenic forcing were both spawned at the same point in a
    control run, both should be assigned the same realization.  Also, each so-called
    RCP (future scenario) simulation should normally be assigned the same realization
    integer as the historical run from which it was initiated.
    This will allow users to easily splice together the appropriate historical and future runs.

#### i (initialization_index)
    an integer (≥1), which should be assigned a value of 1 except to distinguish
    simulations performed under the same conditions but with different initialization
    procedures.  In CMIP6 this index should invariably be assigned the value “1”
    except for some hindcast and forecast experiments called for by the DCPP activity.
    The initialization_index can be used either to distinguish between different
    algorithms used to impose initial conditions on a forecast or to distinguish
    between different observational datasets used to initialize a forecast.

#### p (physics_index)
    an integer (≥1) identifying the physics version used by the model.  In the
    usual case of a single physics version of a model, this argument should normally
    be assigned the value 1, but it is essential that a consistent assignment of
    physics_index be used across all simulations performed by a particular model.
    Use of  “physics_index” is reserved for closely-related model versions (e.g., as
    in a “perturbed physics” ensemble) or for the same model run with slightly
    different parameterizations (e.g., of cloud physics).  Model versions that are
    substantially different from one another should be given a different source_id”
    (rather than simply assigning a different value of the physics_index).

#### f (forcing_index)
    an integer (≥1) used to distinguish runs conforming to the protocol of a single
    CMIP6 experiment, but with different variants of forcing applied.  One can, for
    example, distinguish between two historical simulations, one forced with the
    CMIP6-recommended forcing data sets and another forced by a different dataset,
    which might yield information about how forcing uncertainty affects the simulation.

#### Gridding
    A grid-label suffix is used to distinguish between the gridding conventions used:
 - grid_label = "gn"  (output is reported on the native grid, usually but not invariably at grid cell centers) 
 - grid_label = "gr"   (output is not reported on the native grid, but instead is regridded by the modeling group to a “primary grid” of its choosing) 
 - grid_label = “gm” (global mean output is reported, so data are not gridded)
