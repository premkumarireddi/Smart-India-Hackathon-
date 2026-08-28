# ADR 0004: Erlang-C queueing model instead of a Simulink discrete-event simulation

## Status
Accepted

## Context
PS26038 asks for a Simulink model of the screening pipeline — image
acquisition rates, bandwidth constraints, processing throughput, and review
capacity — to size a district-level rollout. We have no MATLAB/Simulink
license in this environment (see ADR 0001).

## Decision
Model district-level screening as a multi-server queue: patients arrive as
a Poisson process at rate `lambda`, `c` parallel screening stations
(tablet + camera + this AI pipeline) each serve at rate `mu`. This is
exactly the system a Simulink discrete-event model with queue/server blocks
would simulate numerically; we instead solve it with the closed-form
Erlang-C formula (`simulation/screening_throughput.py`), which is faster,
deterministic (no simulation-seed variance to explain), and requires no
extra dependency.

The script answers the same operational question the PS asks for: "how
many stations does a district need to serve 100,000+ patients/year within
a target wait time?" — see `find_min_stations()`.

## Consequences
- Positive: instant, reproducible answers; no MATLAB required; the model
  parameters (arrival rate, service rate, station count) map directly onto
  Simulink queue/server block parameters, so porting the *model* to an
  actual `.slx` file later is a translation exercise, not a redesign.
- Negative: Erlang-C assumes steady-state, infinite-queue-capacity,
  Poisson arrivals — real deployments have daily/seasonal arrival
  variation, finite waiting rooms, and non-exponential service times a full
  discrete-event Simulink model could capture more faithfully. Flagged as
  a roadmap item in the README.
- We deliberately did not adopt a generic Python discrete-event library
  (e.g. SimPy) because the closed-form solution is exact and simpler for
  this steady-state question; a SimPy/Simulink model becomes worth the
  added complexity once we need time-varying arrival patterns (e.g. camps
  concentrated on specific days) rather than an average annual rate.
