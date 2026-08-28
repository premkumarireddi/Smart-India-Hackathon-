"""
District-scale screening-throughput model — the Python stand-in for the
Simulink workflow simulation called for in PS26038 ("Model the telemedicine
screening pipeline in Simulink - image acquisition rates, bandwidth
constraints, processing throughput, and review capacity - to optimize
resource allocation for district-level programs serving 100,000+ patients
annually").

WHY PYTHON, NOT SIMULINK: this dev environment has no MATLAB/Simulink
license. Rather than skip the requirement, this module implements the same
underlying queueing-theory model (M/M/c multi-server queue) that a Simulink
discrete-event simulation would numerically approximate, using closed-form
Erlang-C math. See docs/adr/0004. Porting this to an actual .slx model is
listed as a "next step" in the README/roadmap — the parameters and outputs
here are designed to map directly onto Simulink queue/server blocks.

Model:
    - Patients arrive for screening at rate `lambda` (patients/hour), modeled
      as a Poisson process — standard assumption for patient-arrival
      processes in healthcare operations research.
    - Each of `c` screening stations (a tablet/camera + the AI pipeline)
      processes one patient in mean time `1/mu` hours.
    - We report utilization, mean wait time, and P(wait > target), then
      search for the minimum number of stations needed to hit a target wait
      time — directly answering "how many stations does a district need?"
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass


def _erlang_c(c: int, a: float) -> float:
    """Probability an arriving patient has to wait (Erlang-C formula)."""
    if a >= c:
        return 1.0  # unstable system, queue grows without bound
    sum_terms = sum((a ** k) / math.factorial(k) for k in range(c))
    last_term = (a ** c) / (math.factorial(c) * (1 - a / c))
    return last_term / (sum_terms + last_term)


@dataclass
class ThroughputResult:
    stations: int
    arrival_rate_per_hour: float
    service_rate_per_hour: float
    offered_load_erlangs: float
    utilization: float
    prob_wait: float
    mean_wait_minutes: float
    mean_patients_in_queue: float
    stable: bool


def simulate(stations: int, patients_per_year: int, hours_per_day: float,
             days_per_year: int, seconds_per_screening: float) -> ThroughputResult:
    arrival_rate = patients_per_year / (days_per_year * hours_per_day)  # patients/hour
    service_rate = 3600.0 / seconds_per_screening  # patients/hour/station
    offered_load = arrival_rate / service_rate  # in Erlangs
    utilization = offered_load / stations
    stable = utilization < 1.0

    if not stable:
        return ThroughputResult(
            stations=stations,
            arrival_rate_per_hour=round(arrival_rate, 2),
            service_rate_per_hour=round(service_rate, 2),
            offered_load_erlangs=round(offered_load, 2),
            utilization=round(utilization, 3),
            prob_wait=1.0,
            mean_wait_minutes=float("inf"),
            mean_patients_in_queue=float("inf"),
            stable=False,
        )

    p_wait = _erlang_c(stations, offered_load)
    mean_wait_hours = p_wait / (stations * service_rate - arrival_rate)
    mean_queue_len = arrival_rate * mean_wait_hours

    return ThroughputResult(
        stations=stations,
        arrival_rate_per_hour=round(arrival_rate, 2),
        service_rate_per_hour=round(service_rate, 2),
        offered_load_erlangs=round(offered_load, 2),
        utilization=round(utilization, 3),
        prob_wait=round(p_wait, 3),
        mean_wait_minutes=round(mean_wait_hours * 60, 2),
        mean_patients_in_queue=round(mean_queue_len, 2),
        stable=True,
    )


def find_min_stations(patients_per_year: int, hours_per_day: float, days_per_year: int,
                       seconds_per_screening: float, target_wait_minutes: float,
                       max_stations: int = 200) -> ThroughputResult:
    for c in range(1, max_stations + 1):
        result = simulate(c, patients_per_year, hours_per_day, days_per_year, seconds_per_screening)
        if result.stable and result.mean_wait_minutes <= target_wait_minutes:
            return result
    raise RuntimeError(f"No solution found under {max_stations} stations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients-per-year", type=int, default=100_000,
                         help="District-scale target from PS26038")
    parser.add_argument("--hours-per-day", type=float, default=8.0)
    parser.add_argument("--days-per-year", type=int, default=300)
    parser.add_argument("--seconds-per-screening", type=float, default=45.0,
                         help="End-to-end time per patient: capture + quality-gate + AI inference + review")
    parser.add_argument("--target-wait-minutes", type=float, default=10.0)
    args = parser.parse_args()

    best = find_min_stations(
        args.patients_per_year, args.hours_per_day, args.days_per_year,
        args.seconds_per_screening, args.target_wait_minutes,
    )
    print(f"Target: {args.patients_per_year:,} patients/year, "
          f"<= {args.target_wait_minutes} min average wait\n")
    print(json.dumps(asdict(best), indent=2))
    print(f"\n=> Recommendation: deploy {best.stations} screening stations "
          f"(tablet + portable fundus camera + this AI pipeline each) "
          f"per district to hit the target.")

    print("\nSensitivity table (stations vs. mean wait):")
    print(f"{'stations':>8} | {'utilization':>11} | {'mean_wait_min':>14} | {'stable':>6}")
    for c in range(max(1, best.stations - 3), best.stations + 4):
        r = simulate(c, args.patients_per_year, args.hours_per_day, args.days_per_year, args.seconds_per_screening)
        wait_display = f"{r.mean_wait_minutes:.2f}" if r.stable else "inf"
        print(f"{c:>8} | {r.utilization:>11.3f} | {wait_display:>14} | {str(r.stable):>6}")
