"""
pytest tests for the district-scale throughput model (ADR 0004).
Run from the `simulation/` directory (or repo root with the right rootdir):
    pytest simulation/
"""
import math

from screening_throughput import find_min_stations, simulate


def test_single_station_utilization_matches_manual_calc():
    # 8760 patients/year, 1 patient/hour service rate, 1 station,
    # running 24h/day * 365 days -> offered load exactly 1 Erlang / 1 station.
    result = simulate(
        stations=1, patients_per_year=8760, hours_per_day=24,
        days_per_year=365, seconds_per_screening=3600,
    )
    assert math.isclose(result.arrival_rate_per_hour, 1.0, rel_tol=1e-3)
    assert math.isclose(result.service_rate_per_hour, 1.0, rel_tol=1e-3)
    assert result.stable is False  # utilization == 1.0 is not stable (queue explodes)


def test_more_stations_reduces_wait_time():
    kwargs = dict(patients_per_year=100_000, hours_per_day=8, days_per_year=300, seconds_per_screening=45)
    r1 = simulate(stations=1, **kwargs)
    r2 = simulate(stations=2, **kwargs)
    assert r2.mean_wait_minutes <= r1.mean_wait_minutes
    assert r2.utilization < r1.utilization


def test_unstable_system_flagged_when_understaffed():
    # Way more patients than a single slow station can handle.
    result = simulate(
        stations=1, patients_per_year=1_000_000, hours_per_day=8,
        days_per_year=300, seconds_per_screening=300,
    )
    assert result.stable is False
    assert result.mean_wait_minutes == float("inf")


def test_find_min_stations_meets_target_wait():
    best = find_min_stations(
        patients_per_year=100_000, hours_per_day=8, days_per_year=300,
        seconds_per_screening=45, target_wait_minutes=10.0,
    )
    assert best.stable is True
    assert best.mean_wait_minutes <= 10.0
    # one fewer station should NOT meet the target (proves we found the minimum)
    if best.stations > 1:
        one_less = simulate(
            stations=best.stations - 1, patients_per_year=100_000,
            hours_per_day=8, days_per_year=300, seconds_per_screening=45,
        )
        assert (not one_less.stable) or one_less.mean_wait_minutes > 10.0
